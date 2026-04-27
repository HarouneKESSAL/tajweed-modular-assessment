from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from content.train_content import (
    ManifestContentDataset,
    collate_content_batch,
    normalize_text_target,
    split_content_indices,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs
from tajweed_assessment.models.common.decoding import ctc_prefix_beam_search
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint


def decode_ids(ids, id_to_char):
    return "".join(id_to_char.get(int(i), "") for i in ids)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    dist = levenshtein(gold, pred)
    return max(0.0, 1.0 - (dist / len(gold)))


def collapse_excess_repetitions(text: str, max_run: int = 2) -> str:
    if not text:
        return text
    out = [text[0]]
    run_char = text[0]
    run_len = 1
    for ch in text[1:]:
        if ch == run_char:
            run_len += 1
            if run_len <= max_run:
                out.append(ch)
        else:
            run_char = ch
            run_len = 1
            out.append(ch)
    return "".join(out)


def content_postprocess(text: str) -> str:
    # Conservative cleanup for repeated-character bursts observed in CTC outputs.
    return collapse_excess_repetitions(text, max_run=2)


def repeated_char_error_count(gold: str, pred: str) -> int:
    count = 0
    i = 1
    while i < len(pred):
        if pred[i] == pred[i - 1]:
            if i >= len(gold) or pred[i] != gold[i]:
                count += 1
        i += 1
    return count


@dataclass
class EvalRow:
    gold: str
    pred: str
    exact: bool
    char_acc: float
    edit_distance: int
    repeated_errors: int


def summarize(rows):
    total = len(rows)
    exact = sum(1 for row in rows if row.exact)
    mean_char_acc = sum(row.char_acc for row in rows) / max(total, 1)
    mean_edit = sum(row.edit_distance for row in rows) / max(total, 1)
    return {
        "count": total,
        "exact_match": exact / max(total, 1),
        "char_accuracy": mean_char_acc,
        "edit_distance": mean_edit,
        "repeated_errors": sum(row.repeated_errors for row in rows) / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_quranjson_train.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_module.pt")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_ssl_cache")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--show-misses", type=int, default=10)
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    dataset = ManifestContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        n_mfcc=data_cfg["n_mfcc"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )

    train_idx, val_idx = split_content_indices(dataset.rows, val_fraction=0.2, seed=train_cfg["seed"])
    if args.split == "train":
        indices = train_idx
    elif args.split == "val":
        indices = val_idx
    else:
        indices = list(range(len(dataset)))

    subset = Subset(dataset, indices[: args.limit])
    loader = DataLoader(subset, batch_size=min(train_cfg["batch_size"], max(1, args.limit)), shuffle=False, collate_fn=collate_content_batch)

    checkpoint = load_checkpoint(PROJECT_ROOT / args.checkpoint)
    id_to_char = checkpoint["id_to_char"]
    model = ContentVerificationModule(
        hidden_dim=model_cfg["hidden_dim"],
        num_phonemes=len(checkpoint["char_to_id"]) + 1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    greedy_rows = []
    cleaned_rows = []
    beam_rows = []

    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"])
            decoded = greedy_decode_from_log_probs(log_probs.cpu(), batch["input_lengths"])

            for gold_text, pred_ids, lp, length in zip(batch["texts"], decoded, log_probs.cpu(), batch["input_lengths"]):
                gold = normalize_text_target(gold_text)
                greedy_pred = decode_ids(pred_ids, id_to_char)
                greedy_rows.append(
                    EvalRow(
                        gold=gold,
                        pred=greedy_pred,
                        exact=(greedy_pred == gold),
                        char_acc=char_accuracy(gold, greedy_pred),
                        edit_distance=levenshtein(gold, greedy_pred),
                        repeated_errors=repeated_char_error_count(gold, greedy_pred),
                    )
                )

                cleaned_pred = content_postprocess(greedy_pred)
                cleaned_rows.append(
                    EvalRow(
                        gold=gold,
                        pred=cleaned_pred,
                        exact=(cleaned_pred == gold),
                        char_acc=char_accuracy(gold, cleaned_pred),
                        edit_distance=levenshtein(gold, cleaned_pred),
                        repeated_errors=repeated_char_error_count(gold, cleaned_pred),
                    )
                )

                beam_ids = ctc_prefix_beam_search(lp, int(length), beam_width=args.beam_width)
                beam_pred = decode_ids(beam_ids, id_to_char)
                beam_rows.append(
                    EvalRow(
                        gold=gold,
                        pred=beam_pred,
                        exact=(beam_pred == gold),
                        char_acc=char_accuracy(gold, beam_pred),
                        edit_distance=levenshtein(gold, beam_pred),
                        repeated_errors=repeated_char_error_count(gold, beam_pred),
                    )
                )

    greedy_summary = summarize(greedy_rows)
    cleaned_summary = summarize(cleaned_rows)
    beam_summary = summarize(beam_rows)

    print("Greedy summary:")
    print(
        f"count={greedy_summary['count']} "
        f"exact_match={greedy_summary['exact_match']:.3f} "
        f"char_accuracy={greedy_summary['char_accuracy']:.3f} "
        f"edit_distance={greedy_summary['edit_distance']:.3f} "
        f"repeat_errors={greedy_summary['repeated_errors']:.3f}"
    )
    print("")
    print("Greedy + cleanup summary:")
    print(
        f"count={cleaned_summary['count']} "
        f"exact_match={cleaned_summary['exact_match']:.3f} "
        f"char_accuracy={cleaned_summary['char_accuracy']:.3f} "
        f"edit_distance={cleaned_summary['edit_distance']:.3f} "
        f"repeat_errors={cleaned_summary['repeated_errors']:.3f}"
    )
    print("")
    print("Beam summary:")
    print(
        f"count={beam_summary['count']} "
        f"exact_match={beam_summary['exact_match']:.3f} "
        f"char_accuracy={beam_summary['char_accuracy']:.3f} "
        f"edit_distance={beam_summary['edit_distance']:.3f} "
        f"repeat_errors={beam_summary['repeated_errors']:.3f}"
    )
    print("")

    nearest_misses = [row for row in cleaned_rows if not row.exact]
    nearest_misses.sort(key=lambda row: (row.edit_distance, -row.char_acc, len(row.gold)))
    print(f"Nearest misses after cleanup (top {min(args.show_misses, len(nearest_misses))}):")
    for row in nearest_misses[: args.show_misses]:
        print(f"EDIT={row.edit_distance} CHAR_ACC={row.char_acc:.3f} REPEAT_ERR={row.repeated_errors}")
        print(f"GOLD : {row.gold}")
        print(f"PRED : {row.pred}")
        print("")


if __name__ == "__main__":
    main()

