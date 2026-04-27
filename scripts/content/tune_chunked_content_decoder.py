from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json

import torch
from torch.utils.data import DataLoader

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    normalize_text_target,
    split_content_indices,
)
from tajweed_assessment.data.labels import BLANK_ID
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.common.decoding import ctc_lexicon_decode, ctc_prefix_beam_search
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs
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
    return collapse_excess_repetitions(text, max_run=2)


def apply_blank_penalty(log_probs: torch.Tensor, blank_penalty: float) -> torch.Tensor:
    if blank_penalty == 0.0:
        return log_probs
    adjusted = log_probs.clone()
    adjusted[..., BLANK_ID] -= blank_penalty
    return adjusted


def decode_sequences(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    *,
    decoder: str,
    beam_width: int,
    lexicon_targets: list[list[int]] | None = None,
) -> list[list[int]]:
    if decoder == "beam":
        return [
            ctc_prefix_beam_search(seq_log_probs[: int(length)], int(length), beam_width=beam_width, blank_id=BLANK_ID)
            for seq_log_probs, length in zip(log_probs, lengths)
        ]
    if decoder == "lexicon":
        return [
            ctc_lexicon_decode(
                seq_log_probs,
                int(length),
                lexicon_targets or [],
                blank_id=BLANK_ID,
            )
            for seq_log_probs, length in zip(log_probs, lengths)
        ]
    return greedy_decode_from_log_probs(log_probs, lengths)


def evaluate_setting(
    model,
    loader,
    id_to_char,
    blank_penalty: float,
    use_cleanup: bool,
    decoder: str,
    beam_width: int,
    lexicon_targets: list[list[int]] | None,
) -> dict:
    total = exact = 0
    char_acc_sum = 0.0
    edit_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"]).cpu()
            decoded = decode_sequences(
                apply_blank_penalty(log_probs, blank_penalty),
                batch["input_lengths"],
                decoder=decoder,
                beam_width=beam_width,
                lexicon_targets=lexicon_targets,
            )
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                if use_cleanup:
                    pred = content_postprocess(pred)
                total += 1
                exact += int(pred == gold)
                char_acc_sum += char_accuracy(gold, pred)
                edit_sum += levenshtein(gold, pred)
    return {
        "samples": total,
        "exact_match": exact / max(total, 1),
        "char_accuracy": char_acc_sum / max(total, 1),
        "edit_distance": edit_sum / max(total, 1),
        "blank_penalty": blank_penalty,
        "use_cleanup": use_cleanup,
        "decoder": decoder,
        "beam_width": beam_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module.pt")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--split-mode", choices=["reciter", "text"], default="reciter")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--blank-penalties", default="0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--decoders", default="greedy,beam,lexicon")
    parser.add_argument("--beam-widths", default="3,5,8")
    parser.add_argument("--output-json", default="")
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

    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    full_rows = list(dataset.rows)
    full_char_to_id = dict(dataset.char_to_id)
    train_idx, val_idx = split_content_indices(
        dataset.rows,
        val_fraction=0.2,
        seed=train_cfg["seed"],
        split_mode=args.split_mode,
    )
    indices = train_idx if args.split == "train" else val_idx if args.split == "val" else list(range(len(dataset)))
    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=indices,
        char_to_id=full_char_to_id,
    )
    lexicon_texts = sorted({normalize_text_target(row.get("normalized_text", "")) for row in full_rows})
    lexicon_targets = [
        [full_char_to_id[ch] for ch in text]
        for text in lexicon_texts
        if text and all(ch in full_char_to_id for ch in text)
    ]
    loader = DataLoader(
        dataset,
        batch_size=min(train_cfg["batch_size"], max(1, len(dataset))),
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    checkpoint = load_checkpoint(PROJECT_ROOT / args.checkpoint)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    ckpt_model_cfg = checkpoint.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    model = ContentVerificationModule(hidden_dim=hidden_dim, num_phonemes=len(checkpoint["char_to_id"]) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    penalties = [float(item.strip()) for item in args.blank_penalties.split(",") if item.strip()]
    decoders = [item.strip() for item in args.decoders.split(",") if item.strip()]
    beam_widths = [int(item.strip()) for item in args.beam_widths.split(",") if item.strip()]
    results = []
    for decoder in decoders:
        widths = beam_widths if decoder == "beam" else [1]
        for beam_width in widths:
            for blank_penalty in penalties:
                for use_cleanup in (False, True):
                    results.append(
                        evaluate_setting(
                            model,
                            loader,
                            id_to_char,
                            blank_penalty,
                            use_cleanup,
                            decoder,
                            beam_width,
                            lexicon_targets,
                        )
                    )

    results.sort(key=lambda item: (-item["exact_match"], -item["char_accuracy"], item["edit_distance"]))
    best = results[0] if results else {}

    print("Chunked content decoder tuning:")
    for item in results:
        print(
            f"- decoder={item['decoder']} beam_width={item['beam_width']} "
            f"blank_penalty={item['blank_penalty']:.2f} cleanup={str(item['use_cleanup']).lower()} "
            f"exact_match={item['exact_match']:.3f} char_accuracy={item['char_accuracy']:.3f} "
            f"edit_distance={item['edit_distance']:.3f}"
        )
    if best:
        print("")
        print(
            f"Best setting: decoder={best['decoder']} beam_width={best['beam_width']} "
            f"blank_penalty={best['blank_penalty']:.2f} cleanup={str(best['use_cleanup']).lower()} "
            f"exact_match={best['exact_match']:.3f} char_accuracy={best['char_accuracy']:.3f} "
            f"edit_distance={best['edit_distance']:.3f}"
        )

    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "checkpoint": str(PROJECT_ROOT / args.checkpoint),
                    "split_mode": args.split_mode,
                    "results": results,
                    "best": best,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved tuning JSON to {output_path}")


if __name__ == "__main__":
    main()

