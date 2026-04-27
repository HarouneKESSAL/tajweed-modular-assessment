from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from torch.utils.data import DataLoader
import torch

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    split_content_indices,
    normalize_text_target,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.data.labels import BLANK_ID
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.models.common.decoding import (
    ctc_lexicon_decode,
    ctc_prefix_beam_search,
    ctc_target_log_probability,
)
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


def greedy_decode_single(log_probs: torch.Tensor, length: int) -> list[int]:
    path = log_probs[: int(length)].argmax(dim=-1).tolist()
    decoded: list[int] = []
    prev = None
    for token_id in path:
        if token_id != BLANK_ID and token_id != prev:
            decoded.append(int(token_id))
        prev = token_id
    return decoded


def ctc_lexicon_open_decode(
    log_probs: torch.Tensor,
    length: int,
    lexicon_targets: list[list[int]],
    *,
    fallback_margin: float,
) -> list[int]:
    trimmed = log_probs[: int(length)]
    greedy_ids = greedy_decode_single(trimmed, int(length))
    if not lexicon_targets:
        return greedy_ids
    lexicon_ids = ctc_lexicon_decode(trimmed, int(length), lexicon_targets, blank_id=BLANK_ID)
    greedy_score = ctc_target_log_probability(trimmed, greedy_ids, blank_id=BLANK_ID)
    lexicon_score = ctc_target_log_probability(trimmed, lexicon_ids, blank_id=BLANK_ID)
    if greedy_score >= lexicon_score + fallback_margin:
        return greedy_ids
    return lexicon_ids


def decode_sequences(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    *,
    decoder: str,
    beam_width: int,
    lexicon_targets: list[list[int]] | None = None,
    open_fallback_margin: float = 0.0,
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
    if decoder == "lexicon-open":
        return [
            ctc_lexicon_open_decode(
                seq_log_probs,
                int(length),
                lexicon_targets or [],
                fallback_margin=open_fallback_margin,
            )
            for seq_log_probs, length in zip(log_probs, lengths)
        ]
    return greedy_decode_from_log_probs(log_probs, lengths)


def build_lexicon_texts(rows: list[dict], char_to_id: dict[str, int]) -> tuple[list[str], list[list[int]]]:
    texts = sorted({normalize_text_target(row.get("normalized_text", "")) for row in rows})
    targets = [
        [char_to_id[ch] for ch in text]
        for text in texts
        if text and all(ch in char_to_id for ch in text)
    ]
    return texts, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module.pt")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--split-mode", choices=["reciter", "text"], default="reciter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--use-cleanup", action="store_true")
    parser.add_argument("--blank-penalty", type=float, default=0.0)
    parser.add_argument("--decoder", choices=["greedy", "beam", "lexicon", "lexicon-open"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument(
        "--open-fallback-margin",
        type=float,
        default=0.0,
        help="For lexicon-open, choose greedy only when its CTC score beats the lexicon by this margin.",
    )
    parser.add_argument(
        "--lexicon-source",
        choices=["full", "train", "eval"],
        default="full",
        help="Which chunk texts are allowed by the lexicon decoder. Default keeps the current closed-set baseline.",
    )
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
    if args.split == "train":
        indices = train_idx
    elif args.split == "val":
        indices = val_idx
    else:
        indices = list(range(len(dataset)))
    if args.limit > 0:
        indices = indices[: args.limit]
    eval_rows = [full_rows[i] for i in indices]
    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=indices,
        char_to_id=full_char_to_id,
    )
    if args.lexicon_source == "train":
        lexicon_rows = [full_rows[i] for i in train_idx]
    elif args.lexicon_source == "eval":
        lexicon_rows = eval_rows
    else:
        lexicon_rows = full_rows
    lexicon_texts, lexicon_targets = build_lexicon_texts(lexicon_rows, full_char_to_id)
    lexicon_text_set = set(lexicon_texts)
    eval_texts = [normalize_text_target(row.get("normalized_text", "")) for row in eval_rows]
    eval_texts_in_lexicon = sum(1 for text in eval_texts if text in lexicon_text_set)
    loader = DataLoader(dataset, batch_size=min(train_cfg["batch_size"], max(1, len(dataset))), shuffle=False, collate_fn=collate_content_batch)

    checkpoint = load_checkpoint(PROJECT_ROOT / args.checkpoint)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    ckpt_model_cfg = checkpoint.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    model = ContentVerificationModule(hidden_dim=hidden_dim, num_phonemes=len(checkpoint["char_to_id"]) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = exact = 0
    char_acc_sum = 0.0
    edit_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"])
            decode_log_probs = apply_blank_penalty(log_probs.cpu(), args.blank_penalty)
            decoded = decode_sequences(
                decode_log_probs,
                batch["input_lengths"],
                decoder=args.decoder,
                beam_width=args.beam_width,
                lexicon_targets=lexicon_targets,
                open_fallback_margin=args.open_fallback_margin,
            )
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                if args.use_cleanup:
                    pred = content_postprocess(pred)
                total += 1
                exact += int(pred == gold)
                char_acc_sum += char_accuracy(gold, pred)
                edit_sum += levenshtein(gold, pred)
    summary = {
        "samples": total,
        "split": args.split,
        "split_mode": args.split_mode,
        "use_cleanup": bool(args.use_cleanup),
        "blank_penalty": float(args.blank_penalty),
        "decoder": str(args.decoder),
        "beam_width": int(args.beam_width),
        "open_fallback_margin": float(args.open_fallback_margin),
        "lexicon_source": str(args.lexicon_source),
        "lexicon_size": len(lexicon_targets),
        "eval_unique_text_count": len(set(eval_texts)),
        "eval_texts_in_lexicon": int(eval_texts_in_lexicon),
        "eval_text_coverage": eval_texts_in_lexicon / max(total, 1),
        "exact_match": exact / max(total, 1),
        "char_accuracy": char_acc_sum / max(total, 1),
        "edit_distance": edit_sum / max(total, 1),
        "checkpoint": str(PROJECT_ROOT / args.checkpoint),
    }
    print(
        f"Chunked content summary: samples={summary['samples']} split={summary['split']} "
        f"exact_match={summary['exact_match']:.3f} char_accuracy={summary['char_accuracy']:.3f} "
        f"edit_distance={summary['edit_distance']:.3f} "
        f"lexicon_source={summary['lexicon_source']} coverage={summary['eval_text_coverage']:.3f}"
    )
    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary JSON to {output_path}")


if __name__ == "__main__":
    main()

