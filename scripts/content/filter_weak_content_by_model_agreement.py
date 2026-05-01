from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

from content.evaluate_chunked_content import (
    apply_blank_penalty,
    char_accuracy,
    decode_ids,
    decode_sequences,
    levenshtein,
)
from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    load_jsonl,
    normalize_text_target,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def known_vocab_rows(rows: list[dict], char_to_id: dict[str, int]) -> tuple[list[dict], Counter]:
    kept: list[dict] = []
    skipped = Counter()
    known_chars = set(char_to_id)
    for row in rows:
        text = normalize_text_target(row.get("normalized_text", ""))
        if not text:
            skipped["empty_text"] += 1
            continue
        unknown = sorted(set(text) - known_chars)
        if unknown:
            skipped["unknown_checkpoint_chars"] += 1
            continue
        kept.append(row)
    return kept, skipped


def load_model(checkpoint_path: Path) -> tuple[ContentVerificationModule, dict]:
    checkpoint = load_checkpoint(checkpoint_path)
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    ckpt_model_cfg = checkpoint.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    model = ContentVerificationModule(hidden_dim=hidden_dim, num_phonemes=len(checkpoint["char_to_id"]) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_content_weak_chunks_no_textsplit_val.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_weak_chunks_agree_hd96.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module_hd96_reciter.pt")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--blank-penalty", type=float, default=1.6)
    parser.add_argument("--min-char-accuracy", type=float, default=1.0)
    parser.add_argument("--max-edit-distance", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-analysis-json", default="data/analysis/weak_content_model_agreement_filter.json")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    model, checkpoint = load_model(PROJECT_ROOT / args.checkpoint)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    char_to_id = dict(checkpoint["char_to_id"])
    rows = load_jsonl(PROJECT_ROOT / args.input)
    candidate_rows, skipped = known_vocab_rows(rows, char_to_id)
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    temp_manifest = PROJECT_ROOT / "data" / "interim" / "_weak_content_agreement_candidates.jsonl"
    write_jsonl(candidate_rows, temp_manifest)
    dataset = ChunkedContentDataset(
        temp_manifest,
        sample_rate=int(data_cfg["sample_rate"]),
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        char_to_id=char_to_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    accepted: list[dict] = []
    scored_examples: list[dict] = []
    score_buckets = Counter()
    total = 0
    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"]).cpu()
            decoded = decode_sequences(
                apply_blank_penalty(log_probs, float(args.blank_penalty)),
                batch["input_lengths"],
                decoder=args.decoder,
                beam_width=int(args.beam_width),
            )
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                row = candidate_rows[row_offset]
                row_offset += 1
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                acc = char_accuracy(gold, pred)
                edit = levenshtein(gold, pred)
                total += 1
                bucket = "1.00" if acc >= 1.0 else "0.90-0.99" if acc >= 0.9 else "0.75-0.89" if acc >= 0.75 else "0.50-0.74" if acc >= 0.5 else "<0.50"
                score_buckets[bucket] += 1
                if acc >= float(args.min_char_accuracy) and edit <= int(args.max_edit_distance):
                    kept = dict(row)
                    kept["agreement_filter"] = {
                        "checkpoint": args.checkpoint,
                        "decoder": args.decoder,
                        "blank_penalty": float(args.blank_penalty),
                        "predicted_text": pred,
                        "char_accuracy": acc,
                        "edit_distance": edit,
                    }
                    accepted.append(kept)
                if len(scored_examples) < 20:
                    scored_examples.append(
                        {
                            "id": row.get("id"),
                            "target": gold,
                            "prediction": pred,
                            "char_accuracy": acc,
                            "edit_distance": edit,
                        }
                    )

    output_path = PROJECT_ROOT / args.output
    write_jsonl(accepted, output_path)
    summary = {
        "input": str(PROJECT_ROOT / args.input),
        "output": str(output_path),
        "checkpoint": str(PROJECT_ROOT / args.checkpoint),
        "decoder": args.decoder,
        "blank_penalty": float(args.blank_penalty),
        "min_char_accuracy": float(args.min_char_accuracy),
        "max_edit_distance": int(args.max_edit_distance),
        "input_rows": len(rows),
        "candidate_rows": len(candidate_rows),
        "accepted_rows": len(accepted),
        "unique_accepted_texts": len({normalize_text_target(row.get("normalized_text", "")) for row in accepted}),
        "unique_accepted_source_texts": len({normalize_text_target(row.get("source_normalized_text", "")) for row in accepted}),
        "skipped": dict(skipped),
        "score_buckets": dict(score_buckets),
        "examples": scored_examples,
    }
    print_json(summary)
    if args.output_analysis_json:
        analysis_path = PROJECT_ROOT / args.output_analysis_json
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved analysis JSON to {analysis_path}")


if __name__ == "__main__":
    main()
