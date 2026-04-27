from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter, defaultdict

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
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs
from tajweed_assessment.utils.io import load_checkpoint


def decode_ids(ids, id_to_char):
    return "".join(id_to_char.get(int(i), "") for i in ids)


def safe_text(value: str | None) -> str:
    if value is None:
        return ""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower() in {"cp1252", "charmap"}:
        return str(value).encode("ascii", "backslashreplace").decode("ascii")
    return str(value)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module.pt")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--blank-penalty", type=float, default=1.0)
    parser.add_argument("--use-cleanup", action="store_true")
    parser.add_argument("--output-json", default="data/analysis/chunked_content_hardcases.json")
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

    base_dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    train_idx, val_idx = split_content_indices(base_dataset.rows, val_fraction=0.2, seed=train_cfg["seed"])
    indices = train_idx if args.split == "train" else val_idx
    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(train_cfg["batch_size"], max(1, len(dataset))),
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    checkpoint = load_checkpoint(PROJECT_ROOT / args.checkpoint)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    model = ContentVerificationModule(hidden_dim=model_cfg["hidden_dim"], num_phonemes=len(checkpoint["char_to_id"]) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_phrase = defaultdict(lambda: {"count": 0, "errors": 0, "edit_sum": 0.0})
    hardcases = []
    weights = {}
    category_counts = Counter()

    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"]).cpu()
            decoded = greedy_decode_from_log_probs(apply_blank_penalty(log_probs, args.blank_penalty), batch["input_lengths"])
            batch_size = len(batch["texts"])
            for i in range(batch_size):
                row = dataset.rows[row_offset + i]
                sample_id = str(row.get("id") or row.get("audio_path") or f"row_{row_offset+i}")
                gold = normalize_text_target(batch["texts"][i])
                pred = decode_ids(decoded[i], id_to_char)
                if args.use_cleanup:
                    pred = content_postprocess(pred)
                dist = levenshtein(gold, pred)
                exact = pred == gold

                phrase_stats = per_phrase[gold]
                phrase_stats["count"] += 1
                phrase_stats["errors"] += int(not exact)
                phrase_stats["edit_sum"] += dist

                weight = 1.0
                categories = []
                if not exact:
                    weight += min(3.0, dist * 0.5)
                    categories.append("mismatch")
                    if len(gold) >= 9:
                        weight += 0.5
                        categories.append("long_chunk")
                    if gold and pred and gold[0] != pred[0]:
                        weight += 0.5
                        categories.append("wrong_prefix")
                    if len(pred) < len(gold):
                        weight += 0.5
                        categories.append("deletion_bias")
                if weight > 1.0:
                    weights[sample_id] = round(min(weight, 5.0), 3)
                    hardcases.append(
                        {
                            "sample_id": sample_id,
                            "weight": weights[sample_id],
                            "gold": gold,
                            "pred": pred,
                            "edit_distance": dist,
                            "categories": categories,
                        }
                    )
                    for category in categories:
                        category_counts[category] += 1
            row_offset += batch_size

    phrase_hardness = []
    for phrase, stats in per_phrase.items():
        error_rate = stats["errors"] / max(stats["count"], 1)
        mean_edit = stats["edit_sum"] / max(stats["count"], 1)
        phrase_hardness.append(
            {
                "text": phrase,
                "count": stats["count"],
                "error_rate": error_rate,
                "mean_edit_distance": mean_edit,
            }
        )
    phrase_hardness.sort(key=lambda item: (-item["error_rate"], -item["count"], -item["mean_edit_distance"]))

    summary = {
        "split": args.split,
        "checkpoint": str(PROJECT_ROOT / args.checkpoint),
        "blank_penalty": float(args.blank_penalty),
        "use_cleanup": bool(args.use_cleanup),
        "weights": weights,
        "hardcases": hardcases,
        "stats": {
            "rows": len(dataset),
            "boosted_rows": len(weights),
            "mean_weight": (sum(weights.values()) / max(len(weights), 1)) if weights else 1.0,
            "max_weight": max(weights.values()) if weights else 1.0,
            "categories": dict(category_counts),
        },
        "hardest_phrases": phrase_hardness[:20],
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Chunked hardcases mined: split={args.split} rows={len(dataset)} boosted={len(weights)}")
    print(f"Mean weight={summary['stats']['mean_weight']:.2f} max_weight={summary['stats']['max_weight']:.2f}")
    print(f"Categories={summary['stats']['categories']}")
    print("Top hardest phrases:")
    for item in summary["hardest_phrases"][:10]:
        print(
            f"- {safe_text(item['text'])}: count={item['count']} error_rate={item['error_rate']:.3f} "
            f"mean_edit_distance={item['mean_edit_distance']:.3f}"
        )
    print(f"Saved hardcase JSON to {output_path}")


if __name__ == "__main__":
    main()

