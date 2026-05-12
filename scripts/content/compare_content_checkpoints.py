from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    normalize_text_target,
    split_content_indices,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


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
            curr.append(
                min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + int(ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(gold, pred) / len(gold))


def build_speed_config(data_cfg: dict[str, Any]) -> SpeedNormalizationConfig:
    return SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )


def infer_hidden_dim_from_state(state: dict[str, torch.Tensor], fallback: int) -> int:
    weight = state.get("encoder.lstm.weight_ih_l0")
    if weight is None:
        return fallback
    return int(weight.shape[0] // 4)


def load_content_checkpoint(checkpoint_path: Path) -> tuple[ContentVerificationModule, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model_state_dict"]

    char_to_id = ckpt.get("char_to_id")
    if not isinstance(char_to_id, dict):
        raise RuntimeError(f"Checkpoint has no char_to_id: {checkpoint_path}")

    model_cfg = ckpt.get("config", {}).get("model", {})
    config_hidden = int(model_cfg.get("hidden_dim", 64))
    hidden_dim = infer_hidden_dim_from_state(state, config_hidden)

    if hidden_dim != config_hidden:
        print(
            f"[warning] {checkpoint_path.name}: config hidden_dim={config_hidden}, "
            f"inferred hidden_dim={hidden_dim}. Using inferred."
        )

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    )
    model.load_state_dict(state)
    model.eval()

    return model, ckpt


def decode_ids(ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(int(i), "") for i in ids)


def evaluate_predictions(
    *,
    model: ContentVerificationModule,
    checkpoint: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    char_to_id = checkpoint["char_to_id"]
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}

    model.to(device)
    model.eval()

    predictions: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            log_probs = model(x, input_lengths)
            decoded = greedy_decode_from_log_probs(log_probs.cpu(), batch["input_lengths"])

            ids = batch.get("ids")
            if ids is None:
                ids = [str(len(predictions) + i) for i in range(len(batch["texts"]))]

            for sample_id, gold_text, pred_ids in zip(ids, batch["texts"], decoded):
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                edit = levenshtein(gold, pred)

                predictions[str(sample_id)] = {
                    "gold": gold,
                    "pred": pred,
                    "exact": pred == gold,
                    "edit_distance": edit,
                    "char_accuracy": char_accuracy(gold, pred),
                }

    return predictions


def classify(base: dict[str, Any], cand: dict[str, Any]) -> str:
    if not base["exact"] and cand["exact"]:
        return "fixed"
    if base["exact"] and not cand["exact"]:
        return "broken"
    if base["exact"] and cand["exact"]:
        return "both_correct"

    if cand["edit_distance"] < base["edit_distance"]:
        return "both_wrong_improved"
    if cand["edit_distance"] > base["edit_distance"]:
        return "both_wrong_worsened"
    return "both_wrong_same"


def summarize_by_key(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        value = item.get(key)
        if value is not None:
            grouped[str(value)].append(item)

    out = {}
    for value, rows in grouped.items():
        counts = Counter(row["category"] for row in rows)
        out[value] = {
            "samples": len(rows),
            "counts": dict(counts),
            "broken": counts.get("broken", 0),
            "fixed": counts.get("fixed", 0),
            "net_fixed_minus_broken": counts.get("fixed", 0) - counts.get("broken", 0),
            "mean_base_edit": sum(float(row["base_edit_distance"]) for row in rows) / len(rows),
            "mean_candidate_edit": sum(float(row["candidate_edit_distance"]) for row in rows) / len(rows),
        }

    return dict(sorted(out.items(), key=lambda pair: pair[1]["samples"], reverse=True))


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]

    lines = []
    lines.append("# Content Checkpoint Regression Analysis")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    lines.append(f"- Baseline: `{result['baseline_checkpoint']}`")
    lines.append(f"- Candidate: `{result['candidate_checkpoint']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Samples: {summary['samples']}")
    lines.append(f"- Baseline exact: {summary['baseline_exact_match']:.3f}")
    lines.append(f"- Candidate exact: {summary['candidate_exact_match']:.3f}")
    lines.append(f"- Baseline char accuracy: {summary['baseline_char_accuracy']:.3f}")
    lines.append(f"- Candidate char accuracy: {summary['candidate_char_accuracy']:.3f}")
    lines.append(f"- Baseline mean edit distance: {summary['baseline_mean_edit_distance']:.3f}")
    lines.append(f"- Candidate mean edit distance: {summary['candidate_mean_edit_distance']:.3f}")
    lines.append("")
    lines.append("## Regression Categories")
    lines.append("")
    for key, value in summary["category_counts"].items():
        lines.append(f"- {key}: {value}")

    def add_examples(title: str, key: str, limit: int = 10) -> None:
        rows = [item for item in result["items"] if item["category"] == key]
        rows = rows[:limit]
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("- None")
            return

        for item in rows:
            lines.append(f"### {item['sample_id']}")
            lines.append("")
            lines.append(f"- Gold: `{item['gold']}`")
            lines.append(f"- Baseline: `{item['baseline_pred']}`")
            lines.append(f"- Candidate: `{item['candidate_pred']}`")
            lines.append(f"- Base edit: {item['base_edit_distance']}")
            lines.append(f"- Candidate edit: {item['candidate_edit_distance']}")
            lines.append(f"- Surah: {item.get('surah_name')}")
            lines.append(f"- Verse: {item.get('quranjson_verse_key')}")
            lines.append("")

    add_examples("Fixed examples", "fixed")
    add_examples("Broken examples", "broken")
    add_examples("Improved but still wrong", "both_wrong_improved")
    add_examples("Worsened and still wrong", "both_wrong_worsened")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--baseline-checkpoint", default="checkpoints/content_chunked_module_hd96_reciter.pt")
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_content.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_regression_ssl_cache")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    data_cfg = load_yaml_file(resolve_path(args.data_config))
    model_cfg = load_yaml_file(resolve_path(args.model_config))
    train_cfg = load_yaml_file(resolve_path(args.train_config))

    baseline_model, baseline_ckpt = load_content_checkpoint(resolve_path(args.baseline_checkpoint))
    candidate_model, candidate_ckpt = load_content_checkpoint(resolve_path(args.candidate_checkpoint))

    base_vocab = baseline_ckpt["char_to_id"]
    cand_vocab = candidate_ckpt["char_to_id"]
    if base_vocab != cand_vocab:
        raise RuntimeError("Baseline and candidate char_to_id differ. Comparison would be invalid.")

    dataset = ChunkedContentDataset(
        resolve_path(args.manifest),
        sample_rate=int(data_cfg["sample_rate"]),
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=build_speed_config(data_cfg),
        feature_cache_dir=resolve_path(args.feature_cache_dir),
        char_to_id=base_vocab,
    )

    train_idx, val_idx = split_content_indices(
        dataset.rows,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        seed=int(train_cfg.get("seed", 7)),
    )

    selected_idx = val_idx if args.split == "val" else train_idx
    if args.limit > 0:
        selected_idx = selected_idx[: args.limit]

    subset = Subset(dataset, selected_idx)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    base_preds = evaluate_predictions(
        model=baseline_model,
        checkpoint=baseline_ckpt,
        loader=loader,
        device=device,
    )

    cand_preds = evaluate_predictions(
        model=candidate_model,
        checkpoint=candidate_ckpt,
        loader=loader,
        device=device,
    )

    items: list[dict[str, Any]] = []

    for local_i, dataset_idx in enumerate(selected_idx):
        row = dataset.rows[dataset_idx]
        sample_id = str(row.get("id") or row.get("sample_id") or local_i)

        base = base_preds[sample_id]
        cand = cand_preds[sample_id]
        category = classify(base, cand)

        items.append(
            {
                "sample_id": sample_id,
                "dataset_index": int(dataset_idx),
                "category": category,
                "gold": base["gold"],
                "baseline_pred": base["pred"],
                "candidate_pred": cand["pred"],
                "baseline_exact": base["exact"],
                "candidate_exact": cand["exact"],
                "base_edit_distance": base["edit_distance"],
                "candidate_edit_distance": cand["edit_distance"],
                "edit_delta_candidate_minus_base": cand["edit_distance"] - base["edit_distance"],
                "base_char_accuracy": base["char_accuracy"],
                "candidate_char_accuracy": cand["char_accuracy"],
                "char_accuracy_delta_candidate_minus_base": cand["char_accuracy"] - base["char_accuracy"],
                "surah_name": row.get("surah_name"),
                "quranjson_verse_key": row.get("quranjson_verse_key"),
                "reciter_id": row.get("reciter_id"),
                "word_count": row.get("word_count"),
                "char_count": row.get("char_count"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
            }
        )

    # Sort high-impact regressions first, then fixes.
    items = sorted(
        items,
        key=lambda item: (
            {
                "broken": 0,
                "fixed": 1,
                "both_wrong_worsened": 2,
                "both_wrong_improved": 3,
                "both_wrong_same": 4,
                "both_correct": 5,
            }.get(item["category"], 99),
            -abs(float(item["edit_delta_candidate_minus_base"])),
        ),
    )

    counts = Counter(item["category"] for item in items)
    samples = len(items)

    summary = {
        "samples": samples,
        "category_counts": dict(counts),
        "baseline_exact_match": sum(1 for item in items if item["baseline_exact"]) / samples if samples else 0.0,
        "candidate_exact_match": sum(1 for item in items if item["candidate_exact"]) / samples if samples else 0.0,
        "baseline_char_accuracy": sum(float(item["base_char_accuracy"]) for item in items) / samples if samples else 0.0,
        "candidate_char_accuracy": sum(float(item["candidate_char_accuracy"]) for item in items) / samples if samples else 0.0,
        "baseline_mean_edit_distance": sum(float(item["base_edit_distance"]) for item in items) / samples if samples else 0.0,
        "candidate_mean_edit_distance": sum(float(item["candidate_edit_distance"]) for item in items) / samples if samples else 0.0,
    }

    result = {
        "baseline_checkpoint": str(resolve_path(args.baseline_checkpoint)),
        "candidate_checkpoint": str(resolve_path(args.candidate_checkpoint)),
        "manifest": str(resolve_path(args.manifest)),
        "split": args.split,
        "limit": args.limit,
        "summary": summary,
        "by_surah": summarize_by_key(items, "surah_name"),
        "by_reciter": summarize_by_key(items, "reciter_id"),
        "items": items,
    }

    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output_md, result)

    print("Content regression analysis")
    print("---------------------------")
    print(f"samples: {summary['samples']}")
    print(f"baseline_exact: {summary['baseline_exact_match']:.3f}")
    print(f"candidate_exact: {summary['candidate_exact_match']:.3f}")
    print(f"baseline_char: {summary['baseline_char_accuracy']:.3f}")
    print(f"candidate_char: {summary['candidate_char_accuracy']:.3f}")
    print(f"baseline_edit: {summary['baseline_mean_edit_distance']:.3f}")
    print(f"candidate_edit: {summary['candidate_mean_edit_distance']:.3f}")
    print(f"categories: {summary['category_counts']}")
    print(f"saved_json: {output_json}")
    print(f"saved_md: {output_md}")


if __name__ == "__main__":
    main()
