from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter, defaultdict

import torch

from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json


DEFAULT_TRANSITION_MANIFEST = "data/manifests/retasy_transition_subset.jsonl"
BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_transition_module(checkpoint_name: str = "transition_module.pt") -> TransitionRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / checkpoint_name)
    model = TransitionRuleModule(
        mfcc_dim=model_cfg["mfcc_dim"],
        ssl_dim=model_cfg["ssl_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_transition_thresholds() -> dict[str, float] | None:
    path = PROJECT_ROOT / "checkpoints" / "transition_thresholds.json"
    if not path.exists():
        return None
    data = load_json(path)
    raw = data.get("thresholds", data) if isinstance(data, dict) else {}
    return {str(k): float(v) for k, v in raw.items()}


def gold_transition_label(row: dict) -> str:
    label = normalize_rule_name((row.get("canonical_rules") or ["none"])[0])
    return label if label in TRANSITION_RULES else "none"


def safe_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def confidence_bucket(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    for start, end in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        if start <= confidence < end:
            upper = min(end, 1.0)
            return f"{start:.1f}-{upper:.1f}"
    return "1.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_TRANSITION_MANIFEST)
    parser.add_argument("--checkpoint-name", default=preferred_transition_checkpoint())
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to analyze. 0 means all.")
    parser.add_argument("--show-examples", type=int, default=15)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    model = load_transition_module(args.checkpoint_name).to("cpu").eval()
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)
    thresholds = load_transition_thresholds()

    confusion = defaultdict(Counter)
    confidence_buckets = defaultdict(Counter)
    examples: list[dict] = []
    total = 0
    correct = 0

    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
        ssl = ssl_extractor.from_mfcc(mfcc)
        lengths = torch.tensor([mfcc.size(0)], dtype=torch.long)
        with torch.no_grad():
            logits = model(mfcc.unsqueeze(0), ssl.unsqueeze(0), lengths)
            probs = logits.softmax(dim=-1)[0]
        if thresholds:
            pred_idx = int(probs[1:].argmax().item()) + 1
            pred_name = TRANSITION_RULES[pred_idx]
            pred_confidence = float(probs[pred_idx].item())
            if pred_confidence < float(thresholds.get(pred_name, 0.5)):
                pred_idx = 0
                pred_confidence = float(probs[0].item())
        else:
            pred_idx = int(logits.argmax(dim=-1)[0].item())
            pred_confidence = float(probs[pred_idx].item())
        pred_label = TRANSITION_RULES[pred_idx]
        gold_label = gold_transition_label(row)

        total += 1
        if pred_label == gold_label:
            correct += 1
        confusion[gold_label][pred_label] += 1
        confidence_buckets[f"{gold_label}->{pred_label}"][confidence_bucket(pred_confidence)] += 1

        if pred_label != gold_label:
            examples.append(
                {
                    "sample_id": row.get("sample_id") or row.get("id"),
                    "surah_name": row.get("surah_name"),
                    "verse_key": row.get("quranjson_verse_key"),
                    "text": row.get("text") or row.get("normalized_text"),
                    "expected_rule": gold_label,
                    "predicted_rule": pred_label,
                    "confidence": pred_confidence,
                }
            )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    examples.sort(key=lambda item: item["confidence"], reverse=True)

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "checkpoint": str(PROJECT_ROOT / "checkpoints" / args.checkpoint_name),
        "samples_analyzed": len(rows),
        "accuracy": (correct / total) if total else None,
        "confusion_matrix": {
            expected: {predicted: confusion[expected][predicted] for predicted in TRANSITION_RULES}
            for expected in TRANSITION_RULES
        },
        "confidence_buckets": {pair: dict(buckets) for pair, buckets in confidence_buckets.items()},
        "top_mismatches": examples[: args.show_examples],
    }

    print(f"Samples analyzed : {len(rows)}")
    print(f"Accuracy         : {summary['accuracy']:.3f}" if summary["accuracy"] is not None else "Accuracy         : n/a")
    print("")
    print("Confusion matrix (expected -> predicted):")
    print(f"{'expected\\\\pred':14}{'none':>10}{'ikhfa':>10}{'idgham':>10}")
    for expected in TRANSITION_RULES:
        row_counts = [confusion[expected][pred] for pred in TRANSITION_RULES]
        print(f"{expected:14}{row_counts[0]:>10}{row_counts[1]:>10}{row_counts[2]:>10}")

    print("")
    print("Confidence buckets by prediction pair:")
    for pair in sorted(confidence_buckets):
        bucket_items = confidence_buckets[pair]
        bucket_text = ", ".join(f"{bucket}={count}" for bucket, count in sorted(bucket_items.items()))
        print(f"- {pair}: {bucket_text}")

    print("")
    print(f"Top mismatches (top {min(args.show_examples, len(examples))}):")
    if not examples:
        print("- None")
    else:
        for example in examples[: args.show_examples]:
            print(
                f"- {example['sample_id']}: expected {example['expected_rule']}, "
                f"predicted {example['predicted_rule']}. confidence={example['confidence']:.2f}"
            )
            if example["text"]:
                print(f"  text: {safe_text(example['text'])}")

    if args.output_json:
        save_json(summary, PROJECT_ROOT / args.output_json)
        print("")
        print(f"Saved analysis JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

