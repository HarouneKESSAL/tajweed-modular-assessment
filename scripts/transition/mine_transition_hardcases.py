from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

import torch

from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json


DEFAULT_TRANSITION_MANIFEST = "data/manifests/retasy_transition_subset.jsonl"
DEFAULT_OUTPUT_JSON = "data/analysis/transition_hardcases.json"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_transition_module() -> TransitionRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / "transition_module.pt")
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


def classify_hardcase(expected: str, predicted: str) -> str:
    if expected == "ikhfa" and predicted == "none":
        return "missed_ikhfa"
    if expected == "idgham" and predicted == "none":
        return "missed_idgham"
    if expected == "none" and predicted == "ikhfa":
        return "false_positive_ikhfa"
    if expected == "none" and predicted == "idgham":
        return "false_positive_idgham"
    if expected == "ikhfa" and predicted == "idgham":
        return "ikhfa_as_idgham"
    if expected == "idgham" and predicted == "ikhfa":
        return "idgham_as_ikhfa"
    if expected == predicted and expected in {"ikhfa", "idgham"}:
        return f"correct_positive_{expected}"
    if expected == predicted:
        return "correct_none"
    return "other_mismatch"


def recommend_weight(category: str, confidence: float) -> float:
    base = 1.0
    if category in {"missed_ikhfa", "missed_idgham", "false_positive_ikhfa"}:
        base = 4.0
    elif category in {"false_positive_idgham", "ikhfa_as_idgham", "idgham_as_ikhfa"}:
        base = 3.0
    elif category.startswith("correct_positive_"):
        base = 1.5
    elif category == "other_mismatch":
        base = 2.5
    if "missed" in category or "false_positive" in category or category.endswith("_as_ikhfa") or category.endswith("_as_idgham"):
        if confidence >= 0.9:
            base += 0.5
        elif confidence >= 0.75:
            base += 0.25
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_TRANSITION_MANIFEST)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--show-top", type=int, default=15)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    model = load_transition_module().to("cpu").eval()
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)
    thresholds = load_transition_thresholds()

    hardcases: list[dict] = []
    recommended_weights: dict[str, float] = {}
    category_counts = Counter()
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

        predicted = TRANSITION_RULES[pred_idx]
        expected = gold_transition_label(row)
        category = classify_hardcase(expected, predicted)
        category_counts[category] += 1
        total += 1
        if expected == predicted:
            correct += 1

        if category == "correct_none":
            if idx % 25 == 0:
                print(f"Processed {idx}/{len(rows)} samples...")
            continue

        sample_id = str(row.get("sample_id") or row.get("id") or row.get("audio_path") or f"sample_{idx}")
        weight = recommend_weight(category, pred_confidence)
        recommended_weights[sample_id] = max(recommended_weights.get(sample_id, 1.0), weight)
        hardcases.append(
            {
                "sample_id": sample_id,
                "audio_path": row.get("audio_path"),
                "text": row.get("text") or row.get("normalized_text"),
                "expected_rule": expected,
                "predicted_rule": predicted,
                "confidence": pred_confidence,
                "category": category,
                "weight": weight,
            }
        )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    hardcases.sort(key=lambda item: (item["weight"], item["confidence"]), reverse=True)
    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "samples_analyzed": len(rows),
        "accuracy": (correct / total) if total else None,
        "thresholds": thresholds or {},
        "category_counts": dict(category_counts),
        "weights": recommended_weights,
        "hardcases": hardcases,
    }

    print(f"Samples analyzed : {len(rows)}")
    print(f"Accuracy         : {summary['accuracy']:.3f}" if summary["accuracy"] is not None else "Accuracy         : n/a")
    print("Hard-case categories:")
    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")
    print("")
    print(f"Top weighted rows (top {min(args.show_top, len(hardcases))}):")
    if not hardcases:
        print("- None")
    else:
        for item in hardcases[: args.show_top]:
            print(
                f"- {item['sample_id']}: {item['category']} expected={item['expected_rule']} "
                f"predicted={item['predicted_rule']} confidence={item['confidence']:.2f} weight={item['weight']:.2f}"
            )

    save_json(summary, PROJECT_ROOT / args.output_json)
    print("")
    print(f"Saved hardcase JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

