from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.inference.learned_routing import load_learned_routing_predictor


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bool_targets(row: dict[str, Any]) -> dict[str, bool]:
    return {
        name: bool(value)
        for name, value in zip(row["target_names"], row["targets"])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/manifests/learned_routing_dataset_v2.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/learned_router_v2.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="data/analysis/learned_router_v2_eval.json")
    parser.add_argument("--max-examples", type=int, default=30)
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.dataset))
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]

    predictor = load_learned_routing_predictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    target_names = ["use_duration", "use_transition", "use_burst"]
    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for name in target_names
    }

    exact = 0
    mistakes = []

    for row in rows:
        text = row.get("text") or row.get("normalized_text") or row.get("source_text") or ""
        pred = predictor.predict(audio_path=row["audio_path"], text=text)
        pred_plan = pred.routing_plan
        gold_plan = bool_targets(row)

        if pred_plan == gold_plan:
            exact += 1
        elif len(mistakes) < args.max_examples:
            mistakes.append(
                {
                    "id": row.get("id"),
                    "text": text,
                    "sources": row.get("sources", []),
                    "gold": gold_plan,
                    "predicted": pred_plan,
                    "probabilities": pred.probabilities,
                    "thresholds": pred.thresholds,
                }
            )

        for name in target_names:
            g = gold_plan[name]
            p = pred_plan[name]

            if g and p:
                counts[name]["tp"] += 1
            elif (not g) and p:
                counts[name]["fp"] += 1
            elif g and (not p):
                counts[name]["fn"] += 1
            else:
                counts[name]["tn"] += 1

    per_label = {}
    f1s = []

    for name, c in counts.items():
        tp = c["tp"]
        fp = c["fp"]
        fn = c["fn"]
        tn = c["tn"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

        f1s.append(f1)
        per_label[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    result = {
        "dataset": str(resolve_path(args.dataset)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": args.split,
        "samples": len(rows),
        "exact_match": exact / max(1, len(rows)),
        "macro_f1": sum(f1s) / len(f1s),
        "per_label": per_label,
        "mistakes": mistakes,
    }

    out = resolve_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Learned router evaluation")
    print("-------------------------")
    print(f"samples={result['samples']}")
    print(f"exact={result['exact_match']:.3f}")
    print(f"macro_f1={result['macro_f1']:.3f}")
    for name, metrics in per_label.items():
        print(
            f"- {name}: "
            f"f1={metrics['f1']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"fp={metrics['fp']} fn={metrics['fn']}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
