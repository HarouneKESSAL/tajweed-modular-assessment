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


def metrics_for_thresholds(rows, raw_predictions, thresholds):
    target_names = ["use_duration", "use_transition", "use_burst"]

    counts = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for name in target_names
    }

    exact = 0

    for row, probs in zip(rows, raw_predictions):
        gold = bool_targets(row)
        pred = {
            name: float(probs[name]) >= float(thresholds[name])
            for name in target_names
        }

        if pred == gold:
            exact += 1

        for name in target_names:
            g = gold[name]
            p = pred[name]

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
            "threshold": thresholds[name],
        }

    return {
        "exact_match": exact / max(1, len(rows)),
        "macro_f1": sum(f1s) / len(f1s),
        "per_label": per_label,
        "thresholds": thresholds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/manifests/learned_routing_dataset_v2.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/learned_router_v2.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="data/analysis/learned_router_v2_thresholds.json")
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.dataset))
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]

    predictor = load_learned_routing_predictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    raw_predictions = []

    for row in rows:
        text = row.get("text") or row.get("normalized_text") or row.get("source_text") or ""
        pred = predictor.predict(audio_path=row["audio_path"], text=text)
        raw_predictions.append(pred.probabilities)

    grid = [x / 100.0 for x in range(30, 96, 5)]
    all_results = []
    best = None

    for duration_t in grid:
        for transition_t in grid:
            for burst_t in grid:
                thresholds = {
                    "use_duration": duration_t,
                    "use_transition": transition_t,
                    "use_burst": burst_t,
                }

                metrics = metrics_for_thresholds(rows, raw_predictions, thresholds)
                all_results.append(metrics)

                # Prioritize macro_f1, then exact, then fewer false negatives.
                total_fn = sum(m["fn"] for m in metrics["per_label"].values())
                total_fp = sum(m["fp"] for m in metrics["per_label"].values())

                key = (
                    metrics["macro_f1"],
                    metrics["exact_match"],
                    -total_fn,
                    -total_fp,
                )

                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "metrics": metrics,
                    }

    top_10 = sorted(
        all_results,
        key=lambda m: (
            m["macro_f1"],
            m["exact_match"],
            -sum(x["fn"] for x in m["per_label"].values()),
            -sum(x["fp"] for x in m["per_label"].values()),
        ),
        reverse=True,
    )[:10]

    result = {
        "dataset": str(resolve_path(args.dataset)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": args.split,
        "samples": len(rows),
        "best": best["metrics"],
        "top_10": top_10,
    }

    out = resolve_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Learned router threshold tuning")
    print("-------------------------------")
    print(f"samples={len(rows)}")
    print(f"best thresholds={best['metrics']['thresholds']}")
    print(f"best exact={best['metrics']['exact_match']:.3f}")
    print(f"best macro_f1={best['metrics']['macro_f1']:.3f}")
    for name, metrics in best["metrics"]["per_label"].items():
        print(
            f"- {name}: "
            f"f1={metrics['f1']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"fp={metrics['fp']} fn={metrics['fn']} "
            f"threshold={metrics['threshold']}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
