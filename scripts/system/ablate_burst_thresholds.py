from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.system.evaluate_modular_suite import (
    DEFAULT_BURST_MANIFEST,
    evaluate_burst_manifest,
    load_burst_module,
    load_jsonl,
)


def summarize(summary: dict[str, Any]) -> dict[str, Any]:
    cm = summary.get("confusion_matrix") or [[0, 0], [0, 0]]

    tn = int(cm[0][0])
    fp = int(cm[0][1])
    fn = int(cm[1][0])
    tp = int(cm[1][1])

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "threshold": summary.get("burst_threshold"),
        "decision_rule": summary.get("decision_rule"),
        "samples": summary.get("samples"),
        "accuracy": summary.get("accuracy"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "qalqalah_precision": precision,
        "qalqalah_recall": recall,
        "qalqalah_f1": f1,
        "none_accuracy": summary.get("class_summary", {}).get("none", {}).get("accuracy"),
        "qalqalah_accuracy": summary.get("class_summary", {}).get("qalqalah", {}).get("accuracy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burst-manifest", default=DEFAULT_BURST_MANIFEST)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--include-argmax", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.burst_manifest)
    model = load_burst_module()

    results = []

    print("Burst threshold ablation")
    print("------------------------")
    print("manifest:", args.burst_manifest)
    print("samples:", len(rows) if args.limit <= 0 else min(args.limit, len(rows)))

    if args.include_argmax:
        argmax_summary = evaluate_burst_manifest(model, rows, args.limit, burst_threshold=None)
        results.append(summarize(argmax_summary))

    for threshold in args.thresholds:
        summary = evaluate_burst_manifest(model, rows, args.limit, burst_threshold=threshold)
        results.append(summarize(summary))

    out_json = PROJECT_ROOT / args.output_json
    out_md = PROJECT_ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Burst threshold ablation")
    lines.append("")
    lines.append(f"- manifest: `{args.burst_manifest}`")
    lines.append(f"- limit: {args.limit}")
    lines.append("")
    lines.append("| threshold | rule | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in results:
        th = "argmax" if row["threshold"] is None else f"{row['threshold']:.2f}"
        lines.append(
            f"| {th} | {row['decision_rule']} | {row['accuracy']:.3f} | "
            f"{row['qalqalah_precision']:.3f} | {row['qalqalah_recall']:.3f} | {row['qalqalah_f1']:.3f} | "
            f"{row['fp']} | {row['fn']} | {row['none_accuracy']:.3f} | {row['qalqalah_accuracy']:.3f} |"
        )

    best_acc = max(results, key=lambda x: x["accuracy"])
    best_f1 = max(results, key=lambda x: x["qalqalah_f1"])

    lines.append("")
    lines.append("## Best")
    lines.append("")
    lines.append(f"- best accuracy: `{best_acc}`")
    lines.append(f"- best qalqalah F1: `{best_f1}`")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
