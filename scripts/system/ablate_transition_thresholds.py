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
    build_localized_transition_index,
    evaluate_transition_manifest,
    load_jsonl,
    load_localized_transition_module,
    load_transition_module,
    load_transition_thresholds,
    preferred_transition_checkpoint,
)


def summarize_delta(with_thresholds: dict[str, Any], without_thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "with_thresholds_accuracy": with_thresholds.get("accuracy"),
        "without_thresholds_accuracy": without_thresholds.get("accuracy"),
        "delta_accuracy_without_minus_with": (
            without_thresholds.get("accuracy", 0.0) - with_thresholds.get("accuracy", 0.0)
        ),
        "with_thresholds_errors": int(with_thresholds.get("samples", 0) * (1.0 - with_thresholds.get("accuracy", 0.0))),
        "without_thresholds_errors": int(without_thresholds.get("samples", 0) * (1.0 - without_thresholds.get("accuracy", 0.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-manifest", default="data/manifests/retasy_transition_subset.jsonl")
    parser.add_argument("--localized-transition-manifest", default="data/alignment/transition_time_projection_strict.jsonl")
    parser.add_argument("--transition-checkpoint", default=preferred_transition_checkpoint())
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    transition_rows = load_jsonl(PROJECT_ROOT / args.transition_manifest)
    localized_rows = (
        load_jsonl(PROJECT_ROOT / args.localized_transition_manifest)
        if (PROJECT_ROOT / args.localized_transition_manifest).exists()
        else []
    )

    model = load_transition_module(args.transition_checkpoint)
    localized_model, localized_labels, localized_thresholds = load_localized_transition_module()
    localized_index = build_localized_transition_index(localized_rows)

    tuned_thresholds = load_transition_thresholds()

    print("Transition threshold ablation")
    print("-----------------------------")
    print("manifest:", args.transition_manifest)
    print("samples:", len(transition_rows) if args.limit <= 0 else min(args.limit, len(transition_rows)))
    print("checkpoint:", args.transition_checkpoint)

    with_thresholds = evaluate_transition_manifest(
        model,
        transition_rows,
        args.limit,
        localized_model=localized_model,
        localized_label_vocab=localized_labels,
        localized_thresholds=localized_thresholds,
        localized_index=localized_index,
        transition_thresholds=tuned_thresholds,
    )

    without_thresholds = evaluate_transition_manifest(
        model,
        transition_rows,
        args.limit,
        localized_model=localized_model,
        localized_label_vocab=localized_labels,
        localized_thresholds=localized_thresholds,
        localized_index=localized_index,
        transition_thresholds=None,
    )

    result = {
        "transition_manifest": args.transition_manifest,
        "transition_checkpoint": args.transition_checkpoint,
        "limit": args.limit,
        "with_thresholds": with_thresholds,
        "without_thresholds": without_thresholds,
        "delta": summarize_delta(with_thresholds, without_thresholds),
    }

    out_json = PROJECT_ROOT / args.output_json
    out_md = PROJECT_ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Transition threshold ablation")
    lines.append("")
    lines.append(f"- manifest: `{args.transition_manifest}`")
    lines.append(f"- checkpoint: `{args.transition_checkpoint}`")
    lines.append(f"- limit: {args.limit}")
    lines.append("")
    lines.append("| setting | samples | accuracy |")
    lines.append("|---|---:|---:|")
    lines.append(f"| with thresholds | {with_thresholds.get('samples')} | {with_thresholds.get('accuracy'):.3f} |")
    lines.append(f"| without thresholds / argmax | {without_thresholds.get('samples')} | {without_thresholds.get('accuracy'):.3f} |")
    lines.append("")
    lines.append("## Class summary: with thresholds")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, stats in with_thresholds.get("class_summary", {}).items():
        lines.append(f"| {label} | {stats.get('correct')} | {stats.get('total')} | {stats.get('accuracy'):.3f} |")
    lines.append("")
    lines.append("## Class summary: without thresholds")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, stats in without_thresholds.get("class_summary", {}).items():
        lines.append(f"| {label} | {stats.get('correct')} | {stats.get('total')} | {stats.get('accuracy'):.3f} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("with thresholds:", with_thresholds.get("accuracy"))
    print("without thresholds:", without_thresholds.get("accuracy"))
    print("delta:", result["delta"])
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
