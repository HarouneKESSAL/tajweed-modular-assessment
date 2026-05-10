from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_dict_with_key(obj: Any, key: str) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj
        for value in obj.values():
            found = find_dict_with_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_dict_with_key(item, key)
            if found is not None:
                return found
    return None


def find_number(obj: Any, keys: list[str], default: float | None = None) -> float | None:
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in obj.values():
            found = find_number(value, keys, None)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_number(item, keys, None)
            if found is not None:
                return found
    return default


def pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modular-suite-json", required=True)
    parser.add_argument("--ayah-batch-summary-json", required=True)
    parser.add_argument("--expected-ctc-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    modular = load_json(Path(args.modular_suite_json))
    ayah_batch = load_json(Path(args.ayah_batch_summary_json))
    expected_ctc = load_json(Path(args.expected_ctc_summary_json))

    weighted = find_dict_with_key(modular, "estimated_average_score") or {}
    duration = find_dict_with_key(modular, "duration") or find_dict_with_key(modular, "duration_accuracy") or {}
    transition = find_dict_with_key(modular, "transition") or find_dict_with_key(modular, "transition_accuracy") or {}
    burst = find_dict_with_key(modular, "burst") or find_dict_with_key(modular, "burst_accuracy") or {}
    content = find_dict_with_key(modular, "char_accuracy") or {}

    # Verified full-system smoke baseline.
    # Keep these explicit because the modular-suite JSON contains repeated generic
    # keys like "acc", and naive recursive lookup can accidentally copy duration
    # accuracy into transition/burst.
    baseline = {
        "weighted_score": 98.591,
        "duration_acc": 0.993,
        "transition_acc": 0.901,
        "burst_acc": 0.874,
        "chunk_content_char_accuracy": 0.893,
        "chunk_content_exact_match": 0.707,
        "critical_errors": 122,
        "num_errors": 403,
    }

    # Current weighted-error contribution from modular smoke log.
    # These are not true "module-off" results; they are contribution estimates.
    module_contrib = [
        {
            "module": "duration",
            "errors": 12,
            "weighted_sum": 30.0,
            "severity": "medium/minor",
            "current_acc": baseline["duration_acc"],
        },
        {
            "module": "transition",
            "errors": 68,
            "weighted_sum": 272.0,
            "severity": "medium",
            "current_acc": baseline["transition_acc"],
        },
        {
            "module": "burst",
            "errors": 201,
            "weighted_sum": 521.0,
            "severity": "minor/medium",
            "current_acc": baseline["burst_acc"],
        },
        {
            "module": "chunk_content",
            "errors": 122,
            "weighted_sum": 1220.0,
            "severity": "critical",
            "current_acc": baseline["chunk_content_char_accuracy"],
        },
    ]

    total_weight = sum(x["weighted_sum"] for x in module_contrib)
    for row in module_contrib:
        row["weighted_share"] = row["weighted_sum"] / total_weight if total_weight else 0.0

    ayah_overall = ayah_batch["overall"]
    expected_overall = expected_ctc["overall"]

    result = {
        "baseline": baseline,
        "module_contribution_estimate": module_contrib,
        "ayah_strict_batch": ayah_overall,
        "expected_text_ctc": expected_overall,
        "interpretation": {
            "highest_weight_module": max(module_contrib, key=lambda x: x["weighted_sum"])["module"],
            "highest_error_count_module": max(module_contrib, key=lambda x: x["errors"])["module"],
            "recommended_next_real_ablations": [
                "chunk_content decoder/config ablation",
                "transition localizer ablation",
                "burst threshold/localization ablation",
                "ayah strict free-decode vs expected-text CTC review evidence",
            ],
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Pilot Ablation Study")
    lines.append("")
    lines.append("This is a pilot ablation summary. Module contribution rows are **not** true module-off reruns yet; they estimate which modules carry error weight in the verified full-system smoke.")
    lines.append("")
    lines.append("## Full modular baseline")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key, value in baseline.items():
        if value is not None:
            lines.append(f"| {key} | {value:.3f} |" if isinstance(value, float) else f"| {key} | {value} |")

    lines.append("")
    lines.append("## Module contribution estimate")
    lines.append("")
    lines.append("| module | current_acc | errors | weighted_sum | weighted_share | severity |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in sorted(module_contrib, key=lambda x: x["weighted_sum"], reverse=True):
        lines.append(
            f"| {row['module']} | {row['current_acc']:.3f} | {row['errors']} | {row['weighted_sum']:.1f} | {row['weighted_share']:.3f} | {row['severity']} |"
        )

    lines.append("")
    lines.append("## Ayah strict batch scoring")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in [
        "samples",
        "avg_score",
        "avg_char_accuracy",
        "avg_edit_distance",
        "exact_rate",
        "accepted_rate",
    ]:
        value = ayah_overall.get(key)
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append(f"| acceptance_counts | `{ayah_overall.get('acceptance_counts')}` |")
    lines.append(f"| quality_counts | `{ayah_overall.get('quality_counts')}` |")

    lines.append("")
    lines.append("## Expected-text CTC analysis")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in [
        "samples",
        "free_exact_rate",
        "expected_text_accepted_rate",
        "expected_text_strong_review_rate",
        "expected_text_plausible_review_rate",
        "avg_free_char_accuracy",
        "avg_expected_ctc_loss_per_char",
        "avg_expected_ctc_confidence",
    ]:
        value = expected_overall.get(key)
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append(f"| verdict_counts | `{expected_overall.get('verdict_counts')}` |")

    lines.append("")
    lines.append("## Pilot conclusions")
    lines.append("")
    lines.append("- Chunk content carries the largest weighted error contribution because content errors are critical.")
    lines.append("- Burst has the largest error count, but lower severity weight than content.")
    lines.append("- Duration is already very strong and likely low priority for deeper ablation.")
    lines.append("- Transition is worth localizer/threshold ablation because accuracy is good but local support metrics are mixed.")
    lines.append("- Ayah strict acceptance is intentionally conservative; expected-text CTC is useful as review evidence, not live acceptance.")
    lines.append("")
    lines.append("## Recommended real ablations next")
    lines.append("")
    lines.append("1. Chunk decoder: tuned blank penalty vs old/default decoder.")
    lines.append("2. Transition: whole-verse only vs localizer-assisted.")
    lines.append("3. Burst: threshold/localizer variants.")
    lines.append("4. Ayah: strict free decode vs expected-text review evidence.")
    lines.append("5. Only after these: tiny feature/routing ablations.")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Pilot ablation summary")
    print("----------------------")
    print(json.dumps(result["baseline"], ensure_ascii=False, indent=2))
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
