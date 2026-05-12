from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    d = load_json(Path(args.suite_json))
    burst = d.get("burst", {})
    weighted = d.get("weighted_scoring", {})

    confusion = burst.get("confusion_matrix")
    class_summary = burst.get("class_summary", {})
    weighted_modules = {
        item.get("module"): item
        for item in weighted.get("module_summaries", [])
        if isinstance(item, dict)
    }

    # Fallback from known weighted scoring structure if module_summaries is not present.
    burst_weighted = None
    for e in weighted.get("errors", []):
        if isinstance(e, dict) and e.get("module") == "burst":
            burst_weighted = e

    result = {
        "samples": burst.get("samples"),
        "accuracy": burst.get("accuracy"),
        "class_summary": class_summary,
        "confusion_matrix": confusion,
        "weighted_score": weighted.get("estimated_average_score"),
        "num_errors": weighted.get("num_errors"),
        "burst_weighted": burst_weighted,
        "notes": [],
    }

    none_stats = class_summary.get("none", {})
    qalqalah_stats = class_summary.get("qalqalah", {})

    none_total = int(none_stats.get("total") or 0)
    none_correct = int(none_stats.get("correct") or 0)
    qal_total = int(qalqalah_stats.get("total") or 0)
    qal_correct = int(qalqalah_stats.get("correct") or 0)

    false_positives = none_total - none_correct
    false_negatives = qal_total - qal_correct

    result["false_positives_predicted_qalqalah_for_none"] = false_positives
    result["false_negatives_missed_qalqalah"] = false_negatives
    result["false_positive_share_of_burst_errors"] = false_positives / max(1, false_positives + false_negatives)
    result["false_negative_share_of_burst_errors"] = false_negatives / max(1, false_positives + false_negatives)

    if false_negatives > false_positives:
        result["dominant_error"] = "missing_qalqalah_false_negatives"
        result["notes"].append("Burst mainly misses qalqalah. Lowering the qalqalah threshold or improving recall may help.")
    elif false_positives > false_negatives:
        result["dominant_error"] = "weak_qalqalah_false_positives"
        result["notes"].append("Burst mainly over-predicts qalqalah. Raising the threshold may help.")
    else:
        result["dominant_error"] = "balanced_fp_fn"

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Burst diagnostics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| samples | {result['samples']} |")
    lines.append(f"| accuracy | {result['accuracy']:.3f} |")
    lines.append(f"| false positives: predicted qalqalah for none | {false_positives} |")
    lines.append(f"| false negatives: missed qalqalah | {false_negatives} |")
    lines.append(f"| false positive share | {result['false_positive_share_of_burst_errors']:.3f} |")
    lines.append(f"| false negative share | {result['false_negative_share_of_burst_errors']:.3f} |")
    lines.append("")
    lines.append("## Class summary")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, stats in class_summary.items():
        lines.append(
            f"| {label} | {stats.get('correct')} | {stats.get('total')} | {float(stats.get('accuracy') or 0):.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in result["notes"]:
        lines.append(f"- {note}")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
