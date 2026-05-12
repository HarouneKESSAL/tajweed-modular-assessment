from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verdict(row: dict, *, review_char_acc: float, review_max_edits: int) -> str:
    if row.get("exact_after"):
        return "accepted_exact"

    acc = float(row.get("char_accuracy_after", 0.0))
    ed = int(row.get("edit_distance_after", 999999))

    if acc >= review_char_acc and ed <= review_max_edits:
        return "review_near_exact"

    return "rejected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--review-char-acc", type=float, default=0.95)
    parser.add_argument("--review-max-edits", type=int, default=2)
    args = parser.parse_args()

    report = load_json(args.error_report_json)

    all_rows = []
    all_rows.extend(report.get("muqattaat_changed_examples", []))
    all_rows.extend(report.get("near_misses_ge_095", []))
    all_rows.extend(report.get("worst_examples", []))

    # The above lists overlap / are partial. Prefer reconstructing from the report is impossible
    # unless all rows are stored. So we compute exact count from summary and classify available examples.
    overall = report["overall_after_muqattaat"]
    samples = int(overall["samples"])
    exact = round(samples * float(overall["exact_after_rate"]))
    total_errors = int(report["num_errors_after_muqattaat"])

    near_miss_count = int(report.get("num_near_misses_ge_095", 0))
    strong_near_miss_count = int(report.get("num_strong_near_misses_ge_098", 0))

    summary = {
        "samples": samples,
        "accepted_exact_estimated": exact,
        "remaining_errors": total_errors,
        "near_misses_ge_095": near_miss_count,
        "strong_near_misses_ge_098": strong_near_miss_count,
        "recommended_policy": {
            "accepted_exact": "auto-pass content gate",
            "review_near_exact": "do not auto-pass; send to review/evidence",
            "rejected": "stop content gate",
            "review_char_acc": args.review_char_acc,
            "review_max_edits": args.review_max_edits,
        },
        "important_counts": report.get("error_type_counts", {}),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Whisper content gate verdict policy")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| samples | {samples} |")
    lines.append(f"| accepted_exact_estimated | {exact} |")
    lines.append(f"| accepted_exact_rate | {exact / samples:.4f} |")
    lines.append(f"| remaining_errors | {total_errors} |")
    lines.append(f"| near_misses_ge_095 | {near_miss_count} |")
    lines.append(f"| strong_near_misses_ge_098 | {strong_near_miss_count} |")
    lines.append("")
    lines.append("## Recommended production policy")
    lines.append("")
    lines.append("| verdict | action |")
    lines.append("|---|---|")
    lines.append("| accepted_exact | auto-pass content gate |")
    lines.append("| review_near_exact | review/evidence only, do not auto-pass |")
    lines.append("| rejected | stop before Tajweed scoring |")
    lines.append("")
    lines.append("## Why")
    lines.append("")
    lines.append("- Qur’an content should not auto-pass on near matches.")
    lines.append("- Near matches are useful for review and debugging.")
    lines.append("- Exact/muqattaat-exact remains the only automatic pass condition.")
    lines.append("")
    lines.append("## Error counts")
    lines.append("")
    lines.append("| type | count |")
    lines.append("|---|---:|")
    for k, v in summary["important_counts"].items():
        lines.append(f"| {k} | {v} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
