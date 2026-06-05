from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODULES = ["duration", "transition", "burst"]

RULE_TO_MODULE_KEYWORDS = {
    "duration": [
        "madd",
        "mad",
        "مد",
        "ghunnah",
        "gunnah",
        "ghonna",
        "غنة",
        "غن",
    ],
    "transition": [
        "ikhfa",
        "ikhfaa",
        "إخفاء",
        "اخفاء",
        "idgham",
        "إدغام",
        "ادغام",
    ],
    "burst": [
        "qalqalah",
        "qalkalah",
        "قلقلة",
    ],
}


TEXT_KEYS = [
    "text",
    "content_text",
    "uthmani_text",
    "verse",
    "ayah_text",
    "clean_text",
]


RULE_KEYS = [
    "rule",
    "rule_id",
    "rule_name",
    "rule_type",
    "tajweed_rule",
    "label",
    "class",
    "type",
]


START_KEYS = [
    "start",
    "start_index",
    "start_char",
    "char_start",
    "begin",
]

END_KEYS = [
    "end",
    "end_index",
    "end_char",
    "char_end",
    "stop",
]


def compact_len(text: str) -> int:
    return len("".join(str(text or "").split()))


def get_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in [None, ""]:
            return row[key]
    return default


def get_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # Try common nested structures.
    for key in ["ayah", "reference", "quran", "data"]:
        value = row.get(key)
        if isinstance(value, dict):
            nested = get_text(value)
            if nested:
                return nested

    return ""


def module_for_rule(rule: Any) -> str | None:
    rule_text = str(rule or "").strip().lower()

    if not rule_text:
        return None

    for module, keywords in RULE_TO_MODULE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in rule_text:
                return module

    return None


def find_rule_value(obj: dict[str, Any]) -> Any:
    for key in RULE_KEYS:
        if key in obj and obj[key] not in [None, ""]:
            return obj[key]
    return None


def find_int_value(obj: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        if key not in obj:
            continue
        try:
            return int(obj[key])
        except Exception:
            continue
    return None


def collect_rule_items(obj: Any, out: list[dict[str, Any]]) -> None:
    """
    Recursively collect possible Tajweed rule annotations.

    Works with schemas like:
      {"rule": "madd", "start": 10, "end": 12}
      {"rule_id": "ikhfa", "start_index": 5, "end_index": 6}
      nested lists/dicts.
    """
    if isinstance(obj, dict):
        rule = find_rule_value(obj)

        if rule is not None:
            start = find_int_value(obj, START_KEYS)
            end = find_int_value(obj, END_KEYS)

            out.append(
                {
                    "rule": rule,
                    "start": start,
                    "end": end,
                }
            )

        for value in obj.values():
            collect_rule_items(value, out)

    elif isinstance(obj, list):
        for item in obj:
            collect_rule_items(item, out)


def safe_range(start: int | None, end: int | None, text_length: int) -> set[int]:
    if text_length <= 0:
        text_length = 1

    if start is None:
        return {0}

    start = max(0, min(start, text_length - 1))

    if end is None:
        return {start}

    end = max(0, min(end, text_length))

    # If end is exclusive and larger than start.
    if end > start:
        return set(range(start, end))

    return {start}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_module_calls(summary_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    labels = [row["module"] for row in summary_rows]
    aware = [float(row["rule_aware_calls"]) for row in summary_rows]
    agnostic = [float(row["rule_agnostic_calls"]) for row in summary_rows]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    plt.bar([i - width / 2 for i in x], aware, width, label="Rule-aware routing")
    plt.bar([i + width / 2 for i in x], agnostic, width, label="Rule-agnostic routing")

    plt.xticks(list(x), labels)
    plt.ylabel("Number of module calls")
    plt.title("Routing ablation: module calls")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(figure_dir / "routing_ablation_module_calls.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / "routing_ablation_module_calls.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_false_exposure(summary_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    labels = [row["module"] for row in summary_rows]
    false_exposure = [float(row["agnostic_irrelevant_calls"]) for row in summary_rows]

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(labels, false_exposure)

    plt.ylabel("Irrelevant module calls")
    plt.title("Rule-agnostic false candidate exposure")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, false_exposure):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(false_exposure) * 0.01 if max(false_exposure) else value + 1,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(figure_dir / "routing_ablation_false_exposure.pdf", bbox_inches="tight")
    plt.savefig(figure_dir / "routing_ablation_false_exposure.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Tajweed reference JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/analysis/routing_ablation"),
        type=Path,
    )
    parser.add_argument(
        "--figure-dir",
        default=Path("figures/experiments/routing_ablation"),
        type=Path,
    )
    args = parser.parse_args()

    rows = read_jsonl(args.reference)

    total_positions = 0
    aware_positions_by_module: dict[str, set[tuple[int, int]]] = {
        module: set() for module in MODULES
    }

    rule_counts = Counter()
    unknown_rule_counts = Counter()
    by_surah = defaultdict(lambda: {"positions": 0, "duration": 0, "transition": 0, "burst": 0})

    for row_idx, row in enumerate(rows):
        text = get_text(row)
        text_length = compact_len(text)

        if text_length == 0:
            # Fallback: avoid division by zero if the reference row has only spans.
            text_length = 1

        total_positions += text_length

        surah = row.get("surah") or row.get("sura") or row.get("chapter") or "unknown"

        items: list[dict[str, Any]] = []
        collect_rule_items(row, items)

        by_surah[str(surah)]["positions"] += text_length

        for item in items:
            rule = item["rule"]
            module = module_for_rule(rule)

            if module is None:
                unknown_rule_counts[str(rule)] += 1
                continue

            rule_counts[str(rule)] += 1

            positions = safe_range(item["start"], item["end"], text_length)

            for pos in positions:
                aware_positions_by_module[module].add((row_idx, pos))

            by_surah[str(surah)][module] += len(positions)

    summary_rows: list[dict[str, Any]] = []

    total_aware_calls = 0
    total_agnostic_calls = 0

    for module in MODULES:
        aware_calls = len(aware_positions_by_module[module])
        agnostic_calls = total_positions
        irrelevant_calls = agnostic_calls - aware_calls
        reduction = 1.0 - (aware_calls / agnostic_calls) if agnostic_calls else 0.0

        total_aware_calls += aware_calls
        total_agnostic_calls += agnostic_calls

        summary_rows.append(
            {
                "module": module,
                "rule_aware_calls": aware_calls,
                "rule_agnostic_calls": agnostic_calls,
                "agnostic_irrelevant_calls": irrelevant_calls,
                "call_reduction_percent": round(reduction * 100, 2),
            }
        )

    overall_reduction = (
        1.0 - (total_aware_calls / total_agnostic_calls)
        if total_agnostic_calls
        else 0.0
    )

    overall_row = {
        "module": "overall",
        "rule_aware_calls": total_aware_calls,
        "rule_agnostic_calls": total_agnostic_calls,
        "agnostic_irrelevant_calls": total_agnostic_calls - total_aware_calls,
        "call_reduction_percent": round(overall_reduction * 100, 2),
    }

    save_csv(args.output_dir / "routing_ablation_summary.csv", summary_rows + [overall_row])

    by_surah_rows = []
    for surah, data in sorted(by_surah.items(), key=lambda x: str(x[0])):
        positions = int(data["positions"])
        aware = int(data["duration"] + data["transition"] + data["burst"])
        agnostic = positions * len(MODULES)
        reduction = 1.0 - (aware / agnostic) if agnostic else 0.0

        by_surah_rows.append(
            {
                "surah": surah,
                "positions": positions,
                "rule_aware_calls": aware,
                "rule_agnostic_calls": agnostic,
                "call_reduction_percent": round(reduction * 100, 2),
                "duration_calls": int(data["duration"]),
                "transition_calls": int(data["transition"]),
                "burst_calls": int(data["burst"]),
            }
        )

    save_csv(args.output_dir / "routing_ablation_by_surah.csv", by_surah_rows)

    rule_count_rows = [
        {"rule": rule, "count": count}
        for rule, count in rule_counts.most_common()
    ]
    save_csv(args.output_dir / "routing_ablation_rule_counts.csv", rule_count_rows)

    unknown_rows = [
        {"rule": rule, "count": count}
        for rule, count in unknown_rule_counts.most_common()
    ]
    save_csv(args.output_dir / "routing_ablation_unknown_rules.csv", unknown_rows)

    plot_module_calls(summary_rows, args.figure_dir)
    plot_false_exposure(summary_rows, args.figure_dir)

    print("\n=== Routing ablation summary ===")
    for row in summary_rows + [overall_row]:
        print(
            f"{row['module']}: "
            f"aware={row['rule_aware_calls']} "
            f"agnostic={row['rule_agnostic_calls']} "
            f"irrelevant={row['agnostic_irrelevant_calls']} "
            f"reduction={row['call_reduction_percent']}%"
        )

    print(f"\nWrote: {args.output_dir / 'routing_ablation_summary.csv'}")
    print(f"Wrote: {args.output_dir / 'routing_ablation_by_surah.csv'}")
    print(f"Wrote figures to: {args.figure_dir}")

    if total_aware_calls == 0:
        print("\nWARNING: No supported rules were detected.")
        print("Check routing_ablation_unknown_rules.csv to see the rule names in your reference.")


if __name__ == "__main__":
    main()