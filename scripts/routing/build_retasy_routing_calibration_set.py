from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def combo_for_row(row: dict[str, Any]) -> str:
    names = row.get("target_names", ["use_duration", "use_transition", "use_burst"])
    targets = row.get("targets", [0, 0, 0])

    active = [
        str(name)
        for name, value in zip(names, targets)
        if int(value) == 1
    ]

    return "+".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/learned_routing_dataset_v4_rule_aware_group_text.jsonl",
        help="Trusted Retasy routing dataset, preferably v4 rule-aware.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/learned_routing_retasy_calibration_balanced.jsonl",
    )
    parser.add_argument(
        "--max-per-combo",
        type=int,
        default=120,
        help="Maximum rows per routing combo.",
    )
    parser.add_argument(
        "--max-per-text-per-combo",
        type=int,
        default=25,
        help="Limit repeated same text inside each combo.",
    )
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(resolve_path(args.input))

    # Only trusted Retasy rows.
    rows = [
        row for row in rows
        if row.get("label_source", "trusted_retasy_routing") == "trusted_retasy_routing"
    ]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[combo_for_row(row)].append(row)

    selected: list[dict[str, Any]] = []

    for combo, combo_rows in sorted(buckets.items()):
        rng.shuffle(combo_rows)

        per_text_counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []

        for row in combo_rows:
            text = str(row.get("text", ""))
            if per_text_counts[text] >= args.max_per_text_per_combo:
                continue

            out = dict(row)
            out["split"] = "val"
            out["calibration_source"] = "trusted_retasy_balanced"
            out["calibration_combo"] = combo
            out["sample_weight"] = 1.0

            chosen.append(out)
            per_text_counts[text] += 1

            if len(chosen) >= args.max_per_combo:
                break

        selected.extend(chosen)

    rng.shuffle(selected)

    output = resolve_path(args.output)
    write_jsonl(output, selected)

    combo_counts = Counter(combo_for_row(row) for row in selected)
    target_counts = Counter()
    text_counts = Counter(row.get("text", "") for row in selected)

    for row in selected:
        for name, value in zip(row["target_names"], row["targets"]):
            if int(value) == 1:
                target_counts[name] += 1

    print("Built balanced trusted Retasy routing calibration set")
    print("-----------------------------------------------------")
    print(f"input_rows          : {len(rows)}")
    print(f"selected_rows       : {len(selected)}")
    print(f"combo_counts        : {dict(combo_counts)}")
    print(f"target_counts       : {dict(target_counts)}")
    print(f"unique_texts        : {len(text_counts)}")
    print(f"max_text_repetition : {max(text_counts.values()) if text_counts else 0}")
    print(f"output              : {output}")


if __name__ == "__main__":
    main()
