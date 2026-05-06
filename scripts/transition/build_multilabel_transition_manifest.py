from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.models.transition.multilabel_transition_module import (
    transition_multilabel_label_names,
    transition_rules_to_multihot,
)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_transition_rules(row: dict[str, Any]) -> list[str]:
    raw_rules = row.get("transition_rules") or []
    if not isinstance(raw_rules, list):
        raw_rules = [raw_rules]

    out: list[str] = []
    for rule in raw_rules:
        rule = str(rule).lower().strip()
        if rule.startswith("ikhfa"):
            out.append("ikhfa")
        elif rule.startswith("idgham"):
            out.append("idgham")

    # stable unique order
    ordered = []
    for label in transition_multilabel_label_names():
        if label in out:
            ordered.append(label)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_transition_subset.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_transition_multilabel.jsonl")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    rows = load_jsonl(input_path)
    label_names = transition_multilabel_label_names()

    out_rows: list[dict[str, Any]] = []
    combo_counts: Counter[str] = Counter()

    for row in rows:
        rules = normalize_transition_rules(row)
        target = transition_rules_to_multihot(rules, label_names=label_names)
        combo = "+".join(rules) if rules else "none"
        combo_counts[combo] += 1

        out_row = dict(row)
        out_row["id"] = row.get("id") or row.get("sample_id")
        out_row["transition_label_names"] = label_names
        out_row["transition_multilabel_rules"] = rules
        out_row["transition_multihot"] = target
        out_row["transition_combo"] = combo

        out_rows.append(out_row)

    write_jsonl(output_path, out_rows)

    print("Built multi-label transition manifest")
    print("-------------------------------------")
    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"rows  : {len(out_rows)}")
    print(f"labels: {label_names}")
    print(f"combo_counts: {dict(combo_counts)}")


if __name__ == "__main__":
    main()
