from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_coarse_labels(duration_rules):
    duration_rules = [str(x).strip().lower() for x in duration_rules]

    coarse = []

    if any(rule.startswith("madd") for rule in duration_rules):
        coarse.append("has_madd")

    if "ghunnah" in duration_rules:
        coarse.append("ghunnah")

    return coarse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_subset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_duration_subset_coarse.jsonl",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)

    out_rows = []
    coarse_counter = Counter()
    combo_counter = Counter()

    for row in rows:
        fine_rules = list(row.get("duration_rules", []))
        coarse_rules = to_coarse_labels(fine_rules)

        out_row = dict(row)
        out_row["duration_rules_fine"] = fine_rules
        out_row["duration_rules"] = coarse_rules  # overwrite for easy reuse
        out_row["duration_rules_coarse"] = coarse_rules

        out_rows.append(out_row)

        for label in coarse_rules:
            coarse_counter[label] += 1

        combo_counter[tuple(sorted(coarse_rules))] += 1

    write_jsonl(output_path, out_rows)

    print(f"Input manifest  : {input_path}")
    print(f"Output manifest : {output_path}")
    print(f"Rows            : {len(out_rows)}")
    print(f"Coarse counts   : {dict(coarse_counter)}")
    print(f"Label combos    : {dict(combo_counter)}")


if __name__ == "__main__":
    main()
