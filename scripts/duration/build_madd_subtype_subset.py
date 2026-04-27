from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter


MADD_LABELS = {
    "madd_2",
    "madd_4",
    "madd_6",
    "madd_246",
    "madd_munfasil",
    "madd_muttasil",
}


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


def extract_madd_labels(duration_rules):
    labels = []
    for rule in duration_rules:
        rule = str(rule).strip().lower()
        if rule in MADD_LABELS:
            labels.append(rule)
    return sorted(set(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_subset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_madd_subtype_subset.jsonl",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)

    out_rows = []
    madd_counter = Counter()
    combo_counter = Counter()

    for row in rows:
        fine_rules = list(row.get("duration_rules", []))
        madd_rules = extract_madd_labels(fine_rules)

        # keep only madd-positive rows
        if not madd_rules:
            continue

        out_row = dict(row)
        out_row["duration_rules_original"] = fine_rules
        out_row["duration_rules"] = madd_rules
        out_row["madd_subtype_rules"] = madd_rules

        out_rows.append(out_row)

        for label in madd_rules:
            madd_counter[label] += 1

        combo_counter[tuple(sorted(madd_rules))] += 1

    write_jsonl(output_path, out_rows)

    print(f"Input manifest   : {input_path}")
    print(f"Output manifest  : {output_path}")
    print(f"Rows in input    : {len(rows)}")
    print(f"Rows in subset   : {len(out_rows)}")
    print(f"Madd label counts: {dict(madd_counter)}")
    print(f"Top label combos : {combo_counter.most_common(20)}")


if __name__ == "__main__":
    main()
