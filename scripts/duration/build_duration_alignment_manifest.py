from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from tajweed_assessment.alignment.prep import (
    load_jsonl,
    prepare_alignment_records,
)
from dataclasses import asdict


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_subset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_duration_alignment_prep.jsonl",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 means all rows",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    prepared = prepare_alignment_records(rows)
    prepared_dicts = [asdict(x) for x in prepared]

    write_jsonl(output_path, prepared_dicts)

    label_counter = Counter()
    coarse_counter = Counter()
    char_count = 0

    for row in prepared_dicts:
        char_count += len(row["normalized_text"])
        for label in row["gold_duration_labels"]:
            label_counter[label] += 1
        for span in row["duration_rule_spans_normalized"]:
            cg = span.get("coarse_group")
            if cg:
                coarse_counter[cg] += 1

    avg_len = char_count / max(len(prepared_dicts), 1)

    print(f"Input manifest     : {input_path}")
    print(f"Output manifest    : {output_path}")
    print(f"Prepared rows      : {len(prepared_dicts)}")
    print(f"Average norm chars : {avg_len:.2f}")
    print(f"Fine labels        : {dict(label_counter)}")
    print(f"Coarse span groups : {dict(coarse_counter)}")


if __name__ == "__main__":
    main()
