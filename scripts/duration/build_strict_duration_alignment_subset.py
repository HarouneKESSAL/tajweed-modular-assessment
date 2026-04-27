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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_alignment_prep.jsonl",
    )
    parser.add_argument(
        "--strict-output",
        default="data/manifests/retasy_duration_alignment_strict.jsonl",
    )
    parser.add_argument(
        "--weak-output",
        default="data/manifests/retasy_duration_alignment_weak.jsonl",
    )
    parser.add_argument(
        "--require-nonempty-projection",
        action="store_true",
        default=True,
        help="Keep only rows with at least one projected span in strict set",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    strict_path = PROJECT_ROOT / args.strict_output
    weak_path = PROJECT_ROOT / args.weak_output

    rows = load_jsonl(input_path)

    strict_rows = []
    weak_rows = []

    fine_counter_strict = Counter()
    fine_counter_weak = Counter()

    for row in rows:
        exact = bool(row.get("projection_exact_label_match", False))
        nonempty = len(row.get("duration_rule_spans_normalized", [])) > 0

        is_strict = exact and (nonempty if args.require_nonempty_projection else True)

        if is_strict:
            strict_rows.append(row)
            for label in row.get("gold_duration_labels", []):
                fine_counter_strict[label] += 1
        else:
            weak_rows.append(row)
            for label in row.get("gold_duration_labels", []):
                fine_counter_weak[label] += 1

    write_jsonl(strict_path, strict_rows)
    write_jsonl(weak_path, weak_rows)

    print(f"Input manifest   : {input_path}")
    print(f"Strict output    : {strict_path}")
    print(f"Weak output      : {weak_path}")
    print(f"Rows total       : {len(rows)}")
    print(f"Rows strict      : {len(strict_rows)}")
    print(f"Rows weak        : {len(weak_rows)}")
    print(f"Strict label cnt : {dict(fine_counter_strict)}")
    print(f"Weak label cnt   : {dict(fine_counter_weak)}")


if __name__ == "__main__":
    main()
