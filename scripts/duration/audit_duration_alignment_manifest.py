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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_alignment_prep.jsonl",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    rows = load_jsonl(input_path)

    total = len(rows)
    rows_with_gold = 0
    exact_label_match = 0
    rows_missing_labels = 0
    rows_extra_labels = 0
    empty_projection_with_gold = 0
    spans_with_spaces = 0

    gold_counter = Counter()
    projected_counter = Counter()
    missing_counter = Counter()
    extra_counter = Counter()

    bad_examples = []

    for row in rows:
        gold = set(row.get("gold_duration_labels", []))
        proj = set(row.get("projected_duration_labels", []))

        if gold:
            rows_with_gold += 1

        if gold == proj:
            exact_label_match += 1

        missing = sorted(gold - proj)
        extra = sorted(proj - gold)

        if missing:
            rows_missing_labels += 1
            for label in missing:
                missing_counter[label] += 1

        if extra:
            rows_extra_labels += 1
            for label in extra:
                extra_counter[label] += 1

        if gold and not row.get("duration_rule_spans_normalized"):
            empty_projection_with_gold += 1

        for label in gold:
            gold_counter[label] += 1
        for label in proj:
            projected_counter[label] += 1

        has_space_span = False
        for span in row.get("duration_rule_spans_normalized", []):
            if span.get("contains_space", False):
                spans_with_spaces += 1
                has_space_span = True

        if (missing or extra or has_space_span) and len(bad_examples) < args.max_examples:
            bad_examples.append(
                {
                    "id": row["id"],
                    "surah_name": row.get("surah_name"),
                    "verse_key": row.get("quranjson_verse_key"),
                    "text": row.get("normalized_text"),
                    "gold": sorted(gold),
                    "projected": sorted(proj),
                    "missing": missing,
                    "extra": extra,
                    "projected_spans": row.get("duration_rule_spans_normalized", []),
                }
            )

    print(f"Input manifest              : {input_path}")
    print(f"Rows total                  : {total}")
    print(f"Rows with gold labels       : {rows_with_gold}")
    print(f"Exact label match rows      : {exact_label_match} ({exact_label_match / max(total, 1):.3f})")
    print(f"Rows missing projected      : {rows_missing_labels}")
    print(f"Rows with extra projected   : {rows_extra_labels}")
    print(f"Empty projection with gold  : {empty_projection_with_gold}")
    print(f"Projected spans with spaces : {spans_with_spaces}")
    print()

    print("Gold label counts:")
    for label, count in gold_counter.items():
        print(f"  {label:15s} {count}")

    print("Projected label counts:")
    for label, count in projected_counter.items():
        print(f"  {label:15s} {count}")

    print("Missing label counts:")
    for label, count in missing_counter.items():
        print(f"  {label:15s} {count}")

    print("Extra label counts:")
    for label, count in extra_counter.items():
        print(f"  {label:15s} {count}")

    print("\nSample problematic rows:")
    for i, ex in enumerate(bad_examples, start=1):
        print("-" * 80)
        print(f"[{i}] {ex['id']} | {ex['surah_name']} | {ex['verse_key']}")
        print(f"    text      : {ex['text']}")
        print(f"    gold      : {ex['gold']}")
        print(f"    projected : {ex['projected']}")
        print(f"    missing   : {ex['missing']}")
        print(f"    extra     : {ex['extra']}")
        for span in ex["projected_spans"][:10]:
            print(
                f"      rule={span['rule']:15s} "
                f"norm=({span['norm_start']},{span['norm_end']}) "
                f"text='{span.get('text', '')}' "
                f"space={span.get('contains_space', False)}"
            )


if __name__ == "__main__":
    main()
