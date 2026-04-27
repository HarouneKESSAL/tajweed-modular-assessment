from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random
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


def has_qalqalah(rule_spans):
    for span in rule_spans or []:
        if str(span.get("rule", "")).strip().lower() == "qalqalah":
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_quranjson_train.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_burst_subset.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--negative-multiplier",
        type=float,
        default=1.5,
        help="Maximum negative rows relative to positive rows",
    )
    parser.add_argument(
        "--keep-not-related-quran",
        action="store_true",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)
    positives = []
    negatives = []
    skipped_non_unique = 0
    skipped_not_related = 0

    for row in rows:
        if row.get("match_status") != "matched_unique":
            skipped_non_unique += 1
            continue

        if (not args.keep_not_related_quran) and row.get("final_label") == "not_related_quran":
            skipped_not_related += 1
            continue

        label = 1 if has_qalqalah(row.get("rule_spans", [])) else 0
        out_row = {
            "sample_id": row["id"],
            "audio_path": row.get("audio_path", ""),
            "feature_path": "",
            "canonical_phonemes": None,
            "canonical_rules": ["qalqalah" if label == 1 else "none"],
            "text": row.get("aya_text_norm") or row.get("aya_text") or "",
            "reciter_id": row.get("reciter_id"),
            "surah_name": row.get("surah_name"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
            "burst_label": int(label),
            "rule_spans": row.get("rule_spans", []),
        }
        if label == 1:
            positives.append(out_row)
        else:
            negatives.append(out_row)

    rng = random.Random(args.seed)
    rng.shuffle(negatives)

    max_negatives = int(round(len(positives) * args.negative_multiplier))
    selected_negatives = negatives[:max_negatives]
    final_rows = positives + selected_negatives
    rng.shuffle(final_rows)

    label_counter = Counter(row["burst_label"] for row in final_rows)
    surah_counter = Counter(row.get("surah_name") or "Unknown" for row in final_rows)
    reciter_counter = Counter(row.get("reciter_id") or "Unknown" for row in final_rows)

    write_jsonl(output_path, final_rows)

    print(f"Input manifest      : {input_path}")
    print(f"Output manifest     : {output_path}")
    print(f"Rows in input       : {len(rows)}")
    print(f"Positive rows kept  : {len(positives)}")
    print(f"Negative available  : {len(negatives)}")
    print(f"Negative selected   : {len(selected_negatives)}")
    print(f"Rows in output      : {len(final_rows)}")
    print(f"Skipped non-unique  : {skipped_non_unique}")
    print(f"Skipped not_related : {skipped_not_related}")
    print(f"Label counts        : {dict(label_counter)}")
    print(f"Top surahs          : {surah_counter.most_common(10)}")
    print(f"Top reciters        : {reciter_counter.most_common(10)}")


if __name__ == "__main__":
    main()

