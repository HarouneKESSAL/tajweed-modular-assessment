from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random
from collections import Counter

from tajweed_assessment.data.labels import TRANSITION_RULES


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


def extract_transition_rules(rule_spans):
    labels = []
    for span in rule_spans or []:
        rule = str(span.get("rule", "")).strip().lower()
        if "ikhfa" in rule:
            labels.append("ikhfa")
        elif "idgham" in rule or "idghaam" in rule:
            labels.append(rule)
    normalized = []
    for label in labels:
        if label == "ikhfa":
            normalized.append("ikhfa")
        else:
            normalized.append("idgham")
    return sorted(set(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_quranjson_train.jsonl",
        help="Merged QuranJSON/Retasy manifest with raw rule_spans",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_transition_subset.jsonl",
        help="Output JSONL for transition training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--none-multiplier",
        type=float,
        default=1.5,
        help="Maximum none rows relative to the number of positive rows",
    )
    parser.add_argument(
        "--keep-not-related-quran",
        action="store_true",
        help="Keep rows labeled not_related_quran",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)

    positive_rows = []
    none_rows = []
    skipped_ambiguous = 0
    skipped_match = 0
    skipped_not_related = 0

    for row in rows:
        if row.get("match_status") != "matched_unique":
            skipped_match += 1
            continue

        if (not args.keep_not_related_quran) and row.get("final_label") == "not_related_quran":
            skipped_not_related += 1
            continue

        transition_rules = extract_transition_rules(row.get("rule_spans", []))
        if len(transition_rules) > 1:
            skipped_ambiguous += 1
            continue

        label = transition_rules[0] if transition_rules else "none"
        out_row = {
            "sample_id": row["id"],
            "audio_path": row.get("audio_path", ""),
            "feature_path": "",
            "canonical_phonemes": None,
            "canonical_rules": [label],
            "text": row.get("aya_text_norm") or row.get("aya_text") or "",
            "reciter_id": row.get("reciter_id"),
            "surah_name": row.get("surah_name"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
            "transition_rules": transition_rules,
            "rule_spans": row.get("rule_spans", []),
        }

        if label == "none":
            none_rows.append(out_row)
        else:
            positive_rows.append(out_row)

    rng = random.Random(args.seed)
    rng.shuffle(none_rows)

    max_none = int(round(len(positive_rows) * args.none_multiplier))
    if len(positive_rows) == 0:
        selected_none = []
    else:
        selected_none = none_rows[:max_none]

    final_rows = positive_rows + selected_none
    rng.shuffle(final_rows)

    label_counter = Counter()
    surah_counter = Counter()
    reciter_counter = Counter()

    for row in final_rows:
        label = (row.get("canonical_rules") or ["none"])[0]
        label_counter[label] += 1
        surah_counter[row.get("surah_name") or "Unknown"] += 1
        reciter_counter[row.get("reciter_id") or "Unknown"] += 1

    write_jsonl(output_path, final_rows)

    print(f"Input manifest         : {input_path}")
    print(f"Output manifest        : {output_path}")
    print(f"Rows in input          : {len(rows)}")
    print(f"Positive rows kept     : {len(positive_rows)}")
    print(f"None rows available    : {len(none_rows)}")
    print(f"None rows selected     : {len(selected_none)}")
    print(f"Rows in output         : {len(final_rows)}")
    print(f"Skipped non-unique     : {skipped_match}")
    print(f"Skipped not_related    : {skipped_not_related}")
    print(f"Skipped ambiguous      : {skipped_ambiguous}")
    print(f"Transition label count : {dict((label, label_counter.get(label, 0)) for label in TRANSITION_RULES)}")
    print(f"Top surahs             : {surah_counter.most_common(10)}")
    print(f"Top reciters           : {reciter_counter.most_common(10)}")


if __name__ == "__main__":
    main()

