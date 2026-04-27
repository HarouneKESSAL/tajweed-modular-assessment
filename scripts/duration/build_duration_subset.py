from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from tajweed_assessment.data.real_duration_dataset import (
    RealDurationDataset,
    extract_duration_rules,
    load_jsonl,
)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_quranjson_train.jsonl",
        help="Merged manifest created by merge_manifest.py",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_duration_subset.jsonl",
        help="Output JSONL for matched duration rows",
    )
    parser.add_argument(
        "--keep-ambiguous",
        action="store_true",
        help="Keep matched_ambiguous rows too (default: only matched_unique)",
    )
    parser.add_argument(
        "--keep-empty-rules",
        action="store_true",
        help="Keep rows even if they have no duration rules",
    )
    parser.add_argument(
        "--keep-not-related-quran",
        action="store_true",
        help="Keep rows whose final_label is not_related_quran",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)

    filtered = []
    match_counter = Counter()
    label_counter = Counter()
    surah_counter = Counter()

    for row in rows:
        match_status = row.get("match_status", "unknown")
        match_counter[match_status] += 1

        if not args.keep_ambiguous and match_status != "matched_unique":
            continue

        if not args.keep_not_related_quran and row.get("final_label") == "not_related_quran":
            continue

        duration_rules = extract_duration_rules(row.get("rule_spans", []))
        if not args.keep_empty_rules and not duration_rules:
            continue

        out_row = {
            "id": row["id"],
            "hf_index": row["hf_index"],
            "surah_name": row.get("surah_name"),
            "hf_surah_number": row.get("hf_surah_number"),
            "quranjson_surah_number": row.get("quranjson_surah_number"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
            "quranjson_verse_index": row.get("quranjson_verse_index"),
            "aya_text": row.get("aya_text"),
            "aya_text_norm": row.get("aya_text_norm"),
            "duration_ms": row.get("duration_ms"),
            "final_label": row.get("final_label"),
            "golden": row.get("golden"),
            "reciter_id": row.get("reciter_id"),
            "reciter_country": row.get("reciter_country"),
            "reciter_gender": row.get("reciter_gender"),
            "reciter_age": row.get("reciter_age"),
            "reciter_qiraah": row.get("reciter_qiraah"),
            "original_audio_path": row.get("original_audio_path"),
            "audio_path": row.get("audio_path"),
            "duration_rules": duration_rules,
            "rule_spans": row.get("rule_spans", []),
            "annotation_metadata": row.get("annotation_metadata", {}),
        }

        filtered.append(out_row)

        for rule in duration_rules:
            label_counter[rule] += 1

        surah_name = row.get("surah_name") or "Unknown"
        surah_counter[surah_name] += 1

    write_jsonl(output_path, filtered)

    print(f"Input manifest   : {input_path}")
    print(f"Output subset    : {output_path}")
    print(f"Rows in input    : {len(rows)}")
    print(f"Rows in subset   : {len(filtered)}")
    print(f"Match counts     : {dict(match_counter)}")
    print(f"Duration labels  : {dict(label_counter)}")
    print(f"Top surahs       : {surah_counter.most_common(10)}")

    # Sanity-check with the dataset wrapper
    dataset = RealDurationDataset(
        output_path,
        require_unique_match=False,   # already filtered
        require_nonempty_rules=not args.keep_empty_rules,
        drop_not_related_quran=not args.keep_not_related_quran,
    )
    print("Dataset summary  :", dataset.summary())


if __name__ == "__main__":
    main()
