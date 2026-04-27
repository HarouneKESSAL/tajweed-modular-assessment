from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json


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
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.input)
    row = rows[args.index]

    print("ID            :", row["id"])
    print("Surah         :", row["surah_name"])
    print("Verse key     :", row["quranjson_verse_key"])
    print("Audio path    :", row["audio_path"])
    print("Original text :", row["original_text"])
    print("Normalized    :", row["normalized_text"])
    print("Gold labels   :", row["gold_duration_labels"])
    print()

    print("Projected spans:")
    for span in row["duration_rule_spans_normalized"]:
        snippet = row["normalized_text"][span["norm_start"]:span["norm_end"]]
        print(
            f"  rule={span['rule']:15s} "
            f"group={str(span['coarse_group']):10s} "
            f"norm=({span['norm_start']},{span['norm_end']}) "
            f"text='{snippet}'"
        )

    print()
    print("First 40 normalized chars with labels:")
    for item in row["normalized_char_labels"][:40]:
        print(
            f"{item['norm_index']:>3d} "
            f"char='{item['char']}' "
            f"orig_idx={item['original_index']:>3d} "
            f"rules={item['rules']}"
        )


if __name__ == "__main__":
    main()
