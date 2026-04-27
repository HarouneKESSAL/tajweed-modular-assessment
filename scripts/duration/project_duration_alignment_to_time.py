from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from tajweed_assessment.alignment.time_projection import (
    load_jsonl,
    project_row_to_time,
)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/alignment/torchaudio_forced_alignment_preview.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/alignment/duration_time_projection_preview.jsonl",
    )
    parser.add_argument(
        "--uroman-cmd",
        default="",
        help="Optional custom uroman command. Defaults to current_python -m uroman",
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

    out_rows = []

    fully_timed_counter = Counter()
    rule_counter = Counter()
    exact_char_match_counter = Counter()

    for row in rows:
        out = project_row_to_time(row, uroman_cmd=args.uroman_cmd)
        out_rows.append(out)

        stats = out["alignment_stats"]
        exact_char_match_counter[stats["exact_match_count"] == stats["source_len"]] += 1

        for span in out["duration_rule_time_spans"]:
            rule_counter[span["rule"]] += 1
            fully_timed_counter[span["fully_timed"]] += 1

    write_jsonl(output_path, out_rows)

    print(f"Input alignment file : {input_path}")
    print(f"Output projection    : {output_path}")
    print(f"Rows written         : {len(out_rows)}")
    print(f"Rule counts          : {dict(rule_counter)}")
    print(f"Fully timed spans    : {dict(fully_timed_counter)}")
    print(f"Exact char matches   : {dict(exact_char_match_counter)}")


if __name__ == "__main__":
    main()
