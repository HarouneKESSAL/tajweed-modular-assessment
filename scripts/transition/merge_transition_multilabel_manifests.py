from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    seen = set()

    for input_path in args.inputs:
        rows = load_jsonl(Path(input_path))
        for row in rows:
            row_id = str(row.get("id") or row.get("sample_id") or "")
            if row_id and row_id in seen:
                continue
            if row_id:
                seen.add(row_id)
            merged.append(row)

    combo_counts = Counter(row.get("transition_combo", "unknown") for row in merged)
    source_counts = Counter(row.get("label_source", "unknown") for row in merged)

    write_jsonl(Path(args.output), merged)

    print("Merged transition manifests")
    print("---------------------------")
    print(f"rows         : {len(merged)}")
    print(f"combo_counts : {dict(combo_counts)}")
    print(f"source_counts: {dict(source_counts)}")
    print(f"output       : {args.output}")


if __name__ == "__main__":
    main()
