from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from content.train_chunked_content import load_jsonl, normalize_text_target


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repeated_rows(rows: list[dict], repeat: int, source: str) -> list[dict]:
    out: list[dict] = []
    for repeat_idx in range(max(0, repeat)):
        for row in rows:
            item = dict(row)
            item["curriculum_source"] = source
            item["curriculum_repeat_index"] = repeat_idx
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/manifests/retasy_content_chunks_textsplit_trainonly.jsonl")
    parser.add_argument("--extra", default="data/manifests/retasy_content_chunks_alignment_correct200_no_textsplit_val.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_chunks_curriculum_correct200_r5.jsonl")
    parser.add_argument("--base-repeat", type=int, default=5)
    parser.add_argument("--extra-repeat", type=int, default=1)
    args = parser.parse_args()

    base_rows = load_jsonl(PROJECT_ROOT / args.base)
    extra_rows = load_jsonl(PROJECT_ROOT / args.extra)
    rows = repeated_rows(base_rows, int(args.base_repeat), "base") + repeated_rows(extra_rows, int(args.extra_repeat), "extra")
    output_path = PROJECT_ROOT / args.output
    write_jsonl(rows, output_path)

    source_counts = Counter(row.get("curriculum_source", "unknown") for row in rows)
    text_counts = Counter(normalize_text_target(row.get("normalized_text", "")) for row in rows)
    summary = {
        "base": str(PROJECT_ROOT / args.base),
        "extra": str(PROJECT_ROOT / args.extra),
        "output": str(output_path),
        "base_rows": len(base_rows),
        "extra_rows": len(extra_rows),
        "base_repeat": int(args.base_repeat),
        "extra_repeat": int(args.extra_repeat),
        "rows_written": len(rows),
        "source_counts": dict(source_counts),
        "unique_texts": len(text_counts),
        "top_texts": text_counts.most_common(15),
    }
    print_json(summary)


if __name__ == "__main__":
    main()
