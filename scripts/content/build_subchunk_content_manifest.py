from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json

from content.train_chunked_content import load_jsonl, normalize_text_target


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def span_for_char_window(start_sec: float, end_sec: float, char_start: int, char_end: int, total_chars: int) -> tuple[float, float]:
    duration = max(0.0, end_sec - start_sec)
    total_chars = max(1, total_chars)
    window_start = start_sec + duration * (char_start / total_chars)
    window_end = start_sec + duration * (char_end / total_chars)
    return window_start, window_end


def build_word_subchunks(row: dict, *, min_chars: int, max_chars: int, stride: int) -> list[dict]:
    subchunks: list[dict] = []
    word_spans = row.get("word_spans") or []
    source_text = normalize_text_target(row.get("normalized_text", ""))
    for word_idx, span in enumerate(word_spans):
        text = normalize_text_target(span.get("text", ""))
        if len(text) < min_chars:
            continue
        word_start = float(span.get("start_sec", row.get("start_sec", 0.0)))
        word_end = float(span.get("end_sec", row.get("end_sec", word_start)))
        if word_end <= word_start:
            continue

        # Keep whole-word examples, then add short sliding windows for longer words.
        windows: list[tuple[int, int]] = [(0, len(text))]
        if len(text) > max_chars:
            step = max(1, stride)
            for start in range(0, len(text) - min_chars + 1, step):
                end = min(len(text), start + max_chars)
                if end - start >= min_chars:
                    windows.append((start, end))
                if end == len(text):
                    break

        seen: set[tuple[int, int]] = set()
        for window_idx, (start, end) in enumerate(windows):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            chunk_text = text[start:end]
            chunk_start, chunk_end = span_for_char_window(word_start, word_end, start, end, len(text))
            if chunk_end <= chunk_start:
                continue
            subchunk = {
                **row,
                "id": f"{row.get('id')}_sub_{word_idx:02d}_{window_idx:02d}",
                "source_chunk_id": row.get("id"),
                "source_normalized_text": source_text,
                "subchunk_type": "word" if start == 0 and end == len(text) else "char_window",
                "normalized_text": chunk_text,
                "start_sec": chunk_start,
                "end_sec": chunk_end,
                "word_count": 1,
                "char_count": len(chunk_text),
                "word_spans": [
                    {
                        "word_index": 0,
                        "text": chunk_text,
                        "char_start": 0,
                        "char_end": len(chunk_text),
                        "start_sec": chunk_start,
                        "end_sec": chunk_end,
                    }
                ],
            }
            subchunks.append(subchunk)
    return subchunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_chunks_with_subchunks.jsonl")
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--include-original", action="store_true", default=True)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.input)
    output_rows: list[dict] = []
    subchunk_count = 0
    for row in rows:
        row = {**row, "source_normalized_text": normalize_text_target(row.get("normalized_text", ""))}
        if args.include_original:
            output_rows.append(row)
        subchunks = build_word_subchunks(
            row,
            min_chars=max(1, args.min_chars),
            max_chars=max(args.min_chars, args.max_chars),
            stride=max(1, args.stride),
        )
        output_rows.extend(subchunks)
        subchunk_count += len(subchunks)

    output_path = PROJECT_ROOT / args.output
    write_jsonl(output_rows, output_path)
    summary = {
        "input": str(PROJECT_ROOT / args.input),
        "output": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "subchunk_rows": subchunk_count,
        "min_chars": int(args.min_chars),
        "max_chars": int(args.max_chars),
        "stride": int(args.stride),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
