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


def iter_word_ranges(text: str):
    start = None
    for idx, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                yield start, idx
                start = None
        else:
            if start is None:
                start = idx
    if start is not None:
        yield start, len(text)


def build_word_spans(row: dict):
    text = str(row.get("normalized_text") or "")
    char_spans = row.get("arabic_char_time_spans") or []
    if len(char_spans) != len(text):
        return []

    word_spans = []
    for word_index, (start_idx, end_idx) in enumerate(iter_word_ranges(text)):
        timed = [
            char_spans[i]
            for i in range(start_idx, end_idx)
            if char_spans[i].get("start_sec") is not None and char_spans[i].get("end_sec") is not None
        ]
        if not timed:
            continue
        word_spans.append(
            {
                "word_index": word_index,
                "text": text[start_idx:end_idx],
                "char_start": start_idx,
                "char_end": end_idx,
                "start_sec": float(timed[0]["start_sec"]),
                "end_sec": float(timed[-1]["end_sec"]),
            }
        )
    return word_spans


def chunk_word_spans(word_spans, *, max_words: int, max_chars: int):
    chunks = []
    cur = []
    cur_chars = 0
    for word in word_spans:
        word_chars = len(word["text"])
        next_chars = cur_chars + word_chars + (1 if cur else 0)
        would_overflow = bool(cur) and (len(cur) >= max_words or next_chars > max_chars)
        if would_overflow:
            chunks.append(cur)
            cur = []
            cur_chars = 0
        cur.append(word)
        cur_chars += word_chars + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append(cur)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/alignment/duration_time_projection_strict.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=12)
    parser.add_argument("--pad-sec", type=float, default=0.03)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output
    rows = load_jsonl(input_path)

    out_rows = []
    word_count_counter = Counter()
    duration_counter = Counter()
    skipped_no_words = 0

    for row in rows:
        word_spans = build_word_spans(row)
        if not word_spans:
            skipped_no_words += 1
            continue
        for chunk_index, chunk_words in enumerate(
            chunk_word_spans(word_spans, max_words=args.max_words, max_chars=args.max_chars)
        ):
            chunk_text = " ".join(word["text"] for word in chunk_words).strip()
            if not chunk_text:
                continue
            start_sec = max(0.0, float(chunk_words[0]["start_sec"]) - float(args.pad_sec))
            end_sec = float(chunk_words[-1]["end_sec"]) + float(args.pad_sec)
            if end_sec - start_sec < float(args.min_duration_sec):
                center = 0.5 * (start_sec + end_sec)
                half = 0.5 * float(args.min_duration_sec)
                start_sec = max(0.0, center - half)
                end_sec = center + half

            out_rows.append(
                {
                    "id": f"{row['id']}_chunk_{chunk_index:02d}",
                    "parent_id": row["id"],
                    "audio_path": row["audio_path"],
                    "surah_name": row.get("surah_name"),
                    "quranjson_verse_key": row.get("quranjson_verse_key"),
                    "reciter_id": row.get("reciter_id") or "Unknown",
                    "normalized_text": chunk_text,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_text.replace(" ", "")),
                    "word_spans": chunk_words,
                }
            )
            word_count_counter[len(chunk_words)] += 1
            duration_bucket = "short" if end_sec - start_sec < 0.8 else "medium" if end_sec - start_sec < 1.6 else "long"
            duration_counter[duration_bucket] += 1

    write_jsonl(output_path, out_rows)
    print(f"Input manifest : {input_path}")
    print(f"Output manifest: {output_path}")
    print(f"Source rows    : {len(rows)}")
    print(f"Chunks written : {len(out_rows)}")
    print(f"Skipped rows   : {skipped_no_words}")
    print(f"Chunks by words: {dict(word_count_counter)}")
    print(f"Chunks by dur  : {dict(duration_counter)}")


if __name__ == "__main__":
    main()

