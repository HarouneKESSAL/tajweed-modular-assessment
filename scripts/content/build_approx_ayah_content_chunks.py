from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_words(words: list[str], *, max_words: int, max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for word in words:
        word_chars = len(word)
        next_chars = current_chars + word_chars + (1 if current else 0)
        if current and (len(current) >= max_words or next_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(word)
        current_chars += word_chars + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def word_weights(words: list[str]) -> list[float]:
    # Character-count timing is crude, but better than equal time per word for long Quran ayahs.
    return [max(1.0, float(len(word))) for word in words]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/hf_quran_md_ayahs_unique48_r2.jsonl")
    parser.add_argument("--output", default="data/manifests/hf_quran_md_ayahs_unique48_r2_approx_chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=16)
    parser.add_argument("--pad-sec", type=float, default=0.06)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.input)
    out_rows: list[dict] = []
    skipped = Counter()
    chunk_len_counts = Counter()

    for row in rows:
        text = str(row.get("normalized_text") or "").strip()
        words = [word for word in text.split() if word]
        if not words:
            skipped["missing_words"] += 1
            continue
        start_sec = float(row.get("start_sec", 0.0) or 0.0)
        end_sec = float(row.get("end_sec", row.get("audio_duration_sec", 0.0)) or 0.0)
        if end_sec <= start_sec:
            skipped["bad_duration"] += 1
            continue

        weights = word_weights(words)
        total_weight = sum(weights)
        duration = end_sec - start_sec
        word_spans = []
        cursor = start_sec
        for word_index, (word, weight) in enumerate(zip(words, weights)):
            word_duration = duration * (weight / total_weight)
            word_start = cursor
            word_end = end_sec if word_index == len(words) - 1 else cursor + word_duration
            cursor = word_end
            word_spans.append(
                {
                    "word_index": word_index,
                    "text": word,
                    "start_sec": word_start,
                    "end_sec": word_end,
                    "approx_timing": True,
                }
            )

        cursor_word = 0
        for chunk_index, chunk in enumerate(
            chunk_words(words, max_words=int(args.max_words), max_chars=int(args.max_chars))
        ):
            chunk_word_spans = word_spans[cursor_word : cursor_word + len(chunk)]
            cursor_word += len(chunk)
            chunk_text = " ".join(chunk)
            chunk_start = max(0.0, float(chunk_word_spans[0]["start_sec"]) - float(args.pad_sec))
            chunk_end = float(chunk_word_spans[-1]["end_sec"]) + float(args.pad_sec)
            if chunk_end - chunk_start < float(args.min_duration_sec):
                center = 0.5 * (chunk_start + chunk_end)
                half = 0.5 * float(args.min_duration_sec)
                chunk_start = max(0.0, center - half)
                chunk_end = center + half

            out = dict(row)
            out.update(
                {
                    "id": f"{row['id']}_approx_chunk_{chunk_index:02d}",
                    "parent_id": row["id"],
                    "source_normalized_text": text,
                    "normalized_text": chunk_text,
                    "start_sec": chunk_start,
                    "end_sec": chunk_end,
                    "word_count": len(chunk),
                    "char_count": len(chunk_text.replace(" ", "")),
                    "word_spans": chunk_word_spans,
                    "approx_chunk_timing": True,
                    "content_source": f"{row.get('content_source', 'unknown')}_approx_chunks",
                }
            )
            out_rows.append(out)
            chunk_len_counts[len(chunk)] += 1

    output_path = PROJECT_ROOT / args.output
    write_jsonl(output_path, out_rows)
    print(f"Input rows      : {len(rows)}")
    print(f"Output chunks   : {len(out_rows)}")
    print(f"Skipped         : {dict(skipped)}")
    print(f"Chunks by words : {dict(chunk_len_counts)}")
    print(f"Output          : {output_path}")


if __name__ == "__main__":
    main()
