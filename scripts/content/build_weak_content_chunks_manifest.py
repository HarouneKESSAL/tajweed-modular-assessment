from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from content.train_chunked_content import load_jsonl, normalize_text_target

try:
    import soundfile as sf
except Exception:
    sf = None


EXCLUDED_LABELS = {"not_related_quran", "multiple_aya", "not_match_aya", "in_complete"}


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_words(row: dict) -> list[str]:
    text = str(row.get("aya_text_norm") or row.get("normalized_text") or row.get("text") or "").strip()
    return [word for word in text.split() if normalize_text_target(word)]


def verse_duration_sec(row: dict) -> float:
    if row.get("duration_ms") is not None:
        return max(0.25, float(row["duration_ms"]) / 1000.0)
    if row.get("duration_sec") is not None:
        return max(0.25, float(row["duration_sec"]))
    return 0.0


def audio_duration_sec(path: Path) -> float:
    if sf is None:
        return 0.0
    try:
        info = sf.info(str(path))
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def word_spans_from_weak_timing(words: list[str], duration_sec: float) -> list[dict]:
    weights = [max(1, len(normalize_text_target(word))) for word in words]
    total = max(1, sum(weights))
    spans = []
    cursor = 0.0
    for idx, (word, weight) in enumerate(zip(words, weights)):
        start = cursor
        cursor += duration_sec * (weight / total)
        spans.append(
            {
                "word_index": idx,
                "text": word,
                "char_start": None,
                "char_end": None,
                "start_sec": start,
                "end_sec": cursor,
                "timing_source": "weak_uniform_by_char_count",
            }
        )
    return spans


def chunk_word_spans(word_spans: list[dict], *, max_words: int, max_chars: int) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for word in word_spans:
        word_chars = len(normalize_text_target(word["text"]))
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


def allowed_row(row: dict, *, include_unlabeled: bool) -> bool:
    label = row.get("final_label")
    if label in EXCLUDED_LABELS:
        return False
    if label is None:
        return include_unlabeled
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_train.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_weak_chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=14)
    parser.add_argument("--pad-sec", type=float, default=0.02)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument(
        "--duration-source",
        choices=["audio", "metadata"],
        default="audio",
        help="Use measured audio duration for weak timing when available; metadata is kept for reproducibility/debugging.",
    )
    parser.add_argument("--include-unlabeled", action="store_true", default=True)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.input)
    output_rows: list[dict] = []
    skip_counts = Counter()
    label_counts = Counter()
    for row in rows:
        label_counts[str(row.get("final_label"))] += 1
        if not allowed_row(row, include_unlabeled=bool(args.include_unlabeled)):
            skip_counts["label_excluded"] += 1
            continue
        if not row.get("audio_path") or not (PROJECT_ROOT / row["audio_path"]).exists():
            skip_counts["missing_audio"] += 1
            continue
        audio_duration = audio_duration_sec(PROJECT_ROOT / row["audio_path"])
        if audio_duration < float(args.min_duration_sec):
            skip_counts["short_or_unreadable_audio"] += 1
            continue
        words = normalized_words(row)
        if not words:
            skip_counts["missing_text"] += 1
            continue
        metadata_duration = verse_duration_sec(row)
        duration_sec = audio_duration if args.duration_source == "audio" and audio_duration > 0 else metadata_duration
        if duration_sec <= 0.0:
            skip_counts["missing_duration"] += 1
            continue
        word_spans = word_spans_from_weak_timing(words, duration_sec)
        for chunk_idx, chunk_words in enumerate(
            chunk_word_spans(word_spans, max_words=max(1, args.max_words), max_chars=max(1, args.max_chars))
        ):
            chunk_text = " ".join(word["text"] for word in chunk_words)
            normalized_text = normalize_text_target(chunk_text)
            if not normalized_text:
                continue
            start_sec = max(0.0, float(chunk_words[0]["start_sec"]) - float(args.pad_sec))
            end_sec = min(duration_sec, float(chunk_words[-1]["end_sec"]) + float(args.pad_sec))
            if end_sec - start_sec < float(args.min_duration_sec):
                center = 0.5 * (start_sec + end_sec)
                half = 0.5 * float(args.min_duration_sec)
                start_sec = max(0.0, center - half)
                end_sec = min(duration_sec, center + half)
            if end_sec <= start_sec:
                continue
            output_rows.append(
                {
                    "id": f"{row.get('id', 'sample')}_weakchunk_{chunk_idx:02d}",
                    "parent_id": row.get("id"),
                    "audio_path": row["audio_path"],
                    "surah_name": row.get("surah_name"),
                    "quranjson_verse_key": row.get("quranjson_verse_key"),
                    "reciter_id": row.get("reciter_id") or "Unknown",
                    "source_normalized_text": normalize_text_target(" ".join(words)),
                    "normalized_text": chunk_text,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "word_count": len(chunk_words),
                    "char_count": len(normalized_text),
                    "timing_source": "weak_uniform_by_char_count",
                    "duration_source": args.duration_source,
                    "audio_duration_sec": audio_duration,
                    "metadata_duration_sec": metadata_duration,
                    "source_final_label": row.get("final_label"),
                    "word_spans": chunk_words,
                }
            )

    output_path = PROJECT_ROOT / args.output
    write_jsonl(output_rows, output_path)
    summary = {
        "input": str(PROJECT_ROOT / args.input),
        "output": str(output_path),
        "source_rows": len(rows),
        "chunks_written": len(output_rows),
        "unique_chunk_texts": len({normalize_text_target(row["normalized_text"]) for row in output_rows}),
        "unique_source_texts": len({row["source_normalized_text"] for row in output_rows}),
        "label_counts": dict(label_counts),
        "skip_counts": dict(skip_counts),
    }
    print_json(summary)


if __name__ == "__main__":
    main()
