from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import re
from collections import Counter

import soundfile as sf

from data.build_torchaudio_alignment_corpus import normalize_uroman, run_uroman, sanitize_alignment_text


EXCLUDED_LABELS = {"not_related_quran", "multiple_aya", "not_match_aya", "in_complete"}


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_content_text(text: str) -> str:
    text = sanitize_alignment_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def audio_duration_sec(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def allowed_label(row: dict) -> bool:
    return row.get("final_label") not in EXCLUDED_LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/manifests/retasy_train.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_alignment_corpus.jsonl")
    parser.add_argument("--text-field", default="aya_text_norm")
    parser.add_argument("--uroman-cmd", default="")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no global cap.")
    parser.add_argument("--max-per-text", type=int, default=3, help="0 means no per-text cap.")
    parser.add_argument("--min-duration-sec", type=float, default=0.35)
    parser.add_argument("--max-duration-sec", type=float, default=12.0)
    parser.add_argument("--min-nonspace-chars", type=int, default=3)
    parser.add_argument(
        "--require-word-count-match",
        action="store_true",
        help="Keep only rows where Arabic and uroman word counts match. Not required for character-level content timing.",
    )
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.input)
    out_rows: list[dict] = []
    skipped = Counter()
    text_counts = Counter()
    surah_counts = Counter()
    romanized_cache: dict[str, tuple[str, str]] = {}

    for row in rows:
        if args.max_rows > 0 and len(out_rows) >= args.max_rows:
            break
        if not allowed_label(row):
            skipped["label_excluded"] += 1
            continue
        audio_path = row.get("audio_path")
        if not audio_path or not (PROJECT_ROOT / audio_path).exists():
            skipped["missing_audio"] += 1
            continue
        duration_sec = audio_duration_sec(PROJECT_ROOT / audio_path)
        if duration_sec < float(args.min_duration_sec):
            skipped["short_audio"] += 1
            continue
        if duration_sec > float(args.max_duration_sec):
            skipped["long_audio"] += 1
            continue
        normalized_text = normalize_content_text(str(row.get(args.text_field) or row.get("normalized_text") or row.get("text") or ""))
        if len(normalized_text.replace(" ", "")) < int(args.min_nonspace_chars):
            skipped["short_or_missing_text"] += 1
            continue
        compact_text = normalized_text.replace(" ", "")
        if args.max_per_text > 0 and text_counts[compact_text] >= int(args.max_per_text):
            skipped["max_per_text"] += 1
            continue
        if normalized_text in romanized_cache:
            romanized_raw, romanized_text = romanized_cache[normalized_text]
        else:
            try:
                romanized_raw = run_uroman(normalized_text, args.uroman_cmd)
                romanized_text = normalize_uroman(romanized_raw)
            except Exception:
                skipped["uroman_failed"] += 1
                continue
            romanized_cache[normalized_text] = (romanized_raw, romanized_text)
        romanized_words = romanized_text.split()
        if not romanized_words:
            skipped["empty_romanized"] += 1
            continue
        ar_words = normalized_text.split()
        if args.require_word_count_match and len(ar_words) != len(romanized_words):
            skipped["word_count_mismatch"] += 1
            continue

        text_counts[compact_text] += 1
        surah_counts[str(row.get("surah_name"))] += 1
        out_rows.append(
            {
                "id": row["id"],
                "audio_path": audio_path,
                "surah_name": row.get("surah_name"),
                "quranjson_verse_key": row.get("quranjson_verse_key"),
                "quranjson_verse_index": row.get("quranjson_verse_index"),
                "normalized_text": normalized_text,
                "normalized_words_ar": ar_words,
                "romanized_text_raw": romanized_raw,
                "romanized_text": romanized_text,
                "romanized_words": romanized_words,
                "audio_duration_sec": duration_sec,
                "source_final_label": row.get("final_label"),
                "source_text_field": args.text_field,
                "gold_duration_labels": row.get("gold_duration_labels", []),
                "projected_duration_labels": row.get("projected_duration_labels", []),
                "duration_rule_spans_normalized": row.get("duration_rule_spans_normalized", []),
            }
        )

    output_path = PROJECT_ROOT / args.output
    write_jsonl(out_rows, output_path)
    summary = {
        "input": str(PROJECT_ROOT / args.input),
        "output": str(output_path),
        "rows_written": len(out_rows),
        "unique_texts": len(text_counts),
        "surah_counts": dict(surah_counts),
        "skipped": dict(skipped),
    }
    print_json(summary)


if __name__ == "__main__":
    main()
