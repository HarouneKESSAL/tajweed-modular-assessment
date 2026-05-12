from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ARABIC_CHARS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءةىؤئ")
DIACRITICS = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        print(f"[warn] missing manifest: {path}")
        return rows

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


def normalize_arabic(text: str) -> str:
    text = "".join(ch for ch in str(text) if ch not in DIACRITICS)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def is_clean_arabic_text(text: str) -> bool:
    compact = text.replace(" ", "")
    return bool(compact) and all(ch in ARABIC_CHARS for ch in compact)


def text_value(row: dict[str, Any]) -> str:
    return normalize_arabic(
        row.get("normalized_text")
        or row.get("text")
        or row.get("source_text")
        or ""
    )


def convert_row(row: dict[str, Any], source_kind: str, index: int, max_chars: int) -> dict[str, Any] | None:
    text = text_value(row)

    if not text:
        return None
    if not is_clean_arabic_text(text):
        return None
    if len(text.replace(" ", "")) > max_chars:
        return None

    audio_path = row.get("audio_path", "")
    if not audio_path:
        return None

    out = {
        "id": str(row.get("id") or row.get("sample_id") or f"{source_kind}_{index:06d}"),
        "audio_path": audio_path,
        "normalized_text": text,
        "text": text,
        "source_kind": source_kind,
        "source_dataset": row.get("source_dataset", row.get("content_source", source_kind)),
        "surah_name": row.get("surah_name", ""),
        "surah_name_ar": row.get("surah_name_ar", ""),
        "quranjson_surah_number": row.get("quranjson_surah_number", ""),
        "quranjson_verse_key": row.get("quranjson_verse_key", ""),
        "word_index": row.get("word_index", ""),
    }

    for key in ("start_sec", "end_sec", "audio_duration_sec"):
        value = row.get(key)
        if value is not None and value != "":
            out[key] = value

    return out


def sample_rows(rng: random.Random, rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    if len(rows) <= n:
        out = list(rows)
        rng.shuffle(out)
        return out
    return rng.sample(rows, n)


def assign_text_group_splits(rows: list[dict[str, Any]], seed: int, val_fraction: float) -> None:
    rng = random.Random(seed)
    texts = sorted({row["normalized_text"] for row in rows})
    rng.shuffle(texts)

    val_count = int(round(len(texts) * val_fraction))
    val_texts = set(texts[:val_count])

    for row in rows:
        row["split"] = "val" if row["normalized_text"] in val_texts else "train"


def describe(name: str, rows: list[dict[str, Any]]) -> None:
    source_counts = Counter(row.get("source_kind", "") for row in rows)
    split_counts = Counter(row.get("split", "") for row in rows)
    char_lengths = [len(row["normalized_text"].replace(" ", "")) for row in rows]
    unique_texts = len(set(row["normalized_text"] for row in rows))

    print(name)
    print("-" * len(name))
    print(f"rows        : {len(rows)}")
    print(f"unique_texts: {unique_texts}")
    print(f"sources     : {dict(source_counts)}")
    print(f"splits      : {dict(split_counts)}")
    if char_lengths:
        print(f"chars       : min={min(char_lengths)} max={max(char_lengths)} avg={sum(char_lengths)/len(char_lengths):.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-manifest", default="data/manifests/hf_quran_md_words_pilot5000_r8.jsonl")
    parser.add_argument("--chunk-manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--hf-ayah-manifest", default="data/manifests/hf_quran_md_ayah_routing_weak_all_ayahs_r1.jsonl")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--prefix", default="content_v6_full_vocab_hf_ayah_r1")
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-ayah-chars", type=int, default=140)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_words = load_jsonl(resolve_path(args.word_manifest))
    raw_chunks = load_jsonl(resolve_path(args.chunk_manifest))
    raw_hf = load_jsonl(resolve_path(args.hf_ayah_manifest))

    words = [
        converted
        for i, row in enumerate(raw_words)
        if (converted := convert_row(row, "word", i, max_chars=40)) is not None
    ]

    chunks = [
        converted
        for i, row in enumerate(raw_chunks)
        if (converted := convert_row(row, "retasy_chunk", i, max_chars=80)) is not None
    ]

    hf_ayahs = [
        converted
        for i, row in enumerate(raw_hf)
        if (converted := convert_row(row, "hf_ayah", i, max_chars=args.max_ayah_chars)) is not None
    ]

    hf_short = [row for row in hf_ayahs if len(row["normalized_text"].replace(" ", "")) <= 55]
    hf_medium = [row for row in hf_ayahs if len(row["normalized_text"].replace(" ", "")) <= 95]

    # Full vocab manifest: all usable rows.
    vocab_all = list(words) + list(chunks) + list(hf_ayahs)
    assign_text_group_splits(vocab_all, args.seed, args.val_fraction)

    # Preserve split assignment consistently by text.
    split_by_text = {row["normalized_text"]: row["split"] for row in vocab_all}

    def with_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            row = dict(row)
            row["split"] = split_by_text.get(row["normalized_text"], "train")
            out.append(row)
        return out

    # Curriculum:
    # Stage 1 anchors CTC on short words/chunks.
    stage1 = (
        sample_rows(rng, words, 3500)
        + sample_rows(rng, chunks, 1000)
        + sample_rows(rng, hf_short, 500)
    )

    # Stage 2 introduces more ayah-level short/medium samples.
    stage2 = (
        sample_rows(rng, words, 1800)
        + sample_rows(rng, chunks, 900)
        + sample_rows(rng, hf_short, 1800)
        + sample_rows(rng, hf_medium, 1200)
    )

    # Stage 3 focuses on ayah-level coverage but keeps some short anchors.
    stage3 = (
        sample_rows(rng, words, 800)
        + sample_rows(rng, chunks, 800)
        + sample_rows(rng, hf_medium, 2600)
        + sample_rows(rng, hf_ayahs, 2600)
    )

    outputs = {
        f"{args.prefix}_vocab_all.jsonl": with_splits(vocab_all),
        f"{args.prefix}_stage1_anchor.jsonl": with_splits(stage1),
        f"{args.prefix}_stage2_ayah_intro.jsonl": with_splits(stage2),
        f"{args.prefix}_stage3_ayah_focus.jsonl": with_splits(stage3),
        f"{args.prefix}_hf_ayah_clean_all.jsonl": with_splits(hf_ayahs),
    }

    for filename, rows in outputs.items():
        write_jsonl(output_dir / filename, rows)
        describe(filename, rows)

    char_vocab = sorted({ch for row in vocab_all for ch in row["normalized_text"].replace(" ", "")})
    print("Character vocabulary")
    print("--------------------")
    print("size:", len(char_vocab))
    print("chars:", "".join(char_vocab))


if __name__ == "__main__":
    main()
