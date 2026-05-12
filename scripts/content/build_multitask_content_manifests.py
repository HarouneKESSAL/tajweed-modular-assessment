from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def normalize_arabic_simple(text: str) -> str:
    diacritics = {
        "\u0610", "\u0611", "\u0612", "\u0613", "\u0614", "\u0615",
        "\u0616", "\u0617", "\u0618", "\u0619", "\u061A",
        "\u064B", "\u064C", "\u064D", "\u064E", "\u064F", "\u0650",
        "\u0651", "\u0652", "\u0653", "\u0654", "\u0655", "\u0656",
        "\u0657", "\u0658", "\u0659", "\u065A", "\u065B", "\u065C",
        "\u065D", "\u065E", "\u065F", "\u0670",
    }

    text = "".join(ch for ch in text if ch not in diacritics)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    text = " ".join(text.split())
    return text


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def extract_audio_path(row: dict[str, Any]) -> str:
    value = first_value(
        row,
        (
            "audio_path",
            "path",
            "wav_path",
            "mp3_path",
            "file",
            "audio_filepath",
            "audio_file",
        ),
    )

    if value is None:
        raise KeyError(f"Could not find audio path in row keys: {sorted(row.keys())}")

    return str(value)


def extract_text(row: dict[str, Any]) -> str:
    value = first_value(
        row,
        (
            "normalized_text",
            "expected_text",
            "text",
            "word",
            "arabic_text",
            "transcript",
            "target_text",
            "label",
            "source_text",
        ),
    )

    if value is None:
        raise KeyError(f"Could not find text in row keys: {sorted(row.keys())}")

    return normalize_arabic_simple(str(value).strip())


def normalize_row(
    row: dict[str, Any],
    *,
    source_task: str,
    source_manifest: str,
    index: int,
) -> dict[str, Any]:
    audio_path = extract_audio_path(row)
    text = extract_text(row)

    out = {
        "id": str(row.get("id") or row.get("sample_id") or f"{source_task}_{index:06d}"),
        "audio_path": audio_path,
        "normalized_text": text,
        "source_task": source_task,
        "source_manifest": source_manifest,
        "word_count": len(text.split()),
        "char_count": len(text),
    }

    for key in ("start_sec", "end_sec", "start_time", "end_time"):
        if key in row and row[key] is not None:
            out[key] = row[key]

    for key in (
        "surah_name",
        "surah_name_ar",
        "quranjson_surah_number",
        "quranjson_verse_key",
        "quranjson_verse_index",
        "reciter_id",
        "reciter_name",
        "parent_id",
        "content_source",
    ):
        if key in row and row[key] is not None:
            out[key] = row[key]

    return out


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not rows:
        return []
    if count <= len(rows):
        return rng.sample(rows, count)

    return [rng.choice(rows) for _ in range(count)]


def build_stage(
    *,
    word_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    word_count: int,
    chunk_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected.extend(sample_rows(word_rows, word_count, rng=rng))
    selected.extend(sample_rows(chunk_rows, chunk_count, rng=rng))
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-manifest", required=True)
    parser.add_argument("--chunk-manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--stage1-words", type=int, default=1000)
    parser.add_argument("--stage1-chunks", type=int, default=250)
    parser.add_argument("--stage2-words", type=int, default=750)
    parser.add_argument("--stage2-chunks", type=int, default=750)
    parser.add_argument("--stage3-words", type=int, default=250)
    parser.add_argument("--stage3-chunks", type=int, default=1000)

    args = parser.parse_args()

    rng = random.Random(args.seed)

    word_path = resolve_path(args.word_manifest)
    chunk_path = resolve_path(args.chunk_manifest)
    output_dir = resolve_path(args.output_dir)

    raw_word_rows = load_jsonl(word_path)
    raw_chunk_rows = load_jsonl(chunk_path)

    word_rows = [
        normalize_row(
            row,
            source_task="word",
            source_manifest=str(word_path),
            index=idx,
        )
        for idx, row in enumerate(raw_word_rows)
    ]

    chunk_rows = [
        normalize_row(
            row,
            source_task="chunk",
            source_manifest=str(chunk_path),
            index=idx,
        )
        for idx, row in enumerate(raw_chunk_rows)
    ]

    word_rows = [row for row in word_rows if row["normalized_text"]]
    chunk_rows = [row for row in chunk_rows if row["normalized_text"]]

    stage1 = build_stage(
        word_rows=word_rows,
        chunk_rows=chunk_rows,
        word_count=args.stage1_words,
        chunk_count=args.stage1_chunks,
        rng=rng,
    )

    stage2 = build_stage(
        word_rows=word_rows,
        chunk_rows=chunk_rows,
        word_count=args.stage2_words,
        chunk_count=args.stage2_chunks,
        rng=rng,
    )

    stage3 = build_stage(
        word_rows=word_rows,
        chunk_rows=chunk_rows,
        word_count=args.stage3_words,
        chunk_count=args.stage3_chunks,
        rng=rng,
    )

    vocab_rows = word_rows + chunk_rows

    write_jsonl(output_dir / "multitask_content_vocab_all.jsonl", vocab_rows)
    write_jsonl(output_dir / "multitask_content_stage1_word_heavy.jsonl", stage1)
    write_jsonl(output_dir / "multitask_content_stage2_balanced.jsonl", stage2)
    write_jsonl(output_dir / "multitask_content_stage3_chunk_heavy.jsonl", stage3)

    print("Built multitask manifests")
    print("-------------------------")
    print(f"word_rows_available  : {len(word_rows)}")
    print(f"chunk_rows_available : {len(chunk_rows)}")
    print(f"vocab_manifest       : {output_dir / 'multitask_content_vocab_all.jsonl'}")
    print(
        f"stage1 word-heavy    : total={len(stage1)} "
        f"words={sum(1 for r in stage1 if r['source_task'] == 'word')} "
        f"chunks={sum(1 for r in stage1 if r['source_task'] == 'chunk')}"
    )
    print(
        f"stage2 balanced      : total={len(stage2)} "
        f"words={sum(1 for r in stage2 if r['source_task'] == 'word')} "
        f"chunks={sum(1 for r in stage2 if r['source_task'] == 'chunk')}"
    )
    print(
        f"stage3 chunk-heavy   : total={len(stage3)} "
        f"words={sum(1 for r in stage3 if r['source_task'] == 'word')} "
        f"chunks={sum(1 for r in stage3 if r['source_task'] == 'chunk')}"
    )


if __name__ == "__main__":
    main()
