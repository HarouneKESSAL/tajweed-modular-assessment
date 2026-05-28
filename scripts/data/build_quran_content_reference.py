from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TEXT_KEYS = [
    "content_text",
    "normalized_text",
    "text_imlaei_simple",
    "text_imlaei",
    "imlaei",
    "simple_text",
    "text_simple",
    "clean_text",
    "target",
    "transcript",
    "ayah_text",
    "text",
]

SURAH_KEYS = ["surah", "surah_id", "surah_number", "sura"]
AYAH_KEYS = ["ayah", "ayah_id", "ayah_number", "verse", "verse_number"]


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def extract_surah_ayah(row: dict[str, Any]) -> tuple[int | None, int | None]:
    surah = None
    ayah = None

    for key in SURAH_KEYS:
        if key in row and row[key] not in (None, ""):
            try:
                surah = int(row[key])
                break
            except Exception:
                pass

    for key in AYAH_KEYS:
        if key in row and row[key] not in (None, ""):
            try:
                ayah = int(row[key])
                break
            except Exception:
                pass

    verse_key = str(row.get("verse_key") or row.get("quranjson_verse_key") or "")
    match = re.search(r"(\d+)\s*[:_]\s*(\d+)", verse_key)
    if match:
        surah = surah if surah is not None else int(match.group(1))
        ayah = ayah if ayah is not None else int(match.group(2))

    sample_id = str(row.get("id") or row.get("sample_id") or "")
    match = re.search(r"_(\d{3})_(\d{3})_", sample_id)
    if match:
        surah = surah if surah is not None else int(match.group(1))
        ayah = ayah if ayah is not None else int(match.group(2))

    return surah, ayah


def extract_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if value:
            text = normalize_text(str(value))
            if text:
                return text
    return ""


def iter_rows(path: Path):
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        if isinstance(data.get("verses"), list):
            for item in data["verses"]:
                if isinstance(item, dict):
                    yield item
            return

        if isinstance(data.get("ayahs"), list):
            for item in data["ayahs"]:
                if isinstance(item, dict):
                    yield item
            return

        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item


def build_reference(source: Path, output: Path) -> None:
    index: dict[tuple[int, int], dict[str, Any]] = {}

    for row in iter_rows(source):
        surah, ayah = extract_surah_ayah(row)
        text = extract_text(row)

        if surah is None or ayah is None or not text:
            continue

        key = (surah, ayah)

        if key not in index:
            index[key] = {
                "surah": surah,
                "ayah": ayah,
                "content_text": text,
                "content_text_compact": compact_text(text),
                "source_id": row.get("id") or row.get("sample_id") or row.get("verse_key"),
            }

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for key in sorted(index):
            f.write(json.dumps(index[key], ensure_ascii=False) + "\n")

    print(f"Wrote {len(index)} ayah references to {output}")

    if len(index) < 6236:
        print(
            "WARNING: fewer than 6236 ayahs were written. "
            "The source may not be a complete Quran reference."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/quran_content_reference_full.jsonl"),
    )
    args = parser.parse_args()

    build_reference(args.source, args.output)


if __name__ == "__main__":
    main()