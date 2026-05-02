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


DATASETS = {
    "quran-md-ayahs": {
        "hf_name": "Buraaq/quran-md-ayahs",
        "default_split": "train",
        "description": "Quran-MD ayah-level full-Quran recitations with surah/ayah metadata.",
    },
    "quran-ayah-corpus": {
        "hf_name": "rabah2026/Quran-Ayah-Corpus",
        "default_split": "train",
        "description": "Large ayah-level Quran ASR corpus with exact text and reciter metadata.",
    },
}


ARABIC_DIACRITICS_RE = re.compile(
    "["
    "\u0610-\u061a"
    "\u064b-\u065f"
    "\u0670"
    "\u06d6-\u06ed"
    "]"
)
NON_ARABIC_SPACE_RE = re.compile(r"[^\u0621-\u063a\u0641-\u064a\s]")


def normalize_arabic_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", " ")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.translate(
        str.maketrans(
            {
                "ٱ": "ا",
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ى": "ي",
                "ؤ": "و",
                "ئ": "ي",
                "ة": "ه",
            }
        )
    )
    text = NON_ARABIC_SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def audio_suffix(audio: dict, fallback: str = ".wav") -> str:
    path = str(audio.get("path") or "")
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix
    payload = audio.get("bytes") or b""
    if payload.startswith(b"RIFF"):
        return ".wav"
    if payload.startswith(b"ID3") or payload[:2] == b"\xff\xfb":
        return ".mp3"
    return fallback


def audio_duration_sec(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception:
        return 0.0
    if not info.samplerate:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Hugging Face Quran Content Import",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Split: `{summary['split']}`",
        f"- Rows written: `{summary['rows_written']}`",
        f"- Rows skipped: `{summary['rows_skipped']}`",
        f"- Unique normalized texts: `{summary['unique_texts']}`",
        f"- Unique reciters: `{summary['unique_reciters']}`",
        f"- Unique verse IDs: `{summary['unique_verse_ids']}`",
        f"- Total duration seconds: `{summary['total_duration_sec']:.2f}`",
        "",
        "## Top Reciters",
        "",
    ]
    for reciter, count in summary["top_reciters"]:
        lines.append(f"- `{reciter}`: `{count}`")
    lines.extend(["", "## Top Surahs", ""])
    for surah, count in summary["top_surahs"]:
        lines.append(f"- `{surah}`: `{count}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Audio is stored locally so downstream training does not depend on streaming from Hugging Face.",
            "- Arabic text is normalized by removing diacritics and standardizing common letter variants.",
            "- This import is intentionally capped for pilot experiments; do not download full Quran-scale data blindly.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def convert_buraaq_row(row: dict, *, row_index: int, audio_path: Path, duration_sec: float) -> dict:
    surah_id = int(row.get("surah_id") or 0)
    ayah_id = int(row.get("ayah_id") or 0)
    reciter_id = str(row.get("reciter_id") or "unknown")
    normalized = normalize_arabic_text(row.get("ayah_ar", ""))
    return {
        "id": f"quran_md_ayah_{surah_id:03d}_{ayah_id:03d}_{reciter_id}_{row_index:06d}",
        "source_dataset": "Buraaq/quran-md-ayahs",
        "source_index": row_index,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "surah_name": row.get("surah_name_tr") or row.get("surah_name_en") or row.get("surah_name_ar"),
        "surah_name_ar": row.get("surah_name_ar"),
        "quranjson_surah_number": surah_id,
        "quranjson_verse_key": f"verse_{ayah_id}",
        "quranjson_verse_index": ayah_id,
        "reciter_id": reciter_id,
        "reciter_name": row.get("reciter_name"),
        "normalized_text": normalized,
        "source_text": row.get("ayah_ar"),
        "transliteration": row.get("ayah_tr"),
        "start_sec": 0.0,
        "end_sec": duration_sec,
        "audio_duration_sec": duration_sec,
        "content_source": "hf_quran_md_ayahs",
    }


def convert_ayah_corpus_row(row: dict, *, row_index: int, audio_path: Path, duration_sec: float) -> dict:
    reciter = str(row.get("reciter") or "unknown")
    normalized = normalize_arabic_text(row.get("text", ""))
    return {
        "id": f"quran_ayah_corpus_{row_index:06d}",
        "source_dataset": "rabah2026/Quran-Ayah-Corpus",
        "source_index": row_index,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "surah_name": None,
        "quranjson_verse_key": None,
        "reciter_id": reciter,
        "reciter_name": reciter,
        "normalized_text": normalized,
        "source_text": row.get("text"),
        "start_sec": 0.0,
        "end_sec": duration_sec or float(row.get("duration") or 0.0),
        "audio_duration_sec": duration_sec or float(row.get("duration") or 0.0),
        "content_source": "hf_quran_ayah_corpus",
    }


def prefilter_key(row: dict, dataset_alias: str) -> tuple[str, str]:
    if dataset_alias == "quran-md-ayahs":
        text = normalize_arabic_text(row.get("ayah_ar", "")).replace(" ", "")
        reciter = str(row.get("reciter_id") or "unknown")
    elif dataset_alias == "quran-ayah-corpus":
        text = normalize_arabic_text(row.get("text", "")).replace(" ", "")
        reciter = str(row.get("reciter") or "unknown")
    else:
        text = ""
        reciter = "unknown"
    return text, reciter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="quran-md-ayahs")
    parser.add_argument("--split", default="")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--max-per-text", type=int, default=0, help="0 means unlimited rows per normalized text.")
    parser.add_argument("--max-per-reciter", type=int, default=0, help="0 means unlimited rows per reciter.")
    parser.add_argument("--output", default="data/manifests/hf_quran_content_pilot.jsonl")
    parser.add_argument("--audio-dir", default="data/raw/hf_quran_content_pilot")
    parser.add_argument("--summary-json", default="data/analysis/hf_quran_content_pilot_summary.json")
    parser.add_argument("--summary-md", default="data/analysis/hf_quran_content_pilot_summary.md")
    args = parser.parse_args()

    try:
        from datasets import Audio, load_dataset
    except Exception as exc:
        raise RuntimeError("The `datasets` package is required for Hugging Face Quran imports.") from exc

    dataset_cfg = DATASETS[args.dataset]
    hf_name = dataset_cfg["hf_name"]
    split = args.split or dataset_cfg["default_split"]
    audio_dir = PROJECT_ROOT / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_name, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    rows: list[dict] = []
    skipped = Counter()
    reciters = Counter()
    surahs = Counter()
    text_counts = Counter()
    total_duration = 0.0

    for row_index, row in enumerate(ds):
        if row_index < int(args.skip_rows):
            continue
        if len(rows) >= int(args.max_rows):
            break

        compact_text, reciter_id = prefilter_key(row, args.dataset)
        if not compact_text:
            skipped["empty_normalized_text"] += 1
            continue
        if int(args.max_per_text) > 0 and text_counts[compact_text] >= int(args.max_per_text):
            skipped["max_per_text"] += 1
            continue
        if int(args.max_per_reciter) > 0 and reciters[reciter_id] >= int(args.max_per_reciter):
            skipped["max_per_reciter"] += 1
            continue

        audio = row.get("audio") or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            skipped["missing_audio_bytes"] += 1
            continue

        suffix = audio_suffix(audio)
        item_id = f"{args.dataset}_{row_index:06d}"
        audio_path = audio_dir / f"{item_id}{suffix}"
        audio_path.write_bytes(audio_bytes)
        duration_sec = audio_duration_sec(audio_path)
        if duration_sec <= 0.0:
            skipped["unreadable_audio_duration"] += 1
            continue

        if args.dataset == "quran-md-ayahs":
            out_row = convert_buraaq_row(row, row_index=row_index, audio_path=audio_path, duration_sec=duration_sec)
        elif args.dataset == "quran-ayah-corpus":
            out_row = convert_ayah_corpus_row(row, row_index=row_index, audio_path=audio_path, duration_sec=duration_sec)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        if not out_row["normalized_text"]:
            skipped["empty_normalized_text"] += 1
            continue
        rows.append(out_row)
        text_counts[compact_text] += 1
        reciters[reciter_id] += 1
        surahs[str(out_row.get("surah_name") or "unknown")] += 1
        total_duration += float(out_row["audio_duration_sec"])

    output_path = PROJECT_ROOT / args.output
    write_jsonl(output_path, rows)

    verse_ids = {
        (row.get("quranjson_surah_number"), row.get("quranjson_verse_key"))
        for row in rows
        if row.get("quranjson_surah_number") or row.get("quranjson_verse_key")
    }
    summary = {
        "dataset": hf_name,
        "dataset_alias": args.dataset,
        "split": split,
        "max_rows": int(args.max_rows),
        "skip_rows": int(args.skip_rows),
        "max_per_text": int(args.max_per_text),
        "max_per_reciter": int(args.max_per_reciter),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "audio_dir": str(audio_dir.relative_to(PROJECT_ROOT)),
        "rows_written": len(rows),
        "rows_skipped": sum(skipped.values()),
        "skipped": dict(skipped),
        "unique_texts": len({row["normalized_text"].replace(" ", "") for row in rows}),
        "unique_reciters": len(reciters),
        "unique_verse_ids": len(verse_ids),
        "total_duration_sec": total_duration,
        "top_reciters": reciters.most_common(10),
        "top_surahs": surahs.most_common(10),
        "description": dataset_cfg["description"],
    }
    write_json(PROJECT_ROOT / args.summary_json, summary)
    write_markdown(PROJECT_ROOT / args.summary_md, summary)

    print(f"Dataset          : {hf_name}")
    print(f"Rows written     : {len(rows)}")
    print(f"Unique texts     : {summary['unique_texts']}")
    print(f"Unique reciters  : {summary['unique_reciters']}")
    print(f"Unique verse IDs : {summary['unique_verse_ids']}")
    print(f"Duration seconds : {total_duration:.2f}")
    print(f"Manifest         : {output_path}")
    print(f"Summary JSON     : {PROJECT_ROOT / args.summary_json}")


if __name__ == "__main__":
    main()
