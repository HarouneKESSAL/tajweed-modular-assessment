from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Start with the manifest that we know run_inference.py works with.
# Later we can expand this list for transition / burst specific manifests.
TAJWEED_MANIFESTS = [
    PROJECT_ROOT / "data" / "manifests" / "quran_tajweed_reference_full.jsonl",
    PROJECT_ROOT / "data" / "manifests" / "retasy_duration_alignment_corpus_torchaudio_strict.jsonl",
]


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _extract_surah_ayah(row: dict[str, Any]) -> tuple[int | None, int | None]:
    surah = None
    ayah = None

    for key in ["surah", "surah_id", "surah_number", "sura"]:
        if key in row:
            try:
                surah = int(row[key])
                break
            except Exception:
                pass

    for key in ["ayah", "ayah_id", "ayah_number", "verse", "verse_number"]:
        if key in row:
            try:
                ayah = int(row[key])
                break
            except Exception:
                pass

    # Common project fallback: verse key like verse_3, plus surah_name is less useful.
    # Stronger fallback: sample IDs with _001_002_ if present.
    sample_id = str(row.get("id") or row.get("sample_id") or "")
    m = re.search(r"_(\d{3})_(\d{3})_", sample_id)
    if m:
        surah = surah if surah is not None else int(m.group(1))
        ayah = ayah if ayah is not None else int(m.group(2))

    quranjson_verse_key = str(row.get("quranjson_verse_key") or "")
    m2 = re.search(r"verse_(\d+)", quranjson_verse_key)
    if m2 and ayah is None:
        ayah = int(m2.group(1))

    return surah, ayah


def _row_text(row: dict[str, Any]) -> str:
    for key in ["normalized_text", "text", "target", "transcript", "ayah_text", "original_text"]:
        value = row.get(key)
        if value:
            return normalize_text(str(value))
    return ""


@lru_cache(maxsize=1)
def load_tajweed_index() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    for manifest in TAJWEED_MANIFESTS:
        if not manifest.exists():
            continue

        lines = manifest.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            surah, ayah = _extract_surah_ayah(row)
            text = _row_text(row)

            rows_out.append({
                "manifest": str(manifest),
                "sample_index": idx,
                "sample_id": row.get("id") or row.get("sample_id"),
                "surah": surah,
                "ayah": ayah,
                "text": text,
                "text_compact": compact_text(text),
                "quranjson_verse_key": row.get("quranjson_verse_key"),
                "raw_row": row,
            })

    return rows_out


def find_tajweed_row_for_reference(
    surah: int | None,
    ayah: int | None,
    text: str,
) -> dict[str, Any] | None:
    text_compact = compact_text(text)

    candidates = load_tajweed_index()

    # First try exact surah+ayah if available.
    if surah is not None and ayah is not None:
        exact = [
            r for r in candidates
            if r.get("surah") == int(surah) and r.get("ayah") == int(ayah)
        ]
        if exact:
            return exact[0]

    # Then try exact normalized text.
    exact_text = [r for r in candidates if r.get("text_compact") == text_compact]
    if exact_text:
        return exact_text[0]

    # Then try contains/contained for cases where a module manifest stores a phrase.
    loose = [
        r for r in candidates
        if text_compact
        and r.get("text_compact")
        and (text_compact in r["text_compact"] or r["text_compact"] in text_compact)
    ]
    if loose:
        return loose[0]

    return None


def run_tajweed_for_user_audio(
    audio_path: Path,
    reference: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    tajweed_row = find_tajweed_row_for_reference(
        surah=reference.get("surah"),
        ayah=reference.get("ayah"),
        text=reference.get("text", ""),
    )

    if tajweed_row is None:
        return {
            "available": False,
            "reason": "No matching Tajweed manifest row found for the detected ayah.",
            "reference": reference,
        }

    output_json = PROJECT_ROOT / "data" / "analysis" / "user_inference" / f"{request_id}_tajweed.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "system" / "run_inference.py"),
        "--manifest",
        tajweed_row["manifest"],
        "--sample-index",
        str(tajweed_row["sample_index"]),
        "--audio-override",
        str(audio_path),
        "--error-weights",
        str(PROJECT_ROOT / "configs" / "error_weights.yaml"),
        "--output-json",
        str(output_json),
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "Tajweed inference command failed.",
            "command": cmd,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "matched_row": {
                k: v for k, v in tajweed_row.items()
                if k != "raw_row"
            },
        }

    if not output_json.exists():
        return {
            "available": False,
            "reason": "Tajweed inference finished but did not create output JSON.",
            "command": cmd,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "matched_row": {
                k: v for k, v in tajweed_row.items()
                if k != "raw_row"
            },
        }

    result = json.loads(output_json.read_text(encoding="utf-8"))

    return {
        "available": True,
        "output_json": str(output_json),
        "matched_row": {
            k: v for k, v in tajweed_row.items()
            if k != "raw_row"
        },
        "result": result,
        "stdout_tail": completed.stdout[-4000:],
    }
