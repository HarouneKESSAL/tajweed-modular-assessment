from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTENT_REFERENCE_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "quran_content_reference_full.jsonl"
)

TAJWEED_REFERENCE_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "quran_tajweed_reference_full.jsonl"
)


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


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))

        previous = current

    return previous[-1]


def char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(a=a, b=b).ratio()


def _extract_surah_ayah(row: dict[str, Any]) -> tuple[int | None, int | None]:
    surah = None
    ayah = None

    for key in ["surah", "surah_id", "surah_number", "sura"]:
        if key in row and row[key] not in (None, ""):
            try:
                surah = int(row[key])
                break
            except Exception:
                pass

    for key in ["ayah", "ayah_id", "ayah_number", "verse", "verse_number"]:
        if key in row and row[key] not in (None, ""):
            try:
                ayah = int(row[key])
                break
            except Exception:
                pass

    sample_id = str(row.get("id") or row.get("sample_id") or "")
    match = re.search(r"_(\d{3})_(\d{3})_", sample_id)

    if match:
        surah = surah if surah is not None else int(match.group(1))
        ayah = ayah if ayah is not None else int(match.group(2))

    return surah, ayah


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(row, dict):
            rows.append(row)

    return rows


def _extract_tajweed_text(row: dict[str, Any]) -> str:
    for key in ["text", "normalized_text", "ayah_text", "target", "transcript"]:
        value = row.get(key)

        if value:
            return normalize_text(str(value))

    return ""


def _extract_content_text(row: dict[str, Any]) -> str:
    for key in ["content_text", "normalized_text", "text", "ayah_text", "target", "transcript"]:
        value = row.get(key)

        if value:
            return normalize_text(str(value))

    return ""


@lru_cache(maxsize=1)
def load_ayah_index() -> dict[tuple[int, int], dict[str, Any]]:
    """
    Build a complete ayah index with two references:

    - text / tajweed_text:
      Mushaf/Tajweed reference used for display, rule positions, and Tajweed modules.

    - content_text:
      ASR-friendly Quran reference used for Whisper content verification.
    """
    if not CONTENT_REFERENCE_MANIFEST.exists():
        raise FileNotFoundError(f"Content reference not found: {CONTENT_REFERENCE_MANIFEST}")

    content_index: dict[tuple[int, int], dict[str, Any]] = {}

    for row in _load_jsonl(CONTENT_REFERENCE_MANIFEST):
        surah, ayah = _extract_surah_ayah(row)
        content_text = _extract_content_text(row)

        if surah is None or ayah is None or not content_text:
            continue

        key = (surah, ayah)

        if key not in content_index:
            content_index[key] = {
                "surah": surah,
                "ayah": ayah,
                "content_text": content_text,
                "content_text_compact": compact_text(content_text),
                "content_source_id": row.get("source_id") or row.get("id") or row.get("sample_id"),
            }

    tajweed_index: dict[tuple[int, int], dict[str, Any]] = {}

    for row in _load_jsonl(TAJWEED_REFERENCE_MANIFEST):
        surah, ayah = _extract_surah_ayah(row)
        tajweed_text = _extract_tajweed_text(row)

        if surah is None or ayah is None or not tajweed_text:
            continue

        key = (surah, ayah)

        if key not in tajweed_index:
            tajweed_index[key] = {
                "tajweed_text": tajweed_text,
                "tajweed_text_compact": compact_text(tajweed_text),
                "tajweed_source_id": row.get("id") or row.get("sample_id"),
            }

    index: dict[tuple[int, int], dict[str, Any]] = {}

    for key, content_row in content_index.items():
        tajweed_row = tajweed_index.get(key, {})
        tajweed_text = tajweed_row.get("tajweed_text") or content_row["content_text"]

        index[key] = {
            "surah": content_row["surah"],
            "ayah": content_row["ayah"],

            # Main display/Tajweed text.
            "text": tajweed_text,
            "text_compact": compact_text(tajweed_text),

            # Explicit separated forms.
            "tajweed_text": tajweed_text,
            "tajweed_text_compact": compact_text(tajweed_text),
            "content_text": content_row["content_text"],
            "content_text_compact": content_row["content_text_compact"],

            # Sources.
            "source_id": tajweed_row.get("tajweed_source_id") or content_row.get("content_source_id"),
            "tajweed_source_id": tajweed_row.get("tajweed_source_id"),
            "content_source_id": content_row.get("content_source_id"),
            "content_source_manifest": str(CONTENT_REFERENCE_MANIFEST),
            "tajweed_source_manifest": str(TAJWEED_REFERENCE_MANIFEST),
        }

    return index


def get_ayah_reference(surah: int, ayah: int) -> dict[str, Any]:
    index = load_ayah_index()
    key = (int(surah), int(ayah))

    if key not in index:
        raise KeyError(
            f"No ayah reference found for surah={surah}, ayah={ayah}. "
            f"Content reference: {CONTENT_REFERENCE_MANIFEST}"
        )

    return index[key]


def get_ayah_range_reference(surah: int, ayah_start: int, ayah_end: int) -> dict[str, Any]:
    surah = int(surah)
    ayah_start = int(ayah_start)
    ayah_end = int(ayah_end)

    if ayah_end < ayah_start:
        raise ValueError("ayah_end must be greater than or equal to ayah_start.")

    ayahs = [
        get_ayah_reference(surah, ayah)
        for ayah in range(ayah_start, ayah_end + 1)
    ]

    tajweed_text = normalize_text(" ".join(str(item["text"]) for item in ayahs))
    content_text = normalize_text(" ".join(str(item["content_text"]) for item in ayahs))

    return {
        "surah": surah,
        "ayah": ayah_start,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "text": tajweed_text,
        "text_compact": compact_text(tajweed_text),
        "tajweed_text": tajweed_text,
        "tajweed_text_compact": compact_text(tajweed_text),
        "content_text": content_text,
        "content_text_compact": compact_text(content_text),
        "source_id": f"{surah}:{ayah_start}-{ayah_end}",
        "ayahs": ayahs,
    }


def find_best_ayah_matches(pred_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    pred_norm = normalize_text(pred_text)
    pred_compact = compact_text(pred_norm)

    if not pred_compact:
        return []

    rows: list[dict[str, Any]] = []

    for reference in load_ayah_index().values():
        gold_compact = str(reference["content_text_compact"])

        edit_distance = levenshtein(gold_compact, pred_compact)

        # 🔴 OLD (biased by gold length):
        # cer = edit_distance / max(1, len(gold_compact))

        # ✅ NEW (symmetric — penalizes length mismatch too):
        max_len = max(len(gold_compact), len(pred_compact), 1)
        cer = edit_distance / max_len

        similarity = char_similarity(gold_compact, pred_compact)

        # Also add a length ratio penalty to deprioritize very short ayahs
        length_ratio = min(len(gold_compact), len(pred_compact)) / max_len

        rows.append({
            "surah": reference["surah"],
            "ayah": reference["ayah"],
            "text": reference["text"],
            "content_text": reference["content_text"],
            "text_compact": reference["text_compact"],
            "content_text_compact": gold_compact,
            "source_id": reference.get("source_id"),
            "cer": float(cer),
            "char_similarity": float(similarity),
            "length_ratio": float(length_ratio),
            "edit_distance": int(edit_distance),
            "gold_len": len(gold_compact),
            "pred_len": len(pred_compact),
        })

    # ✅ NEW sort: CER first, then similarity, then length_ratio (prefer same-length ayahs)
    rows.sort(key=lambda item: (
        round(item["cer"], 3),
        -round(item["char_similarity"], 3),
        -item["length_ratio"],   # prefer ayahs closer in length to what was recited
        item["surah"],
        item["ayah"],
    ))

    return rows[:top_k]

def classify_autodetect_match(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {
            "accepted": False,
            "needs_confirmation": False,
            "verdict": "no_match",
            "confidence": 0.0,
        }

    cer = float(best["cer"])
    similarity = float(best["char_similarity"])
    gold_len = int(best["gold_len"])

    if gold_len <= 5:
        accepted = cer == 0.0
        needs_confirmation = (not accepted) and cer <= 0.35
    else:
        accepted = cer <= 0.10 or similarity >= 0.92
        needs_confirmation = (not accepted) and (cer <= 0.25 or similarity >= 0.82)

    confidence = max(0.0, min(1.0, 1.0 - cer))

    if accepted:
        verdict = "autodetect_accepted"
    elif needs_confirmation:
        verdict = "autodetect_needs_confirmation"
    else:
        verdict = "autodetect_rejected_low_confidence"

    return {
        "accepted": bool(accepted),
        "needs_confirmation": bool(needs_confirmation),
        "verdict": verdict,
        "confidence": float(confidence),
    }