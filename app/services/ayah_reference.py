from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "content_v6a_short_hf_ayah_r1_hf_ayah_clean_no_juhaynee.jsonl"


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
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

    for s_key in ["surah", "surah_id", "surah_number", "sura"]:
        if s_key in row:
            try:
                surah = int(row[s_key])
                break
            except Exception:
                pass

    for a_key in ["ayah", "ayah_id", "ayah_number", "verse", "verse_number"]:
        if a_key in row:
            try:
                ayah = int(row[a_key])
                break
            except Exception:
                pass

    if surah is None or ayah is None:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        m = re.search(r"_(\d{3})_(\d{3})_", sample_id)
        if m:
            surah = surah if surah is not None else int(m.group(1))
            ayah = ayah if ayah is not None else int(m.group(2))

    return surah, ayah


def _row_text(row: dict[str, Any]) -> str:
    for key in ["normalized_text", "text", "target", "transcript", "ayah_text"]:
        value = row.get(key)
        if value:
            return normalize_text(str(value))
    return ""


@lru_cache(maxsize=1)
def load_ayah_index() -> dict[tuple[int, int], dict[str, Any]]:
    if not DEFAULT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {DEFAULT_MANIFEST}")

    index: dict[tuple[int, int], dict[str, Any]] = {}

    for line in DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        surah, ayah = _extract_surah_ayah(row)
        text = _row_text(row)

        if surah is None or ayah is None or not text:
            continue

        key = (surah, ayah)

        if key not in index:
            index[key] = {
                "surah": surah,
                "ayah": ayah,
                "text": text,
                "text_compact": compact_text(text),
                "source_id": row.get("id") or row.get("sample_id"),
            }

    return index


def get_ayah_reference(surah: int, ayah: int) -> dict[str, Any]:
    index = load_ayah_index()
    key = (int(surah), int(ayah))

    if key not in index:
        raise KeyError(
            f"No ayah reference found for surah={surah}, ayah={ayah} in {DEFAULT_MANIFEST}"
        )

    return index[key]


def find_best_ayah_matches(pred_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    pred_norm = normalize_text(pred_text)
    pred_compact = compact_text(pred_norm)

    if not pred_compact:
        return []

    rows: list[dict[str, Any]] = []

    for reference in load_ayah_index().values():
        gold_compact = str(reference["text_compact"])

        ed = levenshtein(gold_compact, pred_compact)
        cer = ed / max(1, len(gold_compact))
        sim = char_similarity(gold_compact, pred_compact)

        rows.append({
            "surah": reference["surah"],
            "ayah": reference["ayah"],
            "text": reference["text"],
            "text_compact": gold_compact,
            "source_id": reference.get("source_id"),
            "cer": float(cer),
            "char_similarity": float(sim),
            "edit_distance": int(ed),
            "gold_len": len(gold_compact),
            "pred_len": len(pred_compact),
        })

    rows.sort(key=lambda x: (x["cer"], -x["char_similarity"], x["surah"], x["ayah"]))
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
    sim = float(best["char_similarity"])
    gold_len = int(best["gold_len"])

    # Short ayahs need stricter rules because many short verses are easy to confuse.
    if gold_len <= 5:
        accepted = cer == 0.0
        needs_confirmation = (not accepted) and cer <= 0.35
    else:
        accepted = cer <= 0.05 or sim >= 0.95
        needs_confirmation = (not accepted) and (cer <= 0.15 or sim >= 0.88)

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
