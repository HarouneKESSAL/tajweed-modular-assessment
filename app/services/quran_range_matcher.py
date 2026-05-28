from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from app.services.ayah_reference import compact_text, load_ayah_index, levenshtein, normalize_text


def char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def find_best_ayah_range(
    pred_text: str,
    *,
    max_ayahs: int = 20,
) -> dict[str, Any] | None:
    """
    Find the best contiguous Quran ayah range for a recognized free recitation.

    This compares the ASR transcript against concatenated content_text references.
    It is intended for free-recitation autodetection.
    """
    pred_norm = normalize_text(pred_text)
    pred_compact = compact_text(pred_norm)

    if not pred_compact:
        return None

    by_surah: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ref in load_ayah_index().values():
        by_surah[int(ref["surah"])].append(ref)

    for surah in by_surah:
        by_surah[surah].sort(key=lambda item: int(item["ayah"]))

    best: dict[str, Any] | None = None

    pred_len = len(pred_compact)

    for surah, ayahs in by_surah.items():
        n = len(ayahs)

        for start_idx in range(n):
            combined_parts: list[str] = []

            for end_idx in range(start_idx, min(n, start_idx + max_ayahs)):
                ref = ayahs[end_idx]
                combined_parts.append(str(ref.get("content_text_compact") or ref.get("text_compact") or ""))

                gold_compact = "".join(combined_parts)
                gold_len = len(gold_compact)

                if gold_len == 0:
                    continue

                # Fast length filter so we do not score impossible windows.
                ratio = gold_len / max(1, pred_len)
                if ratio < 0.55:
                    continue
                if ratio > 1.65:
                    break

                edit_distance = levenshtein(gold_compact, pred_compact)
                cer = edit_distance / max(1, gold_len)
                similarity = char_similarity(gold_compact, pred_compact)

                candidate = {
                    "surah": surah,
                    "ayah_start": int(ayahs[start_idx]["ayah"]),
                    "ayah_end": int(ayahs[end_idx]["ayah"]),
                    "ayah_count": end_idx - start_idx + 1,
                    "text": normalize_text(" ".join(str(a.get("text") or "") for a in ayahs[start_idx : end_idx + 1])),
                    "content_text": normalize_text(" ".join(str(a.get("content_text") or a.get("text") or "") for a in ayahs[start_idx : end_idx + 1])),
                    "text_compact": gold_compact,
                    "cer": float(cer),
                    "char_similarity": float(similarity),
                    "edit_distance": int(edit_distance),
                    "gold_len": gold_len,
                    "pred_len": pred_len,
                    "pred_text": pred_norm,
                }

                if best is None:
                    best = candidate
                    continue

                best_key = (best["cer"], -best["char_similarity"], best["ayah_count"])
                candidate_key = (candidate["cer"], -candidate["char_similarity"], candidate["ayah_count"])

                if candidate_key < best_key:
                    best = candidate

    return best


def classify_range_match(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {
            "accepted": False,
            "needs_confirmation": False,
            "verdict": "no_range_match",
            "confidence": 0.0,
        }

    cer = float(best["cer"])
    similarity = float(best["char_similarity"])

    accepted = cer <= 0.12 or similarity >= 0.90
    needs_confirmation = (not accepted) and (cer <= 0.25 or similarity >= 0.80)

    confidence = max(0.0, min(1.0, 1.0 - cer))

    if accepted:
        verdict = "autodetect_range_accepted"
    elif needs_confirmation:
        verdict = "autodetect_range_needs_confirmation"
    else:
        verdict = "autodetect_range_rejected_low_confidence"

    return {
        "accepted": bool(accepted),
        "needs_confirmation": bool(needs_confirmation),
        "verdict": verdict,
        "confidence": float(confidence),
    }