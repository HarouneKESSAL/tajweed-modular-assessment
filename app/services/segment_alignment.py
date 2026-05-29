from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from app.services.ayah_reference import load_ayah_index, levenshtein, normalize_text
from app.services.whisper_gate import content_compare_compact


def char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def _group_references_by_surah() -> dict[int, list[dict[str, Any]]]:
    by_surah: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for ref in load_ayah_index().values():
        by_surah[int(ref["surah"])].append(ref)

    for surah in by_surah:
        by_surah[surah].sort(key=lambda item: int(item["ayah"]))

    return by_surah


def _score_pair(segment_text: str, reference: dict[str, Any]) -> dict[str, Any]:
    pred_compact = content_compare_compact(segment_text)

    gold_compact = str(
        reference.get("content_text_compact")
        or content_compare_compact(reference.get("content_text") or reference.get("text") or "")
    )

    edit_distance = levenshtein(gold_compact, pred_compact)
    cer = edit_distance / max(1, len(gold_compact))
    similarity = char_similarity(gold_compact, pred_compact)

    return {
        "surah": int(reference["surah"]),
        "ayah": int(reference["ayah"]),
        "text": reference.get("text", ""),
        "content_text": reference.get("content_text") or reference.get("text", ""),
        "gold_compact": gold_compact,
        "pred_compact": pred_compact,
        "cer": float(cer),
        "char_similarity": float(similarity),
        "edit_distance": int(edit_distance),
        "gold_len": len(gold_compact),
        "pred_len": len(pred_compact),
    }


def align_segments_to_contiguous_range(
    segment_texts: list[str],
    *,
    candidate_surah: int | None = None,
) -> dict[str, Any] | None:
    """
    Align segment transcripts to one contiguous Quran range.

    Instead of matching each segment independently against the whole Quran,
    this tests valid contiguous candidates only:

        segment 1 -> ayah N
        segment 2 -> ayah N+1
        segment 3 -> ayah N+2

    This prevents jumps like 2:2 -> 83:1 -> 4:162.
    """
    normalized_segments = [normalize_text(text) for text in segment_texts if normalize_text(text)]

    if not normalized_segments:
        return None

    segment_count = len(normalized_segments)
    by_surah = _group_references_by_surah()

    if candidate_surah is not None:
        surahs = [candidate_surah] if candidate_surah in by_surah else []
    else:
        surahs = sorted(by_surah)

    best: dict[str, Any] | None = None

    for surah in surahs:
        ayahs = by_surah[surah]

        if len(ayahs) < segment_count:
            continue

        for start_idx in range(0, len(ayahs) - segment_count + 1):
            candidate_refs = ayahs[start_idx : start_idx + segment_count]

            pair_scores = [
                _score_pair(segment_text, reference)
                for segment_text, reference in zip(normalized_segments, candidate_refs)
            ]

            avg_cer = sum(item["cer"] for item in pair_scores) / segment_count
            avg_similarity = sum(item["char_similarity"] for item in pair_scores) / segment_count
            worst_cer = max(item["cer"] for item in pair_scores)

            # Penalize candidates where one segment is very bad even if the average is okay.
            alignment_cost = avg_cer + (0.15 * worst_cer) + (0.10 * (1.0 - avg_similarity))

            candidate = {
                "surah": surah,
                "ayah_start": int(candidate_refs[0]["ayah"]),
                "ayah_end": int(candidate_refs[-1]["ayah"]),
                "ayah_count": segment_count,
                "text": normalize_text(" ".join(str(ref.get("text") or "") for ref in candidate_refs)),
                "content_text": normalize_text(
                    " ".join(str(ref.get("content_text") or ref.get("text") or "") for ref in candidate_refs)
                ),
                "avg_cer": float(avg_cer),
                "avg_char_similarity": float(avg_similarity),
                "worst_cer": float(worst_cer),
                "alignment_cost": float(alignment_cost),
                "pair_scores": pair_scores,
            }

            if best is None or candidate["alignment_cost"] < best["alignment_cost"]:
                best = candidate

    return best


def classify_alignment(alignment: dict[str, Any] | None) -> dict[str, Any]:
    if alignment is None:
        return {
            "accepted": False,
            "needs_confirmation": False,
            "verdict": "no_alignment",
            "confidence": 0.0,
        }

    avg_cer = float(alignment["avg_cer"])
    avg_similarity = float(alignment["avg_char_similarity"])
    worst_cer = float(alignment["worst_cer"])

    accepted = (
    avg_cer <= 0.22
    or avg_similarity >= 0.88
    or (avg_similarity >= 0.86 and worst_cer <= 0.70)
    )

    needs_confirmation = not accepted and (
        avg_cer <= 0.38
        or avg_similarity >= 0.75
    )

    confidence = max(0.0, min(1.0, 1.0 - avg_cer))

    if accepted:
        verdict = "contiguous_alignment_accepted"
    elif needs_confirmation:
        verdict = "contiguous_alignment_needs_confirmation"
    else:
        verdict = "contiguous_alignment_rejected"

    return {
        "accepted": bool(accepted),
        "needs_confirmation": bool(needs_confirmation),
        "verdict": verdict,
        "confidence": float(confidence),
    }