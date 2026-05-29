from __future__ import annotations

from pathlib import Path
from typing import Any
from difflib import SequenceMatcher

from app.services.audio_segmentation import segment_audio_by_silence
from app.services.ayah_reference import load_ayah_index, normalize_text, levenshtein
from app.services.inference_runner import run_user_audio_inference
from app.services.multi_ayah_runner import run_multi_ayah_guided
from app.services.quran_range_matcher import classify_range_match, find_best_ayah_range
from app.services.whisper_gate import content_compare_compact, transcribe_audio
from app.services.segment_alignment import align_segments_to_contiguous_range, classify_alignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_ROOT = PROJECT_ROOT / "data" / "uploads" / "segments"


def char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def find_best_single_ayah_match(pred_text: str) -> dict[str, Any] | None:
    """
    Match one segment transcription to one Quran ayah.

    This uses content_text references and the same muqatta'at comparison logic
    used by the content gate.
    """
    pred_norm = normalize_text(pred_text)
    pred_compact = content_compare_compact(pred_norm)

    if not pred_compact:
        return None

    best: dict[str, Any] | None = None

    for reference in load_ayah_index().values():
        gold_compact = str(
            reference.get("content_text_compact")
            or reference.get("text_compact")
            or ""
        )

        if not gold_compact:
            continue

        edit_distance = levenshtein(gold_compact, pred_compact)
        cer = edit_distance / max(1, len(gold_compact))
        similarity = char_similarity(gold_compact, pred_compact)

        candidate = {
            "surah": int(reference["surah"]),
            "ayah": int(reference["ayah"]),
            "text": reference.get("text", ""),
            "content_text": reference.get("content_text") or reference.get("text", ""),
            "content_text_compact": gold_compact,
            "pred_text": pred_norm,
            "pred_compact": pred_compact,
            "cer": float(cer),
            "char_similarity": float(similarity),
            "edit_distance": int(edit_distance),
            "gold_len": len(gold_compact),
            "pred_len": len(pred_compact),
            "source_id": reference.get("source_id"),
        }

        if best is None:
            best = candidate
            continue

        best_key = (
            best["cer"],
            -best["char_similarity"],
            best["surah"],
            best["ayah"],
        )
        candidate_key = (
            candidate["cer"],
            -candidate["char_similarity"],
            candidate["surah"],
            candidate["ayah"],
        )

        if candidate_key < best_key:
            best = candidate

    return best


def classify_segment_match(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {
            "accepted": False,
            "needs_confirmation": False,
            "verdict": "no_segment_match",
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
        needs_confirmation = (not accepted) and (cer <= 0.25 or similarity >= 0.80)

    confidence = max(0.0, min(1.0, 1.0 - cer))

    if accepted:
        verdict = "segment_ayah_accepted"
    elif needs_confirmation:
        verdict = "segment_ayah_needs_confirmation"
    else:
        verdict = "segment_ayah_rejected_low_confidence"

    return {
        "accepted": bool(accepted),
        "needs_confirmation": bool(needs_confirmation),
        "verdict": verdict,
        "confidence": float(confidence),
    }


def detect_range_from_segments(
    *,
    audio_path: Path,
    request_id: str,
    max_segments: int = 20,
) -> dict[str, Any] | None:
    """
    Free-mode detection by constrained contiguous alignment.

    Steps:
    1. Split free recitation into pause-based segments.
    2. Transcribe each segment.
    3. Align all segments together to one contiguous Quran range.
    """
    output_dir = SEGMENTS_ROOT / request_id / "free_detect"

    segments = segment_audio_by_silence(
        audio_path=audio_path,
        output_dir=output_dir,
        request_id=f"{request_id}_free",
    )

    if len(segments) <= 1:
        return None

    if len(segments) > max_segments:
        return {
            "accepted": False,
            "needs_confirmation": True,
            "verdict": "too_many_segments",
            "confidence": 0.0,
            "segments": [
                {
                    "segment": {
                        "index": s.index,
                        "start_sec": s.start_sec,
                        "end_sec": s.end_sec,
                        "duration_sec": s.duration_sec,
                        "audio_path": s.audio_path,
                        "method": s.method,
                    }
                }
                for s in segments
            ],
            "message": f"Detected {len(segments)} segments, which is too many for free-mode automatic matching.",
        }

    segment_transcripts: list[str] = []
    raw_segment_payloads: list[dict[str, Any]] = []

    for segment in segments:
        pred_text = transcribe_audio(Path(segment.audio_path))
        segment_transcripts.append(pred_text)

        # Keep independent best match only for debugging/explanation.
        best_independent = find_best_single_ayah_match(pred_text)
        independent_decision = classify_segment_match(best_independent)

        raw_segment_payloads.append(
            {
                "segment": {
                    "index": segment.index,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "duration_sec": segment.duration_sec,
                    "audio_path": segment.audio_path,
                    "method": segment.method,
                },
                "recognized_text": pred_text,
                "best_ayah_independent": best_independent,
                "independent_decision": independent_decision,
            }
        )

    # Rough full-range hint from the combined segment transcript.
    # This is only used as a hint; the final decision comes from contiguous alignment.
    combined_transcript = normalize_text(" ".join(segment_transcripts))
    rough_range = find_best_ayah_range(
        combined_transcript,
        max_ayahs=max(20, len(segment_transcripts) + 4),
    )

    rough_decision = classify_range_match(rough_range)
    candidate_surah = int(rough_range["surah"]) if rough_range else None

    # First try the hinted surah if available.
    hinted_alignment = (
        align_segments_to_contiguous_range(
            segment_transcripts,
            candidate_surah=candidate_surah,
        )
        if candidate_surah is not None
        else None
    )

    hinted_decision = classify_alignment(hinted_alignment)

    # If the hinted surah is not good enough, search globally.
    if hinted_decision.get("accepted") or hinted_decision.get("needs_confirmation"):
        alignment = hinted_alignment
        alignment_decision = hinted_decision
        alignment_strategy = "hinted_surah_contiguous_alignment"
    else:
        alignment = align_segments_to_contiguous_range(segment_transcripts)
        alignment_decision = classify_alignment(alignment)
        alignment_strategy = "global_contiguous_alignment"

    if alignment is None:
        return {
            "accepted": False,
            "needs_confirmation": False,
            "verdict": "contiguous_alignment_failed",
            "confidence": 0.0,
            "segments": raw_segment_payloads,
            "rough_range": rough_range,
            "rough_decision": rough_decision,
        }

    pair_scores = alignment.get("pair_scores", [])

    aligned_segments: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_segment_payloads):
        pair = pair_scores[idx] if idx < len(pair_scores) else {}

        aligned_segments.append(
            {
                **item,
                "aligned_ayah": {
                    "surah": pair.get("surah"),
                    "ayah": pair.get("ayah"),
                    "text": pair.get("text", ""),
                    "content_text": pair.get("content_text", ""),
                    "cer": pair.get("cer", 1.0),
                    "char_similarity": pair.get("char_similarity", 0.0),
                    "edit_distance": pair.get("edit_distance", 0),
                },
            }
        )

    detected_ayah_sequence = [
        {
            "segment_index": int(item["segment"]["index"]),
            "surah": int(item["aligned_ayah"]["surah"]),
            "ayah": int(item["aligned_ayah"]["ayah"]),
            "recognized_text": item.get("recognized_text", ""),
            "expected_text": item["aligned_ayah"].get("content_text", ""),
            "cer": float(item["aligned_ayah"].get("cer", 1.0)),
            "char_similarity": float(item["aligned_ayah"].get("char_similarity", 0.0)),
        }
        for item in aligned_segments
        if item.get("aligned_ayah", {}).get("surah") is not None
    ]

    return {
        "accepted": bool(alignment_decision.get("accepted")),
        "needs_confirmation": bool(alignment_decision.get("needs_confirmation")),
        "verdict": alignment_decision.get("verdict"),
        "confidence": float(alignment_decision.get("confidence", 0.0)),
        "avg_cer": float(alignment.get("avg_cer", 1.0)),
        "avg_char_similarity": float(alignment.get("avg_char_similarity", 0.0)),
        "worst_cer": float(alignment.get("worst_cer", 1.0)),
        "surah": int(alignment["surah"]),
        "ayah_start": int(alignment["ayah_start"]),
        "ayah_end": int(alignment["ayah_end"]),
        "ayah_count": int(alignment["ayah_count"]),
        "text": normalize_text(alignment.get("text", "")),
        "content_text": normalize_text(alignment.get("content_text", "")),
        "detected_ayah_sequence": detected_ayah_sequence,
        "segments": aligned_segments,
        "alignment_strategy": alignment_strategy,
        "rough_range": rough_range,
        "rough_decision": rough_decision,
    }


def make_rejected_content_gate(
    *,
    pred_text: str,
    best: dict[str, Any] | None,
    verdict: str | None,
) -> dict[str, Any]:
    expected_text = str(best.get("content_text") or best.get("text") or "") if best else ""
    expected_compact = str(best.get("text_compact") or best.get("content_text_compact") or "") if best else ""
    pred_compact = content_compare_compact(pred_text)

    return {
        "accepted": False,
        "verdict": verdict or "autodetect_rejected",
        "mode": "autodetect_range",
        "exact": False,
        "gold": expected_text,
        "pred": pred_text,
        "gold_compact": expected_compact,
        "pred_compact": pred_compact,
        "char_accuracy": float(best.get("char_similarity", 0.0)) if best else 0.0,
        "cer": float(best.get("cer", 1.0)) if best else 1.0,
        "edit_distance": int(best.get("edit_distance", 0)) if best else 0,
        "gold_len": int(best.get("gold_len", len(expected_compact))) if best else 0,
        "pred_len": int(best.get("pred_len", len(pred_compact))),
    }


def run_detected_single_ayah(
    *,
    audio_path: Path,
    request_id: str,
    autodetect: dict[str, Any],
    surah: int,
    ayah: int,
) -> dict[str, Any]:
    result = run_user_audio_inference(
        audio_path=audio_path,
        surah=surah,
        ayah=ayah,
        request_id=request_id,
        mode="guided",
    )

    result["mode"] = "autodetect"
    result["autodetect"] = autodetect
    result["detected_surah"] = surah
    result["detected_ayah"] = ayah
    result["message"] = (
        f"Free recitation detected as Surah {surah}, Ayah {ayah}. "
        + str(result.get("message") or "")
    )

    return result


def run_detected_multi_ayah(
    *,
    audio_path: Path,
    request_id: str,
    autodetect: dict[str, Any],
    surah: int,
    ayah_start: int,
    ayah_end: int,
) -> dict[str, Any]:
    result = run_multi_ayah_guided(
        audio_path=audio_path,
        surah=surah,
        ayah_start=ayah_start,
        ayah_end=ayah_end,
        request_id=request_id,
    )

    result["mode"] = "autodetect_multi"
    result["autodetect"] = autodetect
    result["detected_surah"] = surah
    result["detected_ayah_start"] = ayah_start
    result["detected_ayah_end"] = ayah_end
    result["message"] = (
        f"Free recitation detected as Surah {surah}, "
        f"Ayah {ayah_start} to {ayah_end}."
    )

    return result


def run_free_recitation_assessment(
    *,
    audio_path: Path,
    request_id: str,
) -> dict[str, Any]:
    """
    Free mode.

    Preferred path:
      long audio -> pause segmentation -> ASR per segment -> ayah range detection

    Fallback path:
      if segmentation does not produce multiple segments, transcribe full audio once.
    """
    segment_detection = detect_range_from_segments(
        audio_path=Path(audio_path),
        request_id=request_id,
    )

    if segment_detection and segment_detection.get("accepted"):
        surah = int(segment_detection["surah"])
        ayah_start = int(segment_detection["ayah_start"])
        ayah_end = int(segment_detection["ayah_end"])

        autodetect = {
            "strategy": "segment_first",
            "recognized_text": " ".join(
                str(item.get("recognized_text") or "")
                for item in segment_detection.get("segments", [])
            ),
            "best_range": segment_detection,
            "decision": {
                "accepted": True,
                "needs_confirmation": False,
                "verdict": segment_detection.get("verdict"),
                "confidence": segment_detection.get("confidence", 0.0),
            },
        }

        if ayah_start == ayah_end:
            return run_detected_single_ayah(
                audio_path=audio_path,
                request_id=request_id,
                autodetect=autodetect,
                surah=surah,
                ayah=ayah_start,
            )

        return run_detected_multi_ayah(
            audio_path=audio_path,
            request_id=request_id,
            autodetect=autodetect,
            surah=surah,
            ayah_start=ayah_start,
            ayah_end=ayah_end,
        )

    # If segment-first produced a contiguous but low-confidence range, return it
    # for confirmation. If it produced a non-contiguous range, do not stop here:
    # fall back to full-audio range matching, because independent segment matching
    # can select unrelated ayahs with similar words.
    if (
        segment_detection
        and segment_detection.get("needs_confirmation")
        and segment_detection.get("verdict") != "segment_range_not_contiguous"
    ):
        autodetect = {
            "strategy": "segment_first",
            "recognized_text": " ".join(
                str(item.get("recognized_text") or "")
                for item in segment_detection.get("segments", [])
            ),
            "best_range": segment_detection,
            "decision": {
                "accepted": False,
                "needs_confirmation": True,
                "verdict": segment_detection.get("verdict"),
                "confidence": segment_detection.get("confidence", 0.0),
            },
        }

        return {
            "ok": True,
            "mode": "autodetect",
            "request_id": request_id,
            "audio_path": str(audio_path),
            "autodetect": autodetect,
            "content_gate": {
                "accepted": False,
                "verdict": segment_detection.get("verdict"),
                "mode": "autodetect_segment_range",
                "exact": False,
                "gold": segment_detection.get("content_text", ""),
                "pred": autodetect["recognized_text"],
                "gold_compact": content_compare_compact(segment_detection.get("content_text", "")),
                "pred_compact": content_compare_compact(autodetect["recognized_text"]),
                "char_accuracy": float(segment_detection.get("avg_char_similarity", 0.0)),
                "cer": float(segment_detection.get("avg_cer", 1.0)),
                "edit_distance": 0,
                "gold_len": len(content_compare_compact(segment_detection.get("content_text", ""))),
                "pred_len": len(content_compare_compact(autodetect["recognized_text"])),
            },
            "message": "Could not confidently confirm the segmented ayah range.",
        }

    # Fallback: old full-audio transcription.
    pred_text = transcribe_audio(audio_path)

    best = find_best_ayah_range(pred_text, max_ayahs=20)
    decision = classify_range_match(best)

    autodetect = {
        "strategy": "full_audio",
        "recognized_text": pred_text,
        "best_range": best,
        "decision": decision,
    }

    if not decision.get("accepted") or best is None:
        return {
            "ok": True,
            "mode": "autodetect",
            "request_id": request_id,
            "audio_path": str(audio_path),
            "autodetect": autodetect,
            "content_gate": make_rejected_content_gate(
                pred_text=pred_text,
                best=best,
                verdict=decision.get("verdict"),
            ),
            "message": "Could not confidently detect the recited ayah range.",
        }

    surah = int(best["surah"])
    ayah_start = int(best["ayah_start"])
    ayah_end = int(best["ayah_end"])

    if ayah_start == ayah_end:
        return run_detected_single_ayah(
            audio_path=audio_path,
            request_id=request_id,
            autodetect=autodetect,
            surah=surah,
            ayah=ayah_start,
        )

    return run_detected_multi_ayah(
        audio_path=audio_path,
        request_id=request_id,
        autodetect=autodetect,
        surah=surah,
        ayah_start=ayah_start,
        ayah_end=ayah_end,
    )