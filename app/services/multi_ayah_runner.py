from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.audio_segmentation import AudioSegmentInfo, segment_audio_by_silence
from app.services.ayah_reference import get_ayah_reference
from app.services.whisper_gate import run_content_gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_ROOT = PROJECT_ROOT / "data" / "uploads" / "segments"


def try_run_tajweed_for_user_audio(
    audio_path: Path,
    reference: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    try:
        from app.services.tajweed_runner import run_tajweed_for_user_audio

        return run_tajweed_for_user_audio(
            audio_path=audio_path,
            reference=reference,
            request_id=request_id,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Tajweed runner failed: {type(exc).__name__}: {exc}",
            "reference": reference,
        }


def try_get_mushaf_preview(reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if not reference:
        return None

    try:
        from app.services.mushaf_preview import get_mushaf_preview

        return get_mushaf_preview(
            surah=int(reference["surah"]),
            ayah=int(reference["ayah"]),
        )
    except Exception as exc:
        return {
            "available": False,
            "surah": reference.get("surah"),
            "ayah": reference.get("ayah"),
            "text": reference.get("text", ""),
            "segments": [],
            "reason": f"Mushaf preview failed: {type(exc).__name__}: {exc}",
        }


def try_build_tajweed_ui_payload(
    reference: dict[str, Any] | None,
    tajweed: dict[str, Any] | None,
) -> dict[str, Any]:
    reference_text = ""

    if reference:
        reference_text = str(reference.get("text") or "")

    try:
        from app.services.tajweed_ui import build_tajweed_ui_payload

        return build_tajweed_ui_payload(reference_text, tajweed)
    except Exception as exc:
        return {
            "supported_rules": [
                {"name": "madd", "module": "duration", "status": "modeled"},
                {"name": "ghunnah", "module": "duration", "status": "modeled"},
                {"name": "ikhfa", "module": "transition", "status": "modeled"},
                {"name": "idgham", "module": "transition", "status": "modeled"},
                {"name": "qalqalah", "module": "burst", "status": "modeled"},
            ],
            "readable_feedback": [],
            "error": f"Tajweed UI builder failed: {type(exc).__name__}: {exc}",
        }


def try_build_content_feedback(
    *,
    gate: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any] | None:
    try:
        from app.services.content_feedback import build_content_feedback

        return build_content_feedback(
            gate=gate,
            reference=reference,
            mode=mode,
        )
    except Exception as exc:
        if gate and not gate.get("accepted"):
            return {
                "available": False,
                "accepted": False,
                "reason": f"Content feedback failed: {type(exc).__name__}: {exc}",
                "expected": gate.get("gold"),
                "recognized": gate.get("pred"),
                "items": [],
            }

        return None


def extract_tajweed_score(tajweed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tajweed or not tajweed.get("available"):
        return None

    result = tajweed.get("result") or {}

    weighted_score = result.get("weighted_score")
    if isinstance(weighted_score, dict):
        return weighted_score

    diagnosis = result.get("diagnosis") or {}
    weighted_score = diagnosis.get("weighted_score")
    if isinstance(weighted_score, dict):
        return weighted_score

    return None


def run_single_ayah_segment(
    *,
    segment: AudioSegmentInfo,
    surah: int,
    ayah: int,
    request_id: str,
) -> dict[str, Any]:
    reference = get_ayah_reference(surah, ayah)
    segment_path = Path(segment.audio_path)

    gate = (
        run_content_gate(
            audio_path=segment_path,
            gold_text=reference["text"],
            mode="strict",
        )
        or {
            "accepted": False,
            "available": False,
            "reason": "Content gate returned no result.",
        }
    )

    content_feedback = try_build_content_feedback(
        gate=gate,
        reference=reference,
        mode="guided",
    )

    content_accepted = bool(gate.get("accepted"))

    tajweed = (
        try_run_tajweed_for_user_audio(
            audio_path=segment_path,
            reference=reference,
            request_id=f"{request_id}_ayah_{ayah}",
        )
        if content_accepted
        else None
    )

    mushaf = try_get_mushaf_preview(reference)

    if content_accepted:
        tajweed_ui = try_build_tajweed_ui_payload(reference, tajweed)
    else:
        tajweed_ui = {
            "manifest_version": "1.0.0",
            "supported_rules": [],
            "readable_feedback": [],
            "feedback_policy": {
                "content_priority": True,
                "tajweed_skipped": True,
            },
            "message": "Content rejected. Tajweed feedback skipped.",
        }

    tajweed_score = extract_tajweed_score(tajweed)

    return {
        "surah": surah,
        "ayah": ayah,
        "segment": {
            "index": segment.index,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "duration_sec": segment.duration_sec,
            "audio_path": segment.audio_path,
            "method": segment.method,
        },
        "reference": reference,
        "mushaf": mushaf,
        "content_gate": gate,
        "content_feedback": content_feedback,
        "tajweed": tajweed,
        "tajweed_ui": tajweed_ui,
        "tajweed_score": tajweed_score,
        "message": (
            "Content verified and Tajweed inference completed."
            if gate.get("accepted") and tajweed and tajweed.get("available")
            else "Content verified, but Tajweed inference is not available."
            if gate.get("accepted")
            else "Content rejected. Tajweed scoring skipped."
        ),
    }


def run_multi_ayah_guided(
    *,
    audio_path: Path,
    surah: int,
    ayah_start: int,
    ayah_end: int,
    request_id: str,
) -> dict[str, Any]:
    surah = int(surah)
    ayah_start = int(ayah_start)
    ayah_end = int(ayah_end)

    if ayah_end < ayah_start:
        return {
            "ok": False,
            "mode": "guided_multi",
            "error": "ayah_end must be greater than or equal to ayah_start.",
        }

    expected_count = ayah_end - ayah_start + 1
    output_dir = SEGMENTS_ROOT / request_id

    reference_rows = [
    get_ayah_reference(surah, ayah_start + offset)
    for offset in range(expected_count)
    ]

    segment_weights = [
        max(1, len(str(ref.get("text_compact") or ref.get("text") or "")))
        for ref in reference_rows
    ]

    # First attempt: natural pause-based segmentation.
    # This is preferred when the learner clearly pauses between ayahs.
    natural_output_dir = output_dir / "natural"
    natural_segments = segment_audio_by_silence(
        audio_path=Path(audio_path),
        output_dir=natural_output_dir,
        request_id=f"{request_id}_natural",
    )

    if len(natural_segments) == expected_count:
        segments = natural_segments
        segmentation_strategy = "natural_pause"
    else:
        # Fallback: guided expected-count segmentation.
        # This forces the number of segments to match the selected ayah range.
        expected_output_dir = output_dir / "expected_count"
        segments = segment_audio_by_silence(
            audio_path=Path(audio_path),
            output_dir=expected_output_dir,
            request_id=f"{request_id}_expected",
            expected_segment_count=expected_count,
            segment_weights=segment_weights,
        )
    segmentation_strategy = "expected_count_fallback"

    segment_payload = [
        {
            "index": s.index,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "duration_sec": s.duration_sec,
            "audio_path": s.audio_path,
            "method": s.method,
        }
        for s in segments
    ]

    if len(segments) != expected_count:
        return {
            "ok": False,
            "mode": "guided_multi",
            "surah": surah,
            "ayah_start": ayah_start,
            "ayah_end": ayah_end,
            "expected_segments": expected_count,
            "detected_segments": len(segments),
            "segmentation_strategy": segmentation_strategy,
            "segments": segment_payload,
            "error": (
                f"Expected {expected_count} ayah segments but detected "
                f"{len(segments)}. Please pause clearly between ayahs and try again."
            ),
        }

    ayah_results: list[dict[str, Any]] = []

    for offset, segment in enumerate(segments):
        ayah = ayah_start + offset

        ayah_results.append(
            run_single_ayah_segment(
                segment=segment,
                surah=surah,
                ayah=ayah,
                request_id=request_id,
            )
        )

    content_accepted_count = sum(
    1
    for item in ayah_results
    if (item.get("content_gate") or {}).get("accepted")
    )

    tajweed_available_count = sum(
        1
        for item in ayah_results
        if (item.get("tajweed") or {}).get("available")
    )

    scores: list[float] = []
    total_errors = 0

    for item in ayah_results:
        score = item.get("tajweed_score") or {}

        if "score" in score:
            try:
                scores.append(float(score["score"]))
            except Exception:
                pass

        if "num_errors" in score:
            try:
                total_errors += int(score["num_errors"])
            except Exception:
                pass

    average_tajweed_score = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "ok": True,
        "mode": "guided_multi",
        "request_id": request_id,
        "segmentation_strategy": segmentation_strategy,
        "audio_path": str(audio_path),
        "surah": surah,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "expected_segments": expected_count,
        "detected_segments": len(segments),
        "segments": segment_payload,
        "ayah_results": ayah_results,
        "aggregate": {
            "content_accepted_count": content_accepted_count,
            "content_total": expected_count,
            "content_acceptance_rate": round(content_accepted_count / expected_count, 4),
            "tajweed_available_count": tajweed_available_count,
            "average_tajweed_score": average_tajweed_score,
            "total_errors": total_errors,
        },
        "message": (
            f"Multi-ayah guided assessment completed for Surah {surah}, "
            f"Ayah {ayah_start} to {ayah_end}."
        ),
    }