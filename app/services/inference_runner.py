from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.ayah_reference import (
    classify_autodetect_match,
    find_best_ayah_matches,
    get_ayah_reference,
)
from app.services.whisper_gate import run_content_gate, transcribe_audio
from app.services.audio_reference import get_reference_audio_url

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


def build_message(
    *,
    gate: dict[str, Any] | None,
    tajweed: dict[str, Any] | None,
    needs_confirmation: bool = False,
) -> str:
    if gate and not gate.get("accepted"):
        return "Content needs correction. Tajweed scoring was skipped."

    if gate and gate.get("accepted") and tajweed and tajweed.get("available"):
        return "Content verified and Tajweed inference completed."

    if gate and gate.get("accepted") and tajweed and not tajweed.get("available"):
        return "Content verified, but Tajweed inference is not available for this ayah yet."

    if gate and gate.get("accepted"):
        return "Content verified."

    if needs_confirmation:
        return "Possible ayah detected, but confidence is not high enough. Ask the user to confirm."

    return "Content rejected. Tajweed scoring skipped."


def run_user_audio_inference(
    audio_path: Path,
    surah: int | None,
    ayah: int | None,
    request_id: str,
    mode: str = "guided",
) -> dict[str, Any]:
    mode = str(mode or "guided").lower().strip()

    # Free recitation / auto-detect mode.
    if mode == "autodetect" or surah is None or ayah is None:
        pred_text = transcribe_audio(audio_path)

        matches = find_best_ayah_matches(pred_text, top_k=5)
        print("=== AUTODETECT DEBUG ===")
        print(f"pred_text: {pred_text}")
        for m in matches:
            print(f"  Surah {m['surah']} Ayah {m['ayah']} | CER={m['cer']:.3f} | sim={m['char_similarity']:.3f} | len_ratio={m.get('length_ratio', '?'):.3f}")
            print(f"    gold: {m['content_text'][:60]}")
        print(f"decision: {decision}")
        print("========================")
        best = matches[0] if matches else None
        decision = classify_autodetect_match(best)

        if best is None:
            return {
                "ok": True,
                "request_id": request_id,
                "audio_path": str(audio_path),
                "mode": "autodetect",
                "autodetect": {
                    **decision,
                    "pred": pred_text,
                    "matches": [],
                },
                "reference": None,
                "mushaf": None,
                "content_gate": None,
                "content_feedback": None,
                "tajweed": None,
                "tajweed_ui": try_build_tajweed_ui_payload(None, None),
                "message": "Could not identify the ayah.",
            }

        # ✅ ADD THIS — bail out early if confidence is too low
        if not decision["accepted"] and not decision["needs_confirmation"]:
            return {
                "ok": True,
                "request_id": request_id,
                "audio_path": str(audio_path),
                "mode": "autodetect",
                "autodetect": {
                    **decision,
                    "pred": pred_text,
                    "recognized_text": pred_text,
                    "best_match": best,
                    "matches": matches,
                    "best_range": None,
                },
                "reference": None,
                "reference_audio": None,
                "mushaf": None,
                "content_gate": None,       # ← no wrong comparison shown
                "content_feedback": None,   # ← no false rejection shown
                "tajweed": None,
                "tajweed_ui": try_build_tajweed_ui_payload(None, None),
                "message": "Could not confidently detect the recited ayah range.",
            }

        # Only reaches here if accepted=True or needs_confirmation=True
        reference = get_ayah_reference(int(best["surah"]), int(best["ayah"]))
        reference_audio = get_reference_audio_url(
            surah=int(reference["surah"]),
            ayah=int(reference["ayah"]),
        )
        gate = run_content_gate(
            audio_path=audio_path,
            gold_text=reference.get("content_text") or reference["text"],
            mode="cer" if decision.get("accepted") else "strict",
            pred_text=pred_text,
        )
        

        content_feedback = try_build_content_feedback(
            gate=gate,
            reference=reference,
            mode="autodetect",
        )

        tajweed = (
            try_run_tajweed_for_user_audio(audio_path, reference, request_id)
            if gate.get("accepted")
            else None
        )

        mushaf = try_get_mushaf_preview(reference)
        tajweed_ui = try_build_tajweed_ui_payload(reference, tajweed)

        return {
            "ok": True,
            "request_id": request_id,
            "audio_path": str(audio_path),
            "mode": "autodetect",
            "surah": reference["surah"],
            "ayah": reference["ayah"],
            "reference": reference,
            "reference_audio": reference_audio,
            "mushaf": mushaf,
            "autodetect": {
                **decision,
                "pred": pred_text,
                "best_match": best,
                "matches": matches,
            },
            "content_gate": gate,
            "content_feedback": content_feedback,
            "tajweed": tajweed,
            "tajweed_ui": tajweed_ui,
            "message": build_message(
                gate=gate,
                tajweed=tajweed,
                needs_confirmation=bool(decision.get("needs_confirmation")),
            ),
        }

    # Guided mode.
    reference = get_ayah_reference(int(surah), int(ayah))

    reference_audio = get_reference_audio_url(
            surah=int(surah or reference["surah"]),
            ayah=int(ayah or reference["ayah"]),
        )
    gate = run_content_gate(
        audio_path=audio_path,
        gold_text=reference.get("content_text") or reference["text"],
        mode="cer",
    )

    content_feedback = try_build_content_feedback(
        gate=gate,
        reference=reference,
        mode="guided",
    )

    tajweed = (
        try_run_tajweed_for_user_audio(audio_path, reference, request_id)
        if gate.get("accepted")
        else None
    )

    mushaf = try_get_mushaf_preview(reference)
    tajweed_ui = try_build_tajweed_ui_payload(reference, tajweed)

    return {
        "ok": True,
        "request_id": request_id,
        "audio_path": str(audio_path),
        "mode": "guided",
        "surah": int(surah),
        "ayah": int(ayah),
        "reference": reference,
        "reference_audio": reference_audio,
        "mushaf": mushaf,
        "content_gate": gate,
        "content_feedback": content_feedback,
        "tajweed": tajweed,
        "tajweed_ui": tajweed_ui,
        "message": build_message(
            gate=gate,
            tajweed=tajweed,
            needs_confirmation=False,
        ),
    }
