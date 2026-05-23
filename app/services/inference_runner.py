from __future__ import annotations

from pathlib import Path

from app.services.ayah_reference import (
    classify_autodetect_match,
    find_best_ayah_matches,
    get_ayah_reference,
)
from app.services.whisper_gate import run_content_gate, transcribe_audio


def run_user_audio_inference(
    audio_path: Path,
    surah: int | None,
    ayah: int | None,
    request_id: str,
    mode: str = "guided",
) -> dict:
    mode = str(mode or "guided").lower().strip()

    if mode == "autodetect" or surah is None or ayah is None:
        pred_text = transcribe_audio(audio_path)
        matches = find_best_ayah_matches(pred_text, top_k=5)
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
                "content_gate": None,
                "tajweed": None,
                "message": "Could not identify the ayah.",
            }

        reference = {
            "surah": best["surah"],
            "ayah": best["ayah"],
            "text": best["text"],
            "text_compact": best["text_compact"],
            "source_id": best.get("source_id"),
        }

        # Reuse the already-detected reference, but still compute the same gate metrics.
        gate = run_content_gate(
            audio_path=audio_path,
            gold_text=reference["text"],
            mode="cer" if decision["accepted"] else "strict",
            pred_text=pred_text,
        )

        return {
            "ok": True,
            "request_id": request_id,
            "audio_path": str(audio_path),
            "mode": "autodetect",
            "surah": reference["surah"],
            "ayah": reference["ayah"],
            "reference": reference,
            "autodetect": {
                **decision,
                "pred": pred_text,
                "best_match": best,
                "matches": matches,
            },
            "content_gate": gate,
            "tajweed": None,
            "message": (
                "Ayah auto-detected and accepted. Tajweed modules not connected in API yet."
                if decision["accepted"]
                else "Possible ayah detected, but confidence is not high enough. Ask the user to confirm."
                if decision["needs_confirmation"]
                else "Could not confidently identify the ayah."
            ),
        }

    reference = get_ayah_reference(int(surah), int(ayah))

    gate = run_content_gate(
        audio_path=audio_path,
        gold_text=reference["text"],
        mode="strict",
    )

    return {
        "ok": True,
        "request_id": request_id,
        "audio_path": str(audio_path),
        "mode": "guided",
        "surah": int(surah),
        "ayah": int(ayah),
        "reference": reference,
        "content_gate": gate,
        "tajweed": None,
        "message": (
            "Content accepted. Tajweed modules not connected in API yet."
            if gate["accepted"]
            else "Content rejected. Tajweed scoring skipped."
        ),
    }
