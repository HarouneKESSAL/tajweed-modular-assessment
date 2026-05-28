from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.services.audio_io import save_and_convert_upload
from app.services.inference_runner import run_user_audio_inference
from app.services.multi_ayah_runner import run_multi_ayah_guided


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

app = FastAPI(title="Tajweed Recitation Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "tajweed-recitation-api",
    }


@app.post("/api/assess-recitation")
async def assess_recitation(
    audio: UploadFile = File(...),
    mode: str = Form("guided"),
    surah: int | None = Form(None),
    ayah: int | None = Form(None),
    ayah_end: int | None = Form(None),
):
    request_id = f"user_{uuid4().hex}"
    wav_path = UPLOAD_DIR / f"{request_id}.wav"
    normalized_mode = str(mode).lower().strip()

    try:
        if normalized_mode == "guided" and (surah is None or ayah is None):
            return JSONResponse(
                {
                    "ok": False,
                    "error": (
                        "guided mode requires both surah and ayah. "
                        "Use mode=autodetect to detect automatically."
                    ),
                },
                status_code=400,
            )

        if (
            normalized_mode == "guided"
            and ayah_end is not None
            and ayah is not None
            and int(ayah_end) < int(ayah)
        ):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "ayah_end must be greater than or equal to ayah.",
                },
                status_code=400,
            )

        converted_path = await save_and_convert_upload(audio, wav_path)

        if (
            normalized_mode == "guided"
            and surah is not None
            and ayah is not None
            and ayah_end is not None
            and int(ayah_end) > int(ayah)
        ):
            result = run_multi_ayah_guided(
                audio_path=converted_path,
                surah=int(surah),
                ayah_start=int(ayah),
                ayah_end=int(ayah_end),
                request_id=request_id,
            )
        else:
            result = run_user_audio_inference(
                audio_path=converted_path,
                surah=surah,
                ayah=ayah,
                request_id=request_id,
                mode=normalized_mode,
            )

        return JSONResponse(result)

    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
            },
            status_code=500,
        )