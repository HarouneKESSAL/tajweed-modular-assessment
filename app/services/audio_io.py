from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fastapi import UploadFile


def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    candidates = list(
        Path.home().joinpath(
            "AppData", "Local", "Microsoft", "WinGet", "Packages"
        ).glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")
    )

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "Could not find ffmpeg.exe. Install FFmpeg or add it to PATH."
    )


async def save_and_convert_upload(audio: UploadFile, wav_path: Path) -> Path:
    """
    Save an uploaded audio file to a temporary raw path, then convert it to
    16 kHz mono PCM WAV.

    Important: the raw input path and final WAV output path must be different,
    because FFmpeg cannot convert a file in-place.
    """
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    original_name = audio.filename or "upload.webm"
    suffix = Path(original_name).suffix.lower() or ".webm"

    raw_path = wav_path.with_name(f"{wav_path.stem}_raw{suffix}")

    contents = await audio.read()
    raw_path.write_bytes(contents)

    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(wav_path),
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    try:
        raw_path.unlink(missing_ok=True)
    except Exception:
        pass

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed.\n"
            f"command: {' '.join(cmd)}\n\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    return wav_path