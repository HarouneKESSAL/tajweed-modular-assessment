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


async def save_and_convert_upload(upload: UploadFile, output_wav_path: Path) -> Path:
    """
    Save an uploaded browser audio file and convert it to 16 kHz mono PCM WAV.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = Path(upload.filename or "audio.webm").suffix or ".input"
    raw_path = output_wav_path.with_suffix(suffix)

    with raw_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

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
        str(output_wav_path),
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed.\n"
            f"command: {' '.join(cmd)}\n\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    raw_path.unlink(missing_ok=True)
    return output_wav_path
