from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.services.audio_io import find_ffmpeg


@dataclass
class AudioSegmentInfo:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    audio_path: str
    method: str = "silence"


def get_wav_duration_sec(audio_path: Path) -> float:
    """Return duration of a PCM WAV file in seconds."""
    with wave.open(str(audio_path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate)


def detect_silences(
    audio_path: Path,
    silence_db: int = -30,
    min_silence_sec: float = 0.35,
) -> list[tuple[float, float]]:
    """
    Detect silence intervals using FFmpeg silencedetect.

    Returns:
        List of (silence_start_sec, silence_end_sec).
    """
    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={silence_db}dB:d={min_silence_sec}",
        "-f",
        "null",
        "-",
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg silencedetect failed.\n"
            f"command: {' '.join(cmd)}\n\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )

    stderr = completed.stderr

    starts: list[float] = []
    intervals: list[tuple[float, float]] = []

    for line in stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and starts:
            start = starts.pop(0)
            end = float(end_match.group(1))
            if end > start:
                intervals.append((start, end))

    return intervals


def build_split_points(
    silences: list[tuple[float, float]],
    duration_sec: float,
    min_segment_sec: float = 0.7,
) -> list[float]:
    """
    Convert silence intervals into split points.

    Each split point is placed in the middle of a silence interval.
    Very close split points are ignored.
    """
    split_points: list[float] = []

    last_point = 0.0

    for silence_start, silence_end in silences:
        midpoint = (silence_start + silence_end) / 2.0

        if midpoint - last_point < min_segment_sec:
            continue

        if duration_sec - midpoint < min_segment_sec:
            continue

        split_points.append(midpoint)
        last_point = midpoint

    return split_points


def extract_wav_segment(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Extract a segment as 16 kHz mono PCM WAV."""
    ffmpeg = find_ffmpeg()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "FFmpeg segment extraction failed.\n"
            f"command: {' '.join(cmd)}\n\n"
            f"stdout:\n{completed.stdout}\n\n"
            f"stderr:\n{completed.stderr}"
        )


def segment_audio_by_silence(
    audio_path: Path,
    output_dir: Path,
    *,
    request_id: str,
    silence_db: int = -30,
    min_silence_sec: float = 0.35,
    min_segment_sec: float = 0.7,
    drop_edge_segments_shorter_than_sec: float = 1.0,
) -> list[AudioSegmentInfo]:
    """
    Split a long recitation into ayah-level candidate segments using pauses.

    This first version is intended for guided multi-ayah support where
    the learner pauses briefly between ayahs.

    The function also drops short leading/trailing pieces, which are usually
    microphone preparation silence or trailing silence after the recitation.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)

    duration_sec = get_wav_duration_sec(audio_path)

    silences = detect_silences(
        audio_path,
        silence_db=silence_db,
        min_silence_sec=min_silence_sec,
    )

    split_points = build_split_points(
        silences,
        duration_sec=duration_sec,
        min_segment_sec=min_segment_sec,
    )

    boundaries = [0.0, *split_points, duration_sec]

    # Build segment spans first. Do not extract yet, because we may drop
    # leading/trailing silence-only spans before creating files.
    spans: list[tuple[float, float]] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg_duration = end - start

        if seg_duration < min_segment_sec:
            continue

        spans.append((start, end))

    # Drop short leading/trailing pieces, usually silence before/after recitation.
    if spans and (spans[0][1] - spans[0][0]) < drop_edge_segments_shorter_than_sec:
        spans = spans[1:]

    if spans and (spans[-1][1] - spans[-1][0]) < drop_edge_segments_shorter_than_sec:
        spans = spans[:-1]

    segments: list[AudioSegmentInfo] = []

    for idx, (start, end) in enumerate(spans, start=1):
        seg_duration = end - start
        segment_path = output_dir / f"{request_id}_segment_{idx:03d}.wav"

        extract_wav_segment(
            input_path=audio_path,
            output_path=segment_path,
            start_sec=start,
            end_sec=end,
        )

        segments.append(
            AudioSegmentInfo(
                index=idx,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                duration_sec=round(seg_duration, 3),
                audio_path=str(segment_path),
            )
        )

    return segments