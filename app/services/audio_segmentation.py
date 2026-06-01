from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
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

def select_split_points_for_expected_count(
    candidate_points: list[float],
    *,
    duration_sec: float,
    expected_segment_count: int,
    segment_weights: Sequence[float] | None = None,
    min_segment_sec: float = 0.7,
) -> list[float]:
    """
    Select exactly expected_segment_count - 1 split points from silence candidates.

    In guided multi-ayah mode, we know how many ayahs are expected.
    Instead of splitting at every pause, this selects the silence boundaries that
    best match the expected ayah duration proportions.
    """
    if expected_segment_count <= 1:
        return []

    required_splits = expected_segment_count - 1

    candidates = sorted(
        point
        for point in candidate_points
        if point >= min_segment_sec and duration_sec - point >= min_segment_sec
    )

    if len(candidates) <= required_splits:
        return candidates

    if segment_weights and len(segment_weights) == expected_segment_count:
        weights = [max(float(w), 1.0) for w in segment_weights]
    else:
        weights = [1.0 for _ in range(expected_segment_count)]

    total_weight = sum(weights)
    cumulative = 0.0
    targets: list[float] = []

    for idx in range(required_splits):
        cumulative += weights[idx]
        targets.append(duration_sec * cumulative / total_weight)

    # Dynamic programming: choose ordered candidates closest to target boundaries.
    n = len(candidates)
    k = required_splits

    dp = [[float("inf")] * n for _ in range(k)]
    prev = [[-1] * n for _ in range(k)]

    for i, candidate in enumerate(candidates):
        dp[0][i] = (candidate - targets[0]) ** 2

    for split_idx in range(1, k):
        for i, candidate in enumerate(candidates):
            cost = (candidate - targets[split_idx]) ** 2

            for prev_i in range(i):
                previous_candidate = candidates[prev_i]

                if candidate - previous_candidate < min_segment_sec:
                    continue

                candidate_cost = dp[split_idx - 1][prev_i] + cost

                if candidate_cost < dp[split_idx][i]:
                    dp[split_idx][i] = candidate_cost
                    prev[split_idx][i] = prev_i

    best_last = min(range(n), key=lambda i: dp[k - 1][i])

    if dp[k - 1][best_last] == float("inf"):
        # Fallback: greedily choose closest available candidates.
        selected: list[float] = []

        for target in targets:
            remaining = [
                point
                for point in candidates
                if all(abs(point - chosen) >= min_segment_sec for chosen in selected)
            ]

            if not remaining:
                break

            selected.append(min(remaining, key=lambda point: abs(point - target)))

        return sorted(selected[:required_splits])

    selected = []
    current = best_last

    for split_idx in range(k - 1, -1, -1):
        selected.append(candidates[current])
        current = prev[split_idx][current]

    return sorted(selected)



def segment_audio_by_silence(
    audio_path: Path,
    output_dir: Path,
    *,
    request_id: str,
    silence_db: int = -30,
    min_silence_sec: float = 0.35,
    min_segment_sec: float = 0.7,
    drop_edge_segments_shorter_than_sec: float = 1.0,
    expected_segment_count: int | None = None,
    segment_weights: Sequence[float] | None = None,
    start_padding_sec: float = 0.15,
    end_padding_sec: float = 0.20,
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

    candidate_split_points = build_split_points(
    silences,
    duration_sec=duration_sec,
    min_segment_sec=min_segment_sec,
)

    if expected_segment_count is not None and expected_segment_count > 1:
        split_points = select_split_points_for_expected_count(
            candidate_split_points,
            duration_sec=duration_sec,
            expected_segment_count=expected_segment_count,
            segment_weights=segment_weights,
            min_segment_sec=min_segment_sec,
        )
    else:
        split_points = candidate_split_points

    boundaries = [0.0, *split_points, duration_sec]

    # Build segment spans first. Do not extract yet, because we may drop
    # leading/trailing silence-only spans before creating files.
    spans: list[tuple[float, float]] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg_duration = end - start

        if seg_duration < min_segment_sec:
            continue

        spans.append((start, end))

    # Drop short leading/trailing pieces only in free segmentation mode.
    # In expected-count guided mode, selected boundaries already produce the target count.
    if expected_segment_count is None:
        if spans and (spans[0][1] - spans[0][0]) < drop_edge_segments_shorter_than_sec:
            spans = spans[1:]

        if spans and (spans[-1][1] - spans[-1][0]) < drop_edge_segments_shorter_than_sec:
            spans = spans[:-1]

    segments: list[AudioSegmentInfo] = []

    for idx, (start, end) in enumerate(spans, start=1):
        padded_start = max(0.0, start - start_padding_sec)
        padded_end = min(duration_sec, end + end_padding_sec)

        seg_duration = padded_end - padded_start
        segment_path = output_dir / f"{request_id}_segment_{idx:03d}.wav"

        extract_wav_segment(
            input_path=audio_path,
            output_path=segment_path,
            start_sec=padded_start,
            end_sec=padded_end,
        )

        segments.append(
            AudioSegmentInfo(
                index=idx,
                start_sec=round(padded_start, 3),
                end_sec=round(padded_end, 3),
                duration_sec=round(seg_duration, 3),
                audio_path=str(segment_path),
            )
        )

    return segments