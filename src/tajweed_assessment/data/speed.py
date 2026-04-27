from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

try:
    import torchaudio
except Exception:
    torchaudio = None


@dataclass(frozen=True)
class SpeedNormalizationConfig:
    enabled: bool = False
    target_speech_rate: float = 12.0
    min_speed_factor: float = 1.0
    max_speed_factor: float = 1.35
    frame_length: int = 400
    hop_length: int = 160
    energy_threshold_ratio: float = 0.20
    peak_threshold_ratio: float = 0.10


def _frame_rms(waveform: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform shape [C, T], got {tuple(waveform.shape)}")
    if waveform.size(-1) < frame_length:
        pad = frame_length - waveform.size(-1)
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    frames = waveform.unfold(-1, frame_length, hop_length)
    return frames.pow(2).mean(dim=-1).sqrt().mean(dim=0)


def estimate_speed_factor(
    waveform: torch.Tensor,
    sample_rate: int,
    config: SpeedNormalizationConfig,
) -> Dict[str, float | int]:
    rms = _frame_rms(waveform, config.frame_length, config.hop_length)
    if rms.numel() < 3:
        return {
            "speed_factor": 1.0,
            "estimated_speech_rate": 0.0,
            "num_voiced_frames": int(rms.numel()),
        }

    peak_rms = float(rms.max().item())
    if peak_rms <= 0.0:
        return {
            "speed_factor": 1.0,
            "estimated_speech_rate": 0.0,
            "num_voiced_frames": 0,
        }

    energy_threshold = peak_rms * config.energy_threshold_ratio
    voiced = rms >= energy_threshold
    num_voiced_frames = int(voiced.sum().item())
    if num_voiced_frames == 0:
        return {
            "speed_factor": 1.0,
            "estimated_speech_rate": 0.0,
            "num_voiced_frames": 0,
        }

    delta = torch.diff(rms, prepend=rms[:1])
    peak_threshold = peak_rms * config.peak_threshold_ratio
    onsets = (
        (delta[1:-1] > delta[:-2])
        & (delta[1:-1] >= delta[2:])
        & (delta[1:-1] > peak_threshold)
        & voiced[1:-1]
    )
    onset_count = int(onsets.sum().item())

    voiced_indices = torch.nonzero(voiced, as_tuple=False).flatten()
    if voiced_indices.numel() >= 2:
        voiced_span_frames = int(voiced_indices[-1].item() - voiced_indices[0].item() + 1)
    else:
        voiced_span_frames = max(1, num_voiced_frames)

    total_frames = max(1, rms.numel())
    analysis_frames = min(total_frames, max(1, voiced_span_frames))
    analysis_seconds = max(1e-6, analysis_frames * config.hop_length / sample_rate)
    estimated_speech_rate = onset_count / analysis_seconds

    if estimated_speech_rate <= config.target_speech_rate:
        speed_factor = 1.0
    else:
        speed_factor = estimated_speech_rate / max(config.target_speech_rate, 1e-6)

    speed_factor = min(max(speed_factor, config.min_speed_factor), config.max_speed_factor)

    return {
        "speed_factor": float(speed_factor),
        "estimated_speech_rate": float(estimated_speech_rate),
        "num_voiced_frames": num_voiced_frames,
        "num_onsets": onset_count,
        "analysis_frames": analysis_frames,
    }


def normalize_waveform_speed(
    waveform: torch.Tensor,
    sample_rate: int,
    speed_factor: float,
) -> torch.Tensor:
    if torchaudio is None:
        raise ImportError("torchaudio is required for waveform speed normalization")
    if speed_factor <= 1.0:
        return waveform

    target_rate = max(sample_rate + 1, int(round(sample_rate * speed_factor)))
    return torchaudio.functional.resample(waveform, sample_rate, target_rate)


def maybe_normalize_waveform_speed(
    waveform: torch.Tensor,
    sample_rate: int,
    config: SpeedNormalizationConfig | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if config is None or not config.enabled:
        return waveform, {
            "enabled": False,
            "applied": False,
            "speed_factor": 1.0,
            "estimated_speech_rate": None,
        }

    stats = estimate_speed_factor(waveform, sample_rate, config)
    speed_factor = float(stats["speed_factor"])
    normalized = normalize_waveform_speed(waveform, sample_rate, speed_factor)

    return normalized, {
        "enabled": True,
        "applied": speed_factor > 1.0,
        **stats,
    }
