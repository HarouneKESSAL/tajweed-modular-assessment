from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.speed import (
    SpeedNormalizationConfig,
    estimate_speed_factor,
    maybe_normalize_waveform_speed,
    normalize_waveform_speed,
)


def synthetic_pulse_train(num_pulses: int, spacing: int, sample_rate: int = 16000) -> torch.Tensor:
    total = max(sample_rate, num_pulses * spacing + 400)
    waveform = torch.zeros(1, total)
    pulse = torch.hann_window(400)
    for i in range(num_pulses):
        start = i * spacing
        waveform[:, start : start + 400] += pulse
    return waveform


def test_manual_speed_normalization_increases_length():
    waveform = synthetic_pulse_train(num_pulses=8, spacing=1000)
    slowed = normalize_waveform_speed(waveform, sample_rate=16000, speed_factor=1.25)
    assert slowed.size(-1) > waveform.size(-1)


def test_auto_speed_normalization_slows_fast_waveform():
    fast_waveform = synthetic_pulse_train(num_pulses=22, spacing=500)
    slow_waveform = synthetic_pulse_train(num_pulses=8, spacing=1800)
    config = SpeedNormalizationConfig(enabled=True, target_speech_rate=10.0, max_speed_factor=1.4)

    fast_stats = estimate_speed_factor(fast_waveform, 16000, config)
    slow_stats = estimate_speed_factor(slow_waveform, 16000, config)

    assert fast_stats["speed_factor"] > 1.0
    assert slow_stats["speed_factor"] == 1.0


def test_disabled_speed_normalization_is_noop():
    waveform = synthetic_pulse_train(num_pulses=10, spacing=900)
    normalized, stats = maybe_normalize_waveform_speed(
        waveform,
        sample_rate=16000,
        config=SpeedNormalizationConfig(enabled=False),
    )
    assert normalized.shape == waveform.shape
    assert stats["applied"] is False
