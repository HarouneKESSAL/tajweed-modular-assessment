from pathlib import Path
from typing import Tuple
import torch

from tajweed_assessment.data.speed import (
    SpeedNormalizationConfig,
    maybe_normalize_waveform_speed,
)

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    import soundfile as sf
except Exception:
    sf = None

def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    speed_config: SpeedNormalizationConfig | None = None,
) -> Tuple[torch.Tensor, int]:
    waveform = None
    sr = None

    if sf is not None:
        data, sr = sf.read(str(path), always_2d=True)
        waveform = torch.tensor(data.T, dtype=torch.float32)
    elif torchaudio is not None:
        waveform, sr = torchaudio.load(str(path))
    else:
        raise ImportError("torchaudio or soundfile is required for real audio loading")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        if torchaudio is None:
            raise ImportError("torchaudio is required for audio resampling")
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    waveform, _ = maybe_normalize_waveform_speed(waveform, sr, speed_config)
    return waveform, sr
