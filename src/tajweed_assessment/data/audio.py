from pathlib import Path
from typing import Tuple
import torch

try:
    import torchaudio
except Exception:
    torchaudio = None

def load_audio(path: str | Path, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise ImportError("torchaudio is required for real audio loading")
    waveform, sr = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    return waveform, sr
