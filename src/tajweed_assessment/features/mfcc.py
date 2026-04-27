from pathlib import Path
import torch
from tajweed_assessment.data.audio import load_audio
from tajweed_assessment.data.speed import SpeedNormalizationConfig

try:
    import torchaudio
except Exception:
    torchaudio = None

def extract_mfcc_features(
    path: str | Path,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    speed_config: SpeedNormalizationConfig | None = None,
) -> torch.Tensor:
    if torchaudio is None:
        raise ImportError("torchaudio is required for MFCC extraction")
    waveform, sr = load_audio(path, sample_rate=sample_rate, speed_config=speed_config)
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )
    mfcc = transform(waveform).squeeze(0).transpose(0, 1)
    delta = torchaudio.functional.compute_deltas(mfcc.transpose(0, 1)).transpose(0, 1)
    delta2 = torchaudio.functional.compute_deltas(delta.transpose(0, 1)).transpose(0, 1)
    return torch.cat([mfcc, delta, delta2], dim=-1)
