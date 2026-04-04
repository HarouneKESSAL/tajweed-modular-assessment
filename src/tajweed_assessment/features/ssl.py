from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn

try:
    import torchaudio
except Exception:
    torchaudio = None

@dataclass
class DummySSLFeatureExtractor:
    output_dim: int = 64

    def from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = waveform.squeeze(0).unfold(0, size=160, step=160)
        proj = nn.Linear(frames.size(-1), self.output_dim, bias=False)
        with torch.no_grad():
            return proj(frames.float())

    def from_mfcc(self, mfcc: torch.Tensor) -> torch.Tensor:
        proj = nn.Linear(mfcc.size(-1), self.output_dim, bias=False)
        with torch.no_grad():
            return proj(mfcc)

class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, bundle_name: str = "WAV2VEC2_BASE") -> None:
        super().__init__()
        self.bundle_name = bundle_name
        self.fallback = DummySSLFeatureExtractor(output_dim=64)
        if torchaudio is not None and hasattr(torchaudio.pipelines, bundle_name):
            self.bundle = getattr(torchaudio.pipelines, bundle_name)
            self.model = self.bundle.get_model()
            self.output_dim = 64
        else:
            self.bundle = None
            self.model = None
            self.output_dim = self.fallback.output_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return self.fallback.from_waveform(waveform)
        with torch.no_grad():
            feats, _ = self.model.extract_features(waveform)
        return feats[-1].squeeze(0)

    def forward_path(self, path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
        from tajweed_assessment.data.audio import load_audio
        waveform, _ = load_audio(path, sample_rate=sample_rate)
        return self.forward(waveform)
