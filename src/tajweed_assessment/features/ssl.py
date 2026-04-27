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

    def __post_init__(self) -> None:
        self._wave_proj: nn.Linear | None = None
        self._mfcc_proj: nn.Linear | None = None

    def _build_deterministic_proj(self, in_features: int) -> nn.Linear:
        proj = nn.Linear(in_features, self.output_dim, bias=False)
        with torch.no_grad():
            out_idx = torch.arange(self.output_dim, dtype=torch.float32).unsqueeze(1)
            in_idx = torch.arange(in_features, dtype=torch.float32).unsqueeze(0)
            weight = torch.sin((out_idx + 1.0) * (in_idx + 1.0) * 0.017)
            weight = weight / max(float(in_features), 1.0)
            proj.weight.copy_(weight)
        proj.requires_grad_(False)
        return proj

    def from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = waveform.squeeze(0).unfold(0, size=160, step=160)
        if self._wave_proj is None or self._wave_proj.in_features != frames.size(-1):
            self._wave_proj = self._build_deterministic_proj(frames.size(-1))
        with torch.no_grad():
            return self._wave_proj(frames.float())

    def from_mfcc(self, mfcc: torch.Tensor) -> torch.Tensor:
        if self._mfcc_proj is None or self._mfcc_proj.in_features != mfcc.size(-1):
            self._mfcc_proj = self._build_deterministic_proj(mfcc.size(-1))
        with torch.no_grad():
            return self._mfcc_proj(mfcc)

class Wav2VecFeatureExtractor(nn.Module):
    def __init__(self, bundle_name: str = "WAV2VEC2_BASE") -> None:
        super().__init__()
        self.bundle_name = bundle_name
        self.fallback = DummySSLFeatureExtractor(output_dim=64)
        if torchaudio is not None and hasattr(torchaudio.pipelines, bundle_name):
            try:
                self.bundle = getattr(torchaudio.pipelines, bundle_name)
                self.model = self.bundle.get_model()
                self.output_dim = self._infer_output_dim(self.model)
            except Exception:
                self.bundle = None
                self.model = None
                self.output_dim = self.fallback.output_dim
        else:
            self.bundle = None
            self.model = None
            self.output_dim = self.fallback.output_dim

    @staticmethod
    def _infer_output_dim(model: nn.Module) -> int:
        for attr in ("encoder_embed_dim", "embedding_dim", "embed_dim"):
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        aux = getattr(model, "aux", None)
        if isinstance(aux, nn.Module):
            for attr in ("in_features", "out_features"):
                value = getattr(aux, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
        try:
            dummy = torch.zeros(1, 16000)
            with torch.no_grad():
                feats, _ = model.extract_features(dummy)
            if feats:
                return int(feats[-1].size(-1))
        except Exception:
            pass
        return 64

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
