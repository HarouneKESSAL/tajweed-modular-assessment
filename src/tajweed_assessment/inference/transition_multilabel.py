from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
import yaml

from tajweed_assessment.models.transition.multilabel_transition_module import (
    TransitionMultiLabelModule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_audio_mono(path: Path, target_sample_rate: int = 16000) -> torch.Tensor:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.tensor(audio, dtype=torch.float32).transpose(0, 1)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            int(sample_rate),
            target_sample_rate,
        )(waveform)

    return waveform


def extract_mfcc(path: Path, sample_rate: int = 16000, n_mfcc: int = 39) -> torch.Tensor:
    waveform = load_audio_mono(path, target_sample_rate=sample_rate)

    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
            "center": True,
        },
    )

    mfcc = transform(waveform).squeeze(0).transpose(0, 1).contiguous()
    return mfcc


def labels_to_combo(labels: list[str]) -> str:
    if not labels:
        return "none"
    if labels == ["ikhfa"]:
        return "ikhfa"
    if labels == ["idgham"]:
        return "idgham"
    if set(labels) == {"ikhfa", "idgham"}:
        return "ikhfa+idgham"
    return "+".join(labels)


@dataclass
class TransitionMultiLabelPrediction:
    predicted_rules: list[str]
    predicted_combo: str
    probabilities: dict[str, float]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_rules": self.predicted_rules,
            "predicted_combo": self.predicted_combo,
            "probabilities": self.probabilities,
            "thresholds": self.thresholds,
        }


class TransitionMultiLabelPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        thresholds: dict[str, float],
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = resolve_path(checkpoint_path)
        self.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        self.device = torch.device(device)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.label_names = list(ckpt.get("label_names", ["ikhfa", "idgham"]))

        self.model = TransitionMultiLabelModule(
            mfcc_dim=int(cfg.get("mfcc_dim", 39)),
            ssl_dim=int(cfg.get("ssl_dim", 64)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
            num_labels=len(self.label_names),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path: str | Path) -> TransitionMultiLabelPrediction:
        mfcc = extract_mfcc(resolve_path(audio_path))
        ssl = torch.zeros((mfcc.size(0), 64), dtype=torch.float32)

        mfcc = mfcc.unsqueeze(0).to(self.device)
        ssl = ssl.unsqueeze(0).to(self.device)
        lengths = torch.tensor([mfcc.size(1)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(mfcc, ssl, lengths)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu()

        probabilities = {
            label: float(prob)
            for label, prob in zip(self.label_names, probs.tolist())
        }

        predicted_rules = [
            label
            for label in self.label_names
            if probabilities[label] >= float(self.thresholds.get(label, 0.5))
        ]

        return TransitionMultiLabelPrediction(
            predicted_rules=predicted_rules,
            predicted_combo=labels_to_combo(predicted_rules),
            probabilities=probabilities,
            thresholds={label: float(self.thresholds.get(label, 0.5)) for label in self.label_names},
        )


def load_threshold_profile(
    config_path: str | Path,
    profile: str = "gold_safe",
    model_key: str = "transition_multilabel_retasy_hf_pilot_v1",
) -> tuple[str, dict[str, float]]:
    config_path = resolve_path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"threshold config must be a mapping: {config_path}")

    model_cfg = config.get(model_key)
    if not isinstance(model_cfg, dict):
        raise KeyError(f"missing model key in threshold config: {model_key}")

    checkpoint = str(model_cfg["checkpoint"])

    thresholds_by_profile = model_cfg.get("thresholds", {})
    thresholds = thresholds_by_profile.get(profile)
    if not isinstance(thresholds, dict):
        raise KeyError(f"missing threshold profile: {profile}")

    return checkpoint, {str(k): float(v) for k, v in thresholds.items()}


def load_transition_multilabel_predictor_from_config(
    config_path: str | Path,
    profile: str = "gold_safe",
    device: str = "cpu",
) -> TransitionMultiLabelPredictor:
    checkpoint, thresholds = load_threshold_profile(config_path, profile=profile)
    return TransitionMultiLabelPredictor(
        checkpoint_path=checkpoint,
        thresholds=thresholds,
        device=device,
    )
