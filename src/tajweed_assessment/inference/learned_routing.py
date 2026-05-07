from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml

from tajweed_assessment.models.routing.learned_router import (
    LearnedRoutingModule,
    routing_label_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءةىؤئ")

IKHFA_TRIGGER_LETTERS = set("تثجدذزسشصضطظفقك")
IDGHAM_TRIGGER_LETTERS = set("يرملون")
QALQALAH_LETTERS = set("قطبجد")
MADD_LETTERS = set("اوي")
GHUNNAH_LETTERS = set("نم")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_text(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def audio_features(audio_path: str | Path) -> dict[str, float]:
    path = resolve_path(audio_path)

    try:
        audio, sample_rate = sf.read(str(path), always_2d=True)
        if audio.shape[1] > 1:
            values = audio.mean(axis=1)
        else:
            values = audio[:, 0]

        n = len(values)
        if n == 0:
            raise ValueError("empty audio")

        duration = float(n) / float(sample_rate)
        rms = float((values ** 2).mean() ** 0.5)
        peak = float(abs(values).max())
        zcr = float(((values[:-1] * values[1:]) < 0).mean()) if n > 1 else 0.0

        return {
            "audio_duration_sec": duration,
            "audio_log_duration": math.log1p(duration),
            "audio_rms": rms,
            "audio_peak": peak,
            "audio_zcr": zcr,
        }
    except Exception:
        return {
            "audio_duration_sec": 0.0,
            "audio_log_duration": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_zcr": 0.0,
        }


def text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    compact = text.replace(" ", "")
    total_chars = max(1, len(compact))

    feats: dict[str, float] = {
        "text_char_count": float(len(compact)),
        "text_word_count": float(len(text.split())),
        "text_log_char_count": math.log1p(len(compact)),
        "text_log_word_count": math.log1p(len(text.split())),
    }

    for letter in ARABIC_LETTERS:
        feats[f"char_freq_{letter}"] = compact.count(letter) / total_chars

    ikhfa_triggers = set("تثجدذزسشصضطظفقك")
    idgham_triggers = set("يرملون")
    qalqalah_letters = set("قطبجد")

    feats["contains_noon"] = 1.0 if "ن" in compact else 0.0
    feats["contains_meem"] = 1.0 if "م" in compact else 0.0
    feats["ikhfa_trigger_ratio"] = sum(compact.count(ch) for ch in ikhfa_triggers) / total_chars
    feats["idgham_trigger_ratio"] = sum(compact.count(ch) for ch in idgham_triggers) / total_chars
    feats["qalqalah_letter_ratio"] = sum(compact.count(ch) for ch in qalqalah_letters) / total_chars

    return feats




def compact_text(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def count_noon_followed_by(text: str, trigger_letters: set[str]) -> int:
    compact = compact_text(text)
    count = 0
    for idx, ch in enumerate(compact[:-1]):
        if ch == "ن" and compact[idx + 1] in trigger_letters:
            count += 1
    return count


def count_word_final_qalqalah(text: str) -> int:
    words = normalize_text(text).split()
    return sum(1 for word in words if word and word[-1] in QALQALAH_LETTERS)


def count_any_qalqalah(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in QALQALAH_LETTERS)


def count_madd_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in MADD_LETTERS)


def count_ghunnah_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in GHUNNAH_LETTERS)


def rule_aware_text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    compact = text.replace(" ", "")
    total_chars = max(1, len(compact))
    words = text.split()
    total_words = max(1, len(words))

    ikhfa_count = count_noon_followed_by(text, IKHFA_TRIGGER_LETTERS)
    idgham_count = count_noon_followed_by(text, IDGHAM_TRIGGER_LETTERS)
    transition_count = ikhfa_count + idgham_count

    qalqalah_any_count = count_any_qalqalah(text)
    qalqalah_final_count = count_word_final_qalqalah(text)

    madd_count = count_madd_letters(text)
    ghunnah_count = count_ghunnah_letters(text)

    return {
        "rule_ikhfa_candidate_count": float(ikhfa_count),
        "rule_idgham_candidate_count": float(idgham_count),
        "rule_transition_candidate_count": float(transition_count),
        "rule_has_ikhfa_candidate": 1.0 if ikhfa_count > 0 else 0.0,
        "rule_has_idgham_candidate": 1.0 if idgham_count > 0 else 0.0,
        "rule_has_transition_candidate": 1.0 if transition_count > 0 else 0.0,

        "rule_qalqalah_any_count": float(qalqalah_any_count),
        "rule_qalqalah_final_count": float(qalqalah_final_count),
        "rule_has_qalqalah_any": 1.0 if qalqalah_any_count > 0 else 0.0,
        "rule_has_qalqalah_final": 1.0 if qalqalah_final_count > 0 else 0.0,
        "rule_qalqalah_any_ratio": float(qalqalah_any_count) / total_chars,
        "rule_qalqalah_final_ratio": float(qalqalah_final_count) / total_words,

        "rule_madd_letter_count": float(madd_count),
        "rule_madd_letter_ratio": float(madd_count) / total_chars,
        "rule_has_madd_proxy": 1.0 if madd_count > 0 else 0.0,

        "rule_ghunnah_letter_count": float(ghunnah_count),
        "rule_ghunnah_letter_ratio": float(ghunnah_count) / total_chars,
        "rule_has_ghunnah_proxy": 1.0 if ghunnah_count > 0 else 0.0,

        "rule_transition_and_qalqalah": 1.0 if transition_count > 0 and qalqalah_any_count > 0 else 0.0,
        "rule_duration_and_transition": 1.0 if madd_count > 0 and transition_count > 0 else 0.0,
        "rule_duration_and_burst": 1.0 if madd_count > 0 and qalqalah_any_count > 0 else 0.0,
        "rule_all_three_proxy": 1.0 if madd_count > 0 and transition_count > 0 and qalqalah_any_count > 0 else 0.0,
    }


def make_feature_vector(
    *,
    audio_path: str | Path,
    text: str,
    feature_names: list[str],
) -> list[float]:
    fmap: dict[str, float] = {}
    fmap.update(audio_features(audio_path))
    fmap.update(text_features(text))
    fmap.update(rule_aware_text_features(text))
    return [float(fmap.get(name, 0.0)) for name in feature_names]


@dataclass
class LearnedRoutingPrediction:
    routing_plan: dict[str, bool]
    probabilities: dict[str, float]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_plan": self.routing_plan,
            "probabilities": self.probabilities,
            "thresholds": self.thresholds,
        }


class LearnedRoutingPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.checkpoint_path = resolve_path(checkpoint_path)
        self.device = torch.device(device)

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        cfg = ckpt.get("config", {})

        self.feature_names = list(ckpt["feature_names"])
        self.target_names = list(ckpt.get("target_names", routing_label_names()))

        checkpoint_thresholds = {
            name: float(cfg.get("thresholds", {}).get(name, 0.5))
            for name in self.target_names
        }
        if thresholds is not None:
            checkpoint_thresholds.update({str(k): float(v) for k, v in thresholds.items()})

        self.thresholds = checkpoint_thresholds

        self.model = LearnedRoutingModule(
            input_dim=int(cfg.get("input_dim", len(self.feature_names))),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_outputs=len(self.target_names),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, *, audio_path: str | Path, text: str) -> LearnedRoutingPrediction:
        vector = make_feature_vector(
            audio_path=audio_path,
            text=text,
            feature_names=self.feature_names,
        )
        features = torch.tensor([vector], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(features)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu()

        probabilities = {
            name: float(prob)
            for name, prob in zip(self.target_names, probs.tolist())
        }
        routing_plan = {
            name: probabilities[name] >= self.thresholds.get(name, 0.5)
            for name in self.target_names
        }

        return LearnedRoutingPrediction(
            routing_plan=routing_plan,
            probabilities=probabilities,
            thresholds=self.thresholds,
        )


def load_learned_routing_predictor(
    checkpoint_path: str | Path = "checkpoints/learned_router_v2.pt",
    device: str = "cpu",
    thresholds: dict[str, float] | None = None,
) -> LearnedRoutingPredictor:
    return LearnedRoutingPredictor(
        checkpoint_path=checkpoint_path,
        device=device,
        thresholds=thresholds,
    )


def load_threshold_profile(
    config_path: str | Path,
    profile: str = "balanced_safe",
    model_key: str = "learned_router_v2",
) -> tuple[str, dict[str, float]]:
    config_path = resolve_path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"threshold config must be a mapping: {config_path}")

    model_cfg = config.get(model_key)
    if not isinstance(model_cfg, dict):
        model_keys = [key for key, value in config.items() if isinstance(value, dict)]
        if len(model_keys) == 1:
            model_key = str(model_keys[0])
            model_cfg = config[model_key]
        else:
            raise KeyError(f"missing model key in threshold config: {model_key}")

    checkpoint = str(model_cfg["checkpoint"])
    thresholds_by_profile = model_cfg.get("thresholds", {})
    thresholds = thresholds_by_profile.get(profile)

    if not isinstance(thresholds, dict):
        raise KeyError(f"missing threshold profile: {profile}")

    return checkpoint, {str(k): float(v) for k, v in thresholds.items()}


def load_learned_routing_predictor_from_config(
    config_path: str | Path = "configs/learned_router_thresholds.yaml",
    profile: str = "balanced_safe",
    device: str = "cpu",
) -> LearnedRoutingPredictor:
    checkpoint, thresholds = load_threshold_profile(config_path, profile=profile)
    return load_learned_routing_predictor(
        checkpoint_path=checkpoint,
        device=device,
        thresholds=thresholds,
    )
