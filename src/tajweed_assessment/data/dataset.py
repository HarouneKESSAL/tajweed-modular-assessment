import random
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset

from tajweed_assessment.data.labels import (
    BLANK_ID,
    IGNORE_INDEX,
    PHONEMES,
    RULES,
    TRANSITION_RULES,
    id_to_phoneme,
    normalize_rule_name,
    phoneme_to_id,
    rule_to_id,
    transition_rule_to_id,
)
from tajweed_assessment.data.manifests import load_manifest
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor

class ToyDurationDataset(Dataset):
    def __init__(self, n_samples: int = 256, input_dim: int = 39, seed: int = 7) -> None:
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.phoneme_centroids = torch.randn(len(PHONEMES), input_dim)
        self.rule_centroids = torch.randn(len(RULES), input_dim) * 0.2

    def __len__(self) -> int:
        return self.n_samples

    def _sample_phoneme_sequence(self) -> List[int]:
        length = self.rng.randint(4, 6)
        return [self.rng.randint(1, len(PHONEMES) - 1) for _ in range(length)]

    def _assign_rules(self, phoneme_ids: List[int]) -> List[int]:
        rules: List[int] = []
        for i, pid in enumerate(phoneme_ids):
            p = id_to_phoneme[pid]
            nxt = id_to_phoneme[phoneme_ids[i + 1]] if i + 1 < len(phoneme_ids) else None
            if p in {"a", "i", "u"} and self.rng.random() < 0.35:
                rules.append(rule_to_id["madd"])
            elif p == "n" and nxt in {"q", "k"}:
                rules.append(rule_to_id["ikhfa"])
            elif p == "n" and nxt in {"l", "m", "y"}:
                rules.append(rule_to_id["idgham"])
            elif p == "n" and self.rng.random() < 0.25:
                rules.append(rule_to_id["ghunnah"])
            elif p == "q" and self.rng.random() < 0.25:
                rules.append(rule_to_id["qalqalah"])
            else:
                rules.append(rule_to_id["none"])
        return rules

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        phoneme_ids = self._sample_phoneme_sequence()
        rule_ids = self._assign_rules(phoneme_ids)

        frame_phonemes: List[int] = []
        frame_rules: List[int] = []

        for pid, rid in zip(phoneme_ids, rule_ids):
            pre_blank = 0
            frame_phonemes += [BLANK_ID] * pre_blank
            frame_rules += [IGNORE_INDEX] * pre_blank

            n_frames = self.rng.randint(5, 8)
            frame_phonemes += [pid] * n_frames
            frame_rules += [rid] * n_frames

        post_blank = 0
        frame_phonemes += [BLANK_ID] * post_blank
        frame_rules += [IGNORE_INDEX] * post_blank

        frame_phonemes_t = torch.tensor(frame_phonemes, dtype=torch.long)
        frame_rules_t = torch.tensor(frame_rules, dtype=torch.long)

        feat_rules = frame_rules_t.clone()
        feat_rules[feat_rules == IGNORE_INDEX] = rule_to_id["none"]

        x = (
            self.phoneme_centroids[frame_phonemes_t]
            + self.rule_centroids[feat_rules]
            + 0.03 * torch.randn(len(frame_phonemes_t), self.input_dim)
        )

        return {
            "x": x,
            "phoneme_targets": torch.tensor(phoneme_ids, dtype=torch.long),
            "rule_targets": frame_rules_t,
            "canonical_rules": torch.tensor(rule_ids, dtype=torch.long),
            "word": "toy_word",
        }
        
class ToyTransitionDataset(Dataset):
    def __init__(self, n_samples: int = 12, seq_len: int = 24, mfcc_dim: int = 39, ssl_dim: int = 64, seed: int = 7) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.mfcc_dim = mfcc_dim
        self.ssl_dim = ssl_dim
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.rng.choice(list(transition_rule_to_id.values()))
        mfcc = torch.randn(self.seq_len, self.mfcc_dim)
        ssl = torch.randn(self.seq_len, self.ssl_dim)
        if label == transition_rule_to_id["ikhfa"]:
            mfcc[:, 0] += 1.0
        elif label == transition_rule_to_id["idgham"]:
            ssl[:, 0] += 1.0
        return {"mfcc": mfcc, "ssl": ssl, "label": torch.tensor(label, dtype=torch.long)}

class ToyBurstDataset(Dataset):
    def __init__(self, n_samples: int = 12, seq_len: int = 24, input_dim: int = 39, seed: int = 7) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.rng.randint(0, 1)
        x = torch.randn(self.seq_len, self.input_dim)
        if label == 1:
            x[self.seq_len // 2, :5] += 3.0
        return {"x": x, "label": torch.tensor(label, dtype=torch.long)}

class ManifestDurationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        speed_config: SpeedNormalizationConfig | None = None,
    ) -> None:
        self.entries = load_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.speed_config = speed_config

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        if entry.feature_path:
            x = torch.load(entry.feature_path)
        else:
            x = extract_mfcc_features(
                entry.audio_path,
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                speed_config=self.speed_config,
            )
        phoneme_targets = torch.tensor([phoneme_to_id[p] for p in (entry.canonical_phonemes or ["m", "a", "l"])], dtype=torch.long)
        canonical_rules = torch.tensor(
            [rule_to_id[normalize_rule_name(r)] for r in (entry.canonical_rules or ["none"] * len(phoneme_targets))],
            dtype=torch.long,
        )
        frame_rules = torch.full((x.size(0),), IGNORE_INDEX, dtype=torch.long)
        if len(canonical_rules) > 0:
            span = max(1, x.size(0) // len(canonical_rules))
            for i, rid in enumerate(canonical_rules.tolist()):
                frame_rules[i * span : min(x.size(0), (i + 1) * span)] = rid
        return {
            "x": x,
            "phoneme_targets": phoneme_targets,
            "rule_targets": frame_rules,
            "canonical_rules": canonical_rules,
            "word": entry.text or entry.sample_id,
        }

class ManifestTransitionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        speed_config: SpeedNormalizationConfig | None = None,
    ) -> None:
        self.entries = load_manifest(manifest_path)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.speed_config = speed_config
        self.ssl = DummySSLFeatureExtractor(output_dim=64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        mfcc = extract_mfcc_features(
            entry.audio_path,
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            speed_config=self.speed_config,
        )
        ssl = self.ssl.from_mfcc(mfcc)
        raw_label = (entry.canonical_rules or ["none"])[0]
        label_name = normalize_rule_name(raw_label)
        if label_name not in TRANSITION_RULES:
            label_name = "none"
        return {"mfcc": mfcc, "ssl": ssl, "label": torch.tensor(transition_rule_to_id[label_name], dtype=torch.long)}
