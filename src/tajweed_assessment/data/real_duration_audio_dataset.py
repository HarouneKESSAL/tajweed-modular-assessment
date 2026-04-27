from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchaudio

from tajweed_assessment.data.speed import (
    SpeedNormalizationConfig,
    maybe_normalize_waveform_speed,
)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_label_vocab(rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels = set()
    for row in rows:
        for rule in row.get("duration_rules", []):
            labels.add(str(rule))
    return sorted(labels)


def multi_hot(rule_names: Sequence[str], vocab: Sequence[str]) -> torch.Tensor:
    rule_set = set(rule_names)
    return torch.tensor([1.0 if label in rule_set else 0.0 for label in vocab], dtype=torch.float32)


class RealDurationAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        indices: Optional[Sequence[int]] = None,
        rule_vocab: Optional[Sequence[str]] = None,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        feature_cache_dir: Optional[str | Path] = None,
        speed_config: Optional[SpeedNormalizationConfig] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = load_jsonl(self.manifest_path)

        if indices is not None:
            self.rows = [self.rows[i] for i in indices]

        self.rule_vocab = list(rule_vocab) if rule_vocab is not None else build_label_vocab(self.rows)

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.speed_config = speed_config
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
            },
        )

    def __len__(self) -> int:
        return len(self.rows)

    def label_vocab(self) -> List[str]:
        return list(self.rule_vocab)

    def _cache_path(self, row: Dict[str, Any]) -> Optional[Path]:
        if self.feature_cache_dir is None:
            return None
        speed_tag = "speednorm" if self.speed_config and self.speed_config.enabled else "raw"
        return self.feature_cache_dir / f"{row['id']}_{speed_tag}.pt"

    def _load_local_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        data, sr = sf.read(audio_path, always_2d=True)
        waveform = torch.tensor(data.T, dtype=torch.float32)  # [C, T]

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, int(sr)

    def _maybe_resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
        return resampler(waveform)

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self._mfcc(waveform)               # [1, 13, frames]
        mfcc = mfcc.squeeze(0).transpose(0, 1)   # [frames, 13]

        delta = torchaudio.functional.compute_deltas(mfcc.transpose(0, 1)).transpose(0, 1)
        delta2 = torchaudio.functional.compute_deltas(delta.transpose(0, 1)).transpose(0, 1)

        return torch.cat([mfcc, delta, delta2], dim=-1)  # [frames, 39]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cache_path = self._cache_path(row)

        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path)
        else:
            audio_path = row.get("audio_path")
            if not audio_path:
                raise RuntimeError(
                    f"Row {row['id']} has no local audio_path. "
                    "Rebuild retasy_train.jsonl and merge manifests from local audio."
                )

            waveform, sr = self._load_local_audio(audio_path)
            waveform = self._maybe_resample(waveform, sr)
            waveform, _ = maybe_normalize_waveform_speed(waveform, self.sample_rate, self.speed_config)
            x = self._extract_features(waveform)

            if cache_path is not None:
                torch.save(x, cache_path)

        target = multi_hot(row.get("duration_rules", []), self.rule_vocab)

        return {
            "id": row["id"],
            "hf_index": int(row["hf_index"]),
            "surah_name": row.get("surah_name"),
            "hf_surah_number": row.get("hf_surah_number"),
            "quranjson_surah_number": row.get("quranjson_surah_number"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
            "quranjson_verse_index": row.get("quranjson_verse_index"),
            "aya_text": row.get("aya_text", ""),
            "duration_rules": list(row.get("duration_rules", [])),
            "target": target,
            "x": x,
            "input_length": x.shape[0],
        }


def collate_real_duration_audio_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [item["x"] for item in batch]
    x_pad = pad_sequence(xs, batch_first=True)

    lengths = torch.tensor([item["input_length"] for item in batch], dtype=torch.long)
    targets = torch.stack([item["target"] for item in batch], dim=0)

    return {
        "id": [item["id"] for item in batch],
        "hf_index": [item["hf_index"] for item in batch],
        "surah_name": [item["surah_name"] for item in batch],
        "aya_text": [item["aya_text"] for item in batch],
        "duration_rules": [item["duration_rules"] for item in batch],
        "x": x_pad,
        "input_lengths": lengths,
        "target": targets,
    }
