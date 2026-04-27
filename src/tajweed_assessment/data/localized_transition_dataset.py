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


DEFAULT_LOCALIZED_TRANSITION_LABEL_VOCAB = ["ikhfa", "idgham"]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_label_vocab(rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels = set()
    for row in rows:
        for span in row.get("transition_rule_time_spans", []):
            rule = span.get("rule")
            if rule:
                labels.add(str(rule))
    return sorted(labels) if labels else list(DEFAULT_LOCALIZED_TRANSITION_LABEL_VOCAB)


def labels_from_spans(spans: Sequence[Dict[str, Any]]) -> List[str]:
    labels = set()
    for span in spans:
        rule = span.get("rule")
        if rule:
            labels.add(str(rule))
    return sorted(labels)


def multi_hot(rule_names: Sequence[str], vocab: Sequence[str]) -> torch.Tensor:
    rule_set = set(rule_names)
    return torch.tensor([1.0 if label in rule_set else 0.0 for label in vocab], dtype=torch.float32)


def build_frame_targets(
    *,
    num_frames: int,
    spans: Sequence[Dict[str, Any]],
    label_vocab: Sequence[str],
    frame_hop_sec: float,
    pad_sec: float = 0.03,
    min_span_sec: float = 0.08,
    soft: bool = True,
) -> torch.Tensor:
    y = torch.zeros(num_frames, len(label_vocab), dtype=torch.float32)
    label_to_idx = {label: i for i, label in enumerate(label_vocab)}
    total_duration_sec = num_frames * frame_hop_sec

    for span in spans:
        label = span.get("rule")
        if label not in label_to_idx:
            continue
        start_sec = span.get("start_sec")
        end_sec = span.get("end_sec")
        if start_sec is None or end_sec is None:
            continue

        start_sec = float(start_sec)
        end_sec = float(end_sec)
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec

        start_sec = max(0.0, start_sec - pad_sec)
        end_sec = min(total_duration_sec, end_sec + pad_sec)
        if end_sec - start_sec < min_span_sec:
            center = 0.5 * (start_sec + end_sec)
            half = 0.5 * min_span_sec
            start_sec = max(0.0, center - half)
            end_sec = min(total_duration_sec, center + half)

        start_f = int(start_sec / frame_hop_sec)
        end_f = int(end_sec / frame_hop_sec + 0.999999)
        start_f = max(0, min(start_f, num_frames))
        end_f = max(start_f + 1, min(end_f, num_frames))

        label_idx = label_to_idx[label]
        length = end_f - start_f
        if not soft:
            y[start_f:end_f, label_idx] = 1.0
            continue

        if length <= 2:
            y[start_f:end_f, label_idx] = 1.0
            continue

        edge_len = max(1, int(length * 0.2))
        center_len = max(1, int(length * 0.3))
        center_start = start_f + max(0, (length - center_len) // 2)
        center_end = min(end_f, center_start + center_len)
        left_edge_end = min(end_f, start_f + edge_len)
        right_edge_start = max(start_f, end_f - edge_len)

        y[start_f:end_f, label_idx] = torch.maximum(
            y[start_f:end_f, label_idx],
            torch.full((length,), 0.35, dtype=torch.float32),
        )
        if left_edge_end < right_edge_start:
            y[left_edge_end:right_edge_start, label_idx] = torch.maximum(
                y[left_edge_end:right_edge_start, label_idx],
                torch.full((right_edge_start - left_edge_end,), 0.7, dtype=torch.float32),
            )
        if center_start < center_end:
            y[center_start:center_end, label_idx] = torch.maximum(
                y[center_start:center_end, label_idx],
                torch.full((center_end - center_start,), 1.0, dtype=torch.float32),
            )

    return y


class LocalizedTransitionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        indices: Optional[Sequence[int]] = None,
        label_vocab: Optional[Sequence[str]] = None,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        hop_length: int = 160,
        feature_cache_dir: Optional[str | Path] = None,
        speed_config: Optional[SpeedNormalizationConfig] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = load_jsonl(self.manifest_path)
        if indices is not None:
            self.rows = [self.rows[i] for i in indices]

        self.label_vocab = list(label_vocab) if label_vocab is not None else build_label_vocab(self.rows)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.frame_hop_sec = hop_length / sample_rate
        self.speed_config = speed_config
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": self.hop_length, "n_mels": 40},
        )

    def __len__(self) -> int:
        return len(self.rows)

    def label_names(self) -> List[str]:
        return list(self.label_vocab)

    def _cache_path(self, row: Dict[str, Any]) -> Optional[Path]:
        if self.feature_cache_dir is None:
            return None
        speed_tag = "speednorm" if self.speed_config and self.speed_config.enabled else "raw"
        return self.feature_cache_dir / f"{row['id']}_{speed_tag}_localized_transition.pt"

    def _load_local_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        data, sr = sf.read(audio_path, always_2d=True)
        waveform = torch.tensor(data.T, dtype=torch.float32)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, int(sr)

    def _maybe_resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
        return resampler(waveform)

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        mfcc = self._mfcc(waveform).squeeze(0).transpose(0, 1)
        delta = torchaudio.functional.compute_deltas(mfcc.transpose(0, 1)).transpose(0, 1)
        delta2 = torchaudio.functional.compute_deltas(delta.transpose(0, 1)).transpose(0, 1)
        return torch.cat([mfcc, delta, delta2], dim=-1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path)
        else:
            waveform, sr = self._load_local_audio(row["audio_path"])
            waveform = self._maybe_resample(waveform, sr)
            waveform, _ = maybe_normalize_waveform_speed(waveform, self.sample_rate, self.speed_config)
            x = self._extract_features(waveform)
            if cache_path is not None:
                torch.save(x, cache_path)

        spans = list(row.get("transition_rule_time_spans", []))
        frame_targets_soft = build_frame_targets(
            num_frames=x.shape[0],
            spans=spans,
            label_vocab=self.label_vocab,
            frame_hop_sec=self.frame_hop_sec,
            soft=True,
        )
        frame_targets_hard = build_frame_targets(
            num_frames=x.shape[0],
            spans=spans,
            label_vocab=self.label_vocab,
            frame_hop_sec=self.frame_hop_sec,
            soft=False,
        )
        clip_target = multi_hot(labels_from_spans(spans), self.label_vocab)

        return {
            "id": row["id"],
            "surah_name": row.get("surah_name"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
            "normalized_text": row.get("normalized_text", ""),
            "transition_label": row.get("transition_label", "none"),
            "x": x,
            "input_length": x.shape[0],
            "frame_targets_soft": frame_targets_soft,
            "frame_targets_hard": frame_targets_hard,
            "clip_target": clip_target,
        }

    def summary(self) -> Dict[str, Any]:
        label_counts = {label: 0 for label in self.label_vocab}
        rows_with_any = 0
        for row in self.rows:
            labels = labels_from_spans(row.get("transition_rule_time_spans", []))
            if labels:
                rows_with_any += 1
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
        return {
            "manifest_path": str(self.manifest_path),
            "num_rows": len(self.rows),
            "label_vocab": list(self.label_vocab),
            "clip_label_counts": label_counts,
            "rows_with_any_label": rows_with_any,
        }


def collate_localized_transition_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [item["x"] for item in batch]
    ys_soft = [item["frame_targets_soft"] for item in batch]
    ys_hard = [item["frame_targets_hard"] for item in batch]

    x_pad = pad_sequence(xs, batch_first=True)
    y_soft_pad = pad_sequence(ys_soft, batch_first=True)
    y_hard_pad = pad_sequence(ys_hard, batch_first=True)
    lengths = torch.tensor([item["input_length"] for item in batch], dtype=torch.long)
    clip_targets = torch.stack([item["clip_target"] for item in batch], dim=0)
    max_t = x_pad.size(1)
    frame_mask = torch.arange(max_t).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "id": [item["id"] for item in batch],
        "surah_name": [item["surah_name"] for item in batch],
        "quranjson_verse_key": [item["quranjson_verse_key"] for item in batch],
        "normalized_text": [item["normalized_text"] for item in batch],
        "transition_label": [item["transition_label"] for item in batch],
        "x": x_pad,
        "input_lengths": lengths,
        "frame_targets_soft": y_soft_pad,
        "frame_targets_hard": y_hard_pad,
        "clip_targets": clip_targets,
        "frame_mask": frame_mask,
    }
