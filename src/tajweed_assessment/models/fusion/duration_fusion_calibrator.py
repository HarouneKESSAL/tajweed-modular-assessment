from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


DURATION_FUSION_LABELS = ("madd", "ghunnah")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_duration_char_vocab(texts: Iterable[Iterable[str]]) -> dict[str, int]:
    chars = set()
    for text in texts:
        chars.update(str(ch) for ch in text if str(ch))
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_duration_context_chars(
    canonical_chars: list[str] | None,
    position: int,
    char_vocab: dict[str, int] | None,
) -> tuple[int, int, int]:
    if not canonical_chars or char_vocab is None:
        return 0, 0, 0
    pad_id = int(char_vocab.get(PAD_TOKEN, 0))
    unk_id = int(char_vocab.get(UNK_TOKEN, 1))

    def _lookup(idx: int) -> int:
        if idx < 0 or idx >= len(canonical_chars):
            return pad_id
        ch = str(canonical_chars[idx] or "")
        return int(char_vocab.get(ch, unk_id))

    return _lookup(position - 1), _lookup(position), _lookup(position + 1)


def build_duration_fusion_numeric_features(
    *,
    sequence_predicted_rule: str,
    sequence_confidence: float | None,
    localized_clip_probabilities: dict[str, float] | None,
    localized_predicted_labels: list[str] | None,
) -> list[float]:
    clip_probabilities = localized_clip_probabilities or {}
    predicted_labels = set(str(label) for label in (localized_predicted_labels or []))
    madd_prob = float(clip_probabilities.get("madd", 0.0))
    ghunnah_prob = float(clip_probabilities.get("ghunnah", 0.0))
    seq_conf = float(sequence_confidence or 0.0)

    return [
        1.0 if sequence_predicted_rule == "madd" else 0.0,
        1.0 if sequence_predicted_rule == "ghunnah" else 0.0,
        seq_conf,
        madd_prob,
        ghunnah_prob,
        ghunnah_prob - madd_prob,
        1.0 if "madd" in predicted_labels else 0.0,
        1.0 if "ghunnah" in predicted_labels else 0.0,
    ]


class DurationFusionCalibrator(nn.Module):
    def __init__(
        self,
        *,
        num_numeric_features: int,
        char_vocab_size: int,
        char_embedding_dim: int = 8,
        hidden_dim: int = 32,
        num_labels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(num_numeric_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + (char_embedding_dim * 3), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        prev_char_ids: torch.Tensor,
        curr_char_ids: torch.Tensor,
        next_char_ids: torch.Tensor,
    ) -> torch.Tensor:
        numeric_repr = self.numeric_encoder(numeric_features)
        prev_emb = self.char_embedding(prev_char_ids)
        curr_emb = self.char_embedding(curr_char_ids)
        next_emb = self.char_embedding(next_char_ids)
        char_repr = torch.cat([prev_emb, curr_emb, next_emb], dim=-1)
        fused = torch.cat([numeric_repr, char_repr], dim=-1)
        return self.classifier(fused)
