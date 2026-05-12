from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import WhisperModel


@dataclass(frozen=True)
class WhisperCTCConfig:
    model_name: str = "openai/whisper-tiny"
    num_labels: int = 2
    freeze_encoder: bool = True
    dropout: float = 0.1
    head_type: str = "linear"
    head_hidden_dim: int = 128
    head_num_layers: int = 1


class WhisperCTCContentModel(nn.Module):
    """
    Whisper encoder + Quran-specific CTC head.

    The Whisper decoder is not used. This keeps decoding constrained to the
    project's Quran-specific Arabic character vocabulary.
    """

    def __init__(self, config: WhisperCTCConfig) -> None:
        super().__init__()
        self.config = config
        self.whisper = WhisperModel.from_pretrained(config.model_name)
        self.encoder = self.whisper.encoder
        self.dropout = nn.Dropout(config.dropout)

        encoder_dim = int(self.whisper.config.d_model)

        if config.head_type == "linear":
            self.head = nn.Linear(encoder_dim, config.num_labels)
        elif config.head_type == "bilstm":
            self.head = nn.Sequential(
                BiLSTMHead(
                    input_dim=encoder_dim,
                    hidden_dim=config.head_hidden_dim,
                    num_layers=config.head_num_layers,
                    dropout=config.dropout,
                ),
                nn.Dropout(config.dropout),
                nn.Linear(config.head_hidden_dim * 2, config.num_labels),
            )
        elif config.head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(encoder_dim, config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.head_hidden_dim, config.num_labels),
            )
        else:
            raise ValueError(f"Unsupported head_type: {config.head_type}")

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(input_features=input_features)
        hidden = self.dropout(encoder_out.last_hidden_state)
        logits = self.head(hidden)
        return torch.log_softmax(logits, dim=-1)


class BiLSTMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


def collapse_ctc_ids(ids: list[int], blank_id: int = 0) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None

    for item in ids:
        item = int(item)
        if item != blank_id and item != previous:
            collapsed.append(item)
        previous = item

    return collapsed


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor | None = None,
    blank_id: int = 0,
) -> list[list[int]]:
    """
    Greedy CTC decoding.

    Args:
        log_probs: [batch, time, classes]
        input_lengths: optional true time lengths
    """
    pred = log_probs.argmax(dim=-1).cpu()
    decoded: list[list[int]] = []

    for batch_idx in range(pred.size(0)):
        length = int(input_lengths[batch_idx].item()) if input_lengths is not None else pred.size(1)
        ids = pred[batch_idx, :length].tolist()
        decoded.append(collapse_ctc_ids(ids, blank_id=blank_id))

    return decoded