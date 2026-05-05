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


class WhisperCTCContentModel(nn.Module):
    """
    Whisper encoder + Quran-specific CTC head.

    This intentionally does not use Whisper's text decoder. The encoder provides
    acoustic features, while the CTC head learns the project's Arabic character
    vocabulary.
    """

    def __init__(self, config: WhisperCTCConfig) -> None:
        super().__init__()
        self.config = config
        self.whisper = WhisperModel.from_pretrained(config.model_name)
        self.encoder = self.whisper.encoder
        self.dropout = nn.Dropout(config.dropout)
        self.ctc_head = nn.Linear(self.whisper.config.d_model, config.num_labels)

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder(input_features=input_features)
        hidden = self.dropout(encoder_out.last_hidden_state)
        logits = self.ctc_head(hidden)
        return torch.log_softmax(logits, dim=-1)


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