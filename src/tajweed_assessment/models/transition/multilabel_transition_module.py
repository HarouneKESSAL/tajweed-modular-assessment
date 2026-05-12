from __future__ import annotations

import torch
import torch.nn as nn

from tajweed_assessment.models.common.bilstm_encoder import BiLSTMEncoder


class TransitionMultiLabelModule(nn.Module):
    """
    Verse-level multi-label transition classifier.

    This model can predict multiple transition rule families for one verse,
    for example both ikhfa and idgham.

    Inputs match the existing TransitionRuleModule style:
    - mfcc: [batch, time, mfcc_dim]
    - ssl: [batch, time, ssl_dim]
    - lengths: [batch]
    """

    def __init__(
        self,
        mfcc_dim: int = 39,
        ssl_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.mfcc_proj = nn.Linear(mfcc_dim, hidden_dim)
        self.ssl_proj = nn.Linear(ssl_dim, hidden_dim)
        self.encoder = BiLSTMEncoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)

    def forward(
        self,
        mfcc: torch.Tensor,
        ssl: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        mfcc_hidden = self.mfcc_proj(mfcc)
        ssl_hidden = self.ssl_proj(ssl)
        fused = torch.cat([mfcc_hidden, ssl_hidden], dim=-1)

        encoded = self.encoder(fused, lengths)

        mask = (
            torch.arange(encoded.size(1), device=encoded.device)
            .unsqueeze(0)
            .lt(lengths.unsqueeze(1))
        )
        mask = mask.unsqueeze(-1)

        encoded = encoded.masked_fill(~mask, 0.0)
        pooled = encoded.sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)

        return self.classifier(self.dropout(pooled))


def transition_multilabel_label_names() -> list[str]:
    return ["ikhfa", "idgham"]


def normalize_transition_rule(rule: str) -> str | None:
    value = str(rule).lower().strip()

    if value.startswith("ikhfa"):
        return "ikhfa"
    if value.startswith("idgham"):
        return "idgham"

    return None


def transition_rules_to_multihot(
    rules: list[str],
    label_names: list[str] | None = None,
) -> list[float]:
    if label_names is None:
        label_names = transition_multilabel_label_names()

    present = set()
    for rule in rules:
        normalized = normalize_transition_rule(rule)
        if normalized is not None:
            present.add(normalized)

    return [1.0 if label in present else 0.0 for label in label_names]


def logits_to_multilabel_predictions(
    logits: torch.Tensor,
    label_names: list[str] | None = None,
    threshold: float = 0.5,
) -> list[list[str]]:
    if label_names is None:
        label_names = transition_multilabel_label_names()

    probs = torch.sigmoid(logits).detach().cpu()
    outputs: list[list[str]] = []

    for row in probs:
        labels = [
            label
            for label, prob in zip(label_names, row.tolist())
            if float(prob) >= threshold
        ]
        outputs.append(labels)

    return outputs
