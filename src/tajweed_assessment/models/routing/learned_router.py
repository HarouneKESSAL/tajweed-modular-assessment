from __future__ import annotations

import torch
import torch.nn as nn


def routing_label_names() -> list[str]:
    return ["use_duration", "use_transition", "use_burst"]


class LearnedRoutingModule(nn.Module):
    """
    Lightweight multi-label router.

    Input:
      feature vector from text/audio

    Output:
      independent logits for module decisions:
      - use_duration
      - use_transition
      - use_burst
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_outputs: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def logits_to_routing_plan(
    logits: torch.Tensor,
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, bool]]:
    labels = routing_label_names()

    if thresholds is None:
        thresholds = {label: 0.5 for label in labels}

    probs = torch.sigmoid(logits).detach().cpu()
    plans: list[dict[str, bool]] = []

    for row in probs:
        plan = {}
        for label, prob in zip(labels, row.tolist()):
            plan[label] = float(prob) >= float(thresholds.get(label, 0.5))
        plans.append(plan)

    return plans
