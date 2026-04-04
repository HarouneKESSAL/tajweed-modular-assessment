import torch
import torch.nn as nn

class RuleHead(nn.Module):
    def __init__(self, input_dim: int, num_rules: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, num_rules)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.proj(encoded)
