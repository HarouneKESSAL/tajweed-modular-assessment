import torch
import torch.nn as nn

class CTCHead(nn.Module):
    def __init__(self, input_dim: int, num_phonemes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, num_phonemes)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.proj(encoded)

    def log_probs(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.forward(encoded).log_softmax(dim=-1)
