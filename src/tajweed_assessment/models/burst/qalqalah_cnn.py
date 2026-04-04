import torch
import torch.nn as nn

class QalqalahCNN(nn.Module):
    def __init__(self, input_dim: int = 39, channels: tuple[int, int] = (16, 32), num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        c1, c2 = channels
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        feats = self.net(x)
        return self.classifier(feats)
