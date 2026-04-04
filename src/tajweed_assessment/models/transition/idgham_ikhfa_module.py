import torch
import torch.nn as nn
from tajweed_assessment.models.common.bilstm_encoder import BiLSTMEncoder

class TransitionRuleModule(nn.Module):
    def __init__(self, mfcc_dim: int = 39, ssl_dim: int = 64, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1, num_rules: int = 3) -> None:
        super().__init__()
        self.mfcc_proj = nn.Linear(mfcc_dim, hidden_dim)
        self.ssl_proj = nn.Linear(ssl_dim, hidden_dim)
        self.encoder = BiLSTMEncoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Linear(self.encoder.output_dim, num_rules)

    def forward(self, mfcc: torch.Tensor, ssl: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.mfcc_proj(mfcc), self.ssl_proj(ssl)], dim=-1)
        encoded = self.encoder(fused, lengths)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
