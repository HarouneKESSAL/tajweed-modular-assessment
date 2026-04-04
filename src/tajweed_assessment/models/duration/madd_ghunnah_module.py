import torch
import torch.nn as nn
from tajweed_assessment.models.common.bilstm_encoder import BiLSTMEncoder
from tajweed_assessment.models.common.ctc_head import CTCHead
from tajweed_assessment.models.common.rule_head import RuleHead

class DurationRuleModule(nn.Module):
    def __init__(self, input_dim: int = 39, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1, num_phonemes: int = 11, num_rules: int = 6) -> None:
        super().__init__()
        self.encoder = BiLSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.ctc_head = CTCHead(self.encoder.output_dim, num_phonemes)
        self.rule_head = RuleHead(self.encoder.output_dim, num_rules)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x, lengths)
        return self.ctc_head.log_probs(encoded), self.rule_head(encoded)
