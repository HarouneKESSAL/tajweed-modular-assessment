from dataclasses import dataclass
import torch
import torch.nn as nn
from tajweed_assessment.data.labels import BLANK_ID, IGNORE_INDEX

@dataclass
class DurationLossOutput:
    total: torch.Tensor
    ctc: torch.Tensor
    rule: torch.Tensor

class DurationLoss(nn.Module):
    def __init__(self, lambda_ctc: float = 1.0, lambda_rule: float = 1.0) -> None:
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.lambda_rule = lambda_rule
        self.ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
        self.rule = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, log_probs, rule_logits, phoneme_targets, input_lengths, target_lengths, rule_targets) -> DurationLossOutput:
        ctc_loss = self.ctc(log_probs.transpose(0, 1), phoneme_targets, input_lengths, target_lengths)
        rule_loss = self.rule(rule_logits.reshape(-1, rule_logits.size(-1)), rule_targets.reshape(-1))
        total = self.lambda_ctc * ctc_loss + self.lambda_rule * rule_loss
        return DurationLossOutput(total=total, ctc=ctc_loss, rule=rule_loss)
