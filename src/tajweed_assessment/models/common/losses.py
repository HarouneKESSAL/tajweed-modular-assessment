from dataclasses import dataclass
import torch
import torch.nn as nn
from tajweed_assessment.data.labels import BLANK_ID, IGNORE_INDEX, RULES, rule_to_id

@dataclass
class DurationLossOutput:
    total: torch.Tensor
    ctc: torch.Tensor
    rule: torch.Tensor


def build_rule_class_weights(weight_overrides: dict[str, float] | None = None) -> torch.Tensor | None:
    if not weight_overrides:
        return None
    weights = torch.ones(len(RULES), dtype=torch.float32)
    used = False
    for rule_name, value in weight_overrides.items():
        if rule_name not in rule_to_id:
            continue
        weights[rule_to_id[rule_name]] = float(value)
        used = True
    return weights if used else None


class DurationLoss(nn.Module):
    def __init__(
        self,
        lambda_ctc: float = 1.0,
        lambda_rule: float = 1.0,
        rule_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_ctc = lambda_ctc
        self.lambda_rule = lambda_rule
        self.ctc = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
        if rule_class_weights is not None:
            self.register_buffer("rule_class_weights", rule_class_weights.to(dtype=torch.float32))
        else:
            self.rule_class_weights = None
        self.rule = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=self.rule_class_weights)

    def forward(self, log_probs, rule_logits, phoneme_targets, input_lengths, target_lengths, rule_targets) -> DurationLossOutput:
        ctc_loss = self.ctc(log_probs.transpose(0, 1), phoneme_targets, input_lengths, target_lengths)
        rule_loss = self.rule(rule_logits.reshape(-1, rule_logits.size(-1)), rule_targets.reshape(-1))
        total = self.lambda_ctc * ctc_loss + self.lambda_rule * rule_loss
        return DurationLossOutput(total=total, ctc=ctc_loss, rule=rule_loss)
