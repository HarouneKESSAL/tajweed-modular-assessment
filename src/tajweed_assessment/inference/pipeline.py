from dataclasses import dataclass
from typing import Optional
import torch

from tajweed_assessment.data.labels import id_to_rule
from tajweed_assessment.models.common.decoding import decode_with_majority_rules
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.models.fusion.aggregator import aggregate_diagnosis
from tajweed_assessment.models.fusion.feedback import render_feedback

@dataclass
class TajweedInferencePipeline:
    duration_module: DurationRuleModule
    transition_module: Optional[TransitionRuleModule] = None
    burst_module: Optional[QalqalahCNN] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.duration_module.to(self.device).eval()
        if self.transition_module is not None:
            self.transition_module.to(self.device).eval()
        if self.burst_module is not None:
            self.burst_module.to(self.device).eval()

    @torch.no_grad()
    def run_duration_only(self, x: torch.Tensor, input_length: int, canonical_phonemes: list[int], canonical_rules: list[int], word: str = "sample") -> dict:
        batch_x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([input_length], dtype=torch.long, device=self.device)
        log_probs, rule_logits = self.duration_module(batch_x, lengths)
        pred_phonemes, pred_rules = decode_with_majority_rules(log_probs[0].cpu(), rule_logits[0].cpu(), input_length)

        judgments = []
        for idx, expected in enumerate(canonical_rules):
            pred = pred_rules[idx] if idx < len(pred_rules) else 0
            judgments.append({
                "position": idx,
                "rule": id_to_rule.get(expected, "none"),
                "predicted_rule": id_to_rule.get(pred, "none"),
                "is_correct": expected == pred,
                "detail": "" if expected == pred else f"expected {id_to_rule.get(expected, 'none')} but got {id_to_rule.get(pred, 'none')}",
            })

        report = aggregate_diagnosis(
            word=word,
            canonical_phonemes=canonical_phonemes,
            predicted_phonemes=pred_phonemes,
            canonical_rules=canonical_rules,
            module_judgments=judgments,
        )
        return {"report": report.to_dict(), "feedback": render_feedback(report)}
