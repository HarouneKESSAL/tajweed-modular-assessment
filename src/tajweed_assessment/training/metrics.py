from typing import Iterable
import torch
from tajweed_assessment.models.common.decoding import greedy_ctc_decode

def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())

def phoneme_sequence_accuracy(log_probs: torch.Tensor, lengths: torch.Tensor, targets: Iterable[torch.Tensor]) -> float:
    decoded = greedy_ctc_decode(log_probs, lengths)
    total = len(decoded)
    if total == 0:
        return 0.0
    correct = sum(1 for pred, gold in zip(decoded, targets) if pred == gold.tolist())
    return correct / total
