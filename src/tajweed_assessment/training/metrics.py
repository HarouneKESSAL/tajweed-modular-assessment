from typing import Iterable, List
import torch

def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())

def masked_classification_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    correct = ((preds == targets) & mask).sum().item()
    return float(correct / total)

def greedy_decode_from_log_probs(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    frame_ids = log_probs.argmax(dim=-1)
    decoded = []

    for seq, length in zip(frame_ids, lengths):
        seq = seq[: int(length)].tolist()
        out = []
        prev = None

        for token in seq:
            if token != blank_id and token != prev:
                out.append(token)
            prev = token

        decoded.append(out)

    return decoded

def phoneme_token_accuracy(pred_seqs: Iterable[List[int]], target_seqs: Iterable[torch.Tensor]) -> float:
    correct = 0
    total = 0

    for pred, target in zip(pred_seqs, target_seqs):
        gold = target.tolist()
        n = min(len(pred), len(gold))
        correct += sum(int(p == t) for p, t in zip(pred[:n], gold[:n]))
        total += len(gold)

    return correct / total if total > 0 else 0.0

def phoneme_sequence_accuracy(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    targets: Iterable[torch.Tensor],
    blank_id: int = 0,
) -> float:
    decoded = greedy_decode_from_log_probs(log_probs, lengths, blank_id=blank_id)
    total = len(decoded)
    if total == 0:
        return 0.0

    correct = 0
    for pred, gold in zip(decoded, targets):
        if pred == gold.tolist():
            correct += 1

    return correct / total

def phoneme_accuracy_from_log_probs(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    targets: Iterable[torch.Tensor],
    blank_id: int = 0,
) -> float:
    decoded = greedy_decode_from_log_probs(log_probs, lengths, blank_id=blank_id)
    return phoneme_token_accuracy(decoded, targets)
