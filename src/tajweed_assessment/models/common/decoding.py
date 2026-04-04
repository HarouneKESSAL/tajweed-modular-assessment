from collections import Counter
from typing import List, Tuple
import torch
from tajweed_assessment.data.labels import BLANK_ID

def greedy_ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
    frame_paths = log_probs.argmax(dim=-1)
    decoded: List[List[int]] = []
    for path, length in zip(frame_paths, lengths):
        seq: List[int] = []
        prev = None
        for pid in path[: int(length)].tolist():
            if pid != BLANK_ID and pid != prev:
                seq.append(pid)
            prev = pid
        decoded.append(seq)
    return decoded

def decode_with_majority_rules(log_probs: torch.Tensor, rule_logits: torch.Tensor, length: int) -> Tuple[List[int], List[int]]:
    path = log_probs.argmax(dim=-1)[:length].tolist()
    rule_path = rule_logits.argmax(dim=-1)[:length].tolist()

    phonemes: List[int] = []
    grouped: List[List[int]] = []
    prev = BLANK_ID
    for pid, rid in zip(path, rule_path):
        if pid == BLANK_ID:
            prev = BLANK_ID
            continue
        if pid != prev:
            phonemes.append(pid)
            grouped.append([rid])
        else:
            grouped[-1].append(rid)
        prev = pid
    majority = [Counter(votes).most_common(1)[0][0] for votes in grouped]
    return phonemes, majority
