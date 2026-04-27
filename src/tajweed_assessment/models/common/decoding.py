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

def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    length: int,
    beam_width: int = 5,
    blank_id: int = BLANK_ID,
) -> List[int]:
    # Based on the standard prefix beam search recursion using log-space scores.
    neg_inf = float("-inf")
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}

    for t in range(int(length)):
        frame = log_probs[t]
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        topk = torch.topk(frame, k=min(beam_width, frame.numel()))
        for token, token_logp in zip(topk.indices.tolist(), topk.values.tolist()):
            for prefix, (p_blank, p_non_blank) in beams.items():
                n_blank, n_non_blank = next_beams.get(prefix, (neg_inf, neg_inf))
                if token == blank_id:
                    n_blank = torch.logsumexp(
                        torch.tensor([n_blank, p_blank + token_logp, p_non_blank + token_logp]),
                        dim=0,
                    ).item()
                    next_beams[prefix] = (n_blank, n_non_blank)
                    continue

                end = prefix[-1] if prefix else None
                new_prefix = prefix + (token,)
                nb_blank, nb_non_blank = next_beams.get(new_prefix, (neg_inf, neg_inf))

                if token == end:
                    n_non_blank = torch.logsumexp(
                        torch.tensor([n_non_blank, p_blank + token_logp]),
                        dim=0,
                    ).item()
                    next_beams[prefix] = (n_blank, n_non_blank)

                    nb_non_blank = torch.logsumexp(
                        torch.tensor([nb_non_blank, p_non_blank + token_logp]),
                        dim=0,
                    ).item()
                    next_beams[new_prefix] = (nb_blank, nb_non_blank)
                else:
                    nb_non_blank = torch.logsumexp(
                        torch.tensor([nb_non_blank, p_blank + token_logp, p_non_blank + token_logp]),
                        dim=0,
                    ).item()
                    next_beams[new_prefix] = (nb_blank, nb_non_blank)

        scored = sorted(
            next_beams.items(),
            key=lambda item: torch.logsumexp(torch.tensor(item[1]), dim=0).item(),
            reverse=True,
        )
        beams = dict(scored[:beam_width])

    best_prefix = max(
        beams.items(),
        key=lambda item: torch.logsumexp(torch.tensor(item[1]), dim=0).item(),
    )[0]
    return list(best_prefix)


def ctc_target_log_probability(
    log_probs: torch.Tensor,
    target_ids: List[int],
    *,
    blank_id: int = BLANK_ID,
) -> float:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be 2D (time, vocab)")
    if not target_ids:
        return float(log_probs[:, blank_id].sum().item())

    extended: List[int] = [blank_id]
    for token in target_ids:
        extended.append(int(token))
        extended.append(blank_id)

    num_states = len(extended)
    neg_inf = float("-inf")
    alpha = log_probs.new_full((num_states,), neg_inf)
    alpha[0] = log_probs[0, blank_id]
    if num_states > 1:
        alpha[1] = log_probs[0, extended[1]]

    for t in range(1, int(log_probs.size(0))):
        next_alpha = log_probs.new_full((num_states,), neg_inf)
        for s, symbol in enumerate(extended):
            candidates = [alpha[s]]
            if s - 1 >= 0:
                candidates.append(alpha[s - 1])
            if s - 2 >= 0 and symbol != blank_id and symbol != extended[s - 2]:
                candidates.append(alpha[s - 2])
            next_alpha[s] = log_probs[t, symbol] + torch.logsumexp(torch.stack(candidates), dim=0)
        alpha = next_alpha

    if num_states == 1:
        return float(alpha[0].item())
    return float(torch.logsumexp(torch.stack([alpha[-1], alpha[-2]]), dim=0).item())


def ctc_lexicon_decode(
    log_probs: torch.Tensor,
    length: int,
    lexicon_targets: List[List[int]],
    *,
    blank_id: int = BLANK_ID,
) -> List[int]:
    if not lexicon_targets:
        return []
    trimmed = log_probs[: int(length)]
    best_target = lexicon_targets[0]
    best_score = ctc_target_log_probability(trimmed, best_target, blank_id=blank_id)
    for target_ids in lexicon_targets[1:]:
        score = ctc_target_log_probability(trimmed, target_ids, blank_id=blank_id)
        if score > best_score:
            best_score = score
            best_target = target_ids
    return list(best_target)
