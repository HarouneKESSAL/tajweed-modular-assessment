from typing import Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_duration_batch(batch: List[dict]) -> Dict[str, torch.Tensor | list]:
    xs = [item["x"] for item in batch]
    phoneme_targets = [item["phoneme_targets"] for item in batch]
    rule_targets = [item["rule_targets"] for item in batch]
    canonical_rules = [item["canonical_rules"] for item in batch]
    words = [item["word"] for item in batch]

    x_pad = pad_sequence(xs, batch_first=True)
    rule_pad = pad_sequence(rule_targets, batch_first=True, padding_value=-100)
    input_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in phoneme_targets], dtype=torch.long)
    flat_targets = torch.cat(phoneme_targets, dim=0)

    return {
        "x": x_pad,
        "input_lengths": input_lengths,
        "phoneme_targets": flat_targets,
        "target_lengths": target_lengths,
        "rule_targets": rule_pad,
        "raw_phoneme_targets": phoneme_targets,
        "raw_canonical_rules": canonical_rules,
        "words": words,
    }

def collate_sequence_classification_batch(batch: List[dict]) -> Dict[str, torch.Tensor]:
    keys = [k for k in batch[0].keys() if k != "label"]
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        seqs = [item[key] for item in batch]
        out[key] = pad_sequence(seqs, batch_first=True)
    out["lengths"] = torch.tensor([item[keys[0]].size(0) for item in batch], dtype=torch.long)
    out["label"] = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return out
