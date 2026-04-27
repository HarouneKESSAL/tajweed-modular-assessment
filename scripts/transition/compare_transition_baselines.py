from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name
from tajweed_assessment.data.localized_transition_dataset import (
    LocalizedTransitionDataset,
    collate_localized_transition_batch,
    load_jsonl,
)
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json
from tajweed_assessment.utils.seed import seed_everything


LOCALIZED_LABELS = ["idgham", "ikhfa"]


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


class LocalizedTransitionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)


def split_indices_by_reciter(rows, val_fraction: float = 0.2, seed: int = 7):
    groups = {}
    for idx, row in enumerate(rows):
        key = row.get("reciter_id") or "Unknown"
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    for key in group_keys:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(groups[key])
    train_idx, val_idx = [], []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    if not train_idx or not val_idx:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))
    return train_idx, val_idx


def gold_label_index(row: dict) -> int:
    label = normalize_rule_name(row.get("transition_label", "none"))
    if label not in TRANSITION_RULES:
        labels = {span.get("rule") for span in row.get("transition_rule_time_spans", []) if span.get("rule")}
        if "idgham" in labels:
            label = "idgham"
        elif "ikhfa" in labels:
            label = "ikhfa"
        else:
            label = "none"
    return TRANSITION_RULES.index(label)


def safe_accuracy(correct: int, total: int) -> float | None:
    return (correct / total) if total > 0 else None


def format_acc(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def load_whole_verse_model() -> TransitionRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / preferred_transition_checkpoint())
    model = TransitionRuleModule(
        mfcc_dim=model_cfg["mfcc_dim"],
        ssl_dim=model_cfg["ssl_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_whole_verse_thresholds() -> dict[str, float]:
    path = PROJECT_ROOT / "checkpoints" / "transition_thresholds.json"
    if not path.exists():
        return {}
    data = load_json(path)
    raw = data.get("thresholds", data) if isinstance(data, dict) else {}
    return {str(k): float(v) for k, v in raw.items()}


def load_localized_thresholds() -> dict[str, float]:
    path = PROJECT_ROOT / "checkpoints" / "localized_transition_decoder.json"
    if not path.exists():
        return {label: 0.5 for label in LOCALIZED_LABELS}
    data = load_json(path)
    raw = data.get("decoder_config", data) if isinstance(data, dict) else {}
    return {label: float(raw.get(label, {}).get("threshold", 0.5)) for label in LOCALIZED_LABELS}


def contiguous_spans_from_probs(probs: torch.Tensor, threshold: float) -> list[dict]:
    pred = probs >= threshold
    spans = []
    start = None
    for i, flag in enumerate(pred.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append({"start_frame": start, "end_frame": i})
            start = None
    if start is not None:
        spans.append({"start_frame": start, "end_frame": len(pred)})
    return spans


def evaluate_whole_verse(rows: list[dict]) -> dict:
    model = load_whole_verse_model().to("cpu").eval()
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)
    thresholds = load_whole_verse_thresholds()
    confusion = torch.zeros(len(TRANSITION_RULES), len(TRANSITION_RULES), dtype=torch.long)

    for row in rows:
        mfcc = extract_mfcc_features(row["audio_path"]).unsqueeze(0)
        ssl = ssl_extractor.from_mfcc(mfcc[0]).unsqueeze(0)
        lengths = torch.tensor([mfcc.size(1)], dtype=torch.long)
        with torch.no_grad():
            logits = model(mfcc, ssl, lengths)
            probs = logits.softmax(dim=-1)[0]
        if thresholds:
            pred = int(probs[1:].argmax().item()) + 1
            pred_name = TRANSITION_RULES[pred]
            if float(probs[pred].item()) < float(thresholds.get(pred_name, 0.5)):
                pred = 0
        else:
            pred = int(logits.argmax(dim=-1)[0].item())
        gold = gold_label_index(row)
        confusion[gold, pred] += 1

    return summarize_confusion(confusion)


def evaluate_localized(rows: list[dict], indices: list[int]) -> dict:
    ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "localized_transition_model.pt", map_location="cpu")
    config = ckpt.get("config", {})
    label_vocab = ckpt["label_vocab"]
    thresholds = load_localized_thresholds()

    dataset = LocalizedTransitionDataset(
        PROJECT_ROOT / "data/alignment/transition_time_projection_strict.jsonl",
        indices=indices,
        label_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        hop_length=int(config.get("hop_length", 160)),
        feature_cache_dir=PROJECT_ROOT / "data/interim/localized_transition_mfcc_cache",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_localized_transition_batch,
    )

    model = LocalizedTransitionBiLSTM(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    ).to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    confusion = torch.zeros(len(TRANSITION_RULES), len(TRANSITION_RULES), dtype=torch.long)
    span_activation = {label: {"clips": 0, "total_spans": 0} for label in label_vocab}

    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            logits = model(batch["x"], batch["input_lengths"]).cpu()
            probs = torch.sigmoid(logits)
            for b in range(probs.size(0)):
                row = dataset.rows[row_offset + b]
                length = int(batch["input_lengths"][b].item())
                clip_probs = probs[b, :length].max(dim=0).values

                positive = []
                for j, label in enumerate(label_vocab):
                    threshold = float(thresholds.get(label, 0.5))
                    if float(clip_probs[j].item()) >= threshold:
                        positive.append((label, float(clip_probs[j].item())))
                        spans = contiguous_spans_from_probs(probs[b, :length, j], threshold)
                        span_activation[label]["clips"] += 1
                        span_activation[label]["total_spans"] += len(spans)

                if positive:
                    pred_label = max(positive, key=lambda item: item[1])[0]
                else:
                    pred_label = "none"

                gold = gold_label_index(row)
                pred = TRANSITION_RULES.index(pred_label)
                confusion[gold, pred] += 1
            row_offset += probs.size(0)

    summary = summarize_confusion(confusion)
    summary["decoder_thresholds"] = thresholds
    summary["span_activation"] = {
        label: {
            **stats,
            "avg_spans_per_positive_clip": (stats["total_spans"] / stats["clips"]) if stats["clips"] else None,
        }
        for label, stats in span_activation.items()
    }
    return summary


def summarize_confusion(confusion: torch.Tensor) -> dict:
    per_class = {}
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    for idx, label in enumerate(TRANSITION_RULES):
        support = int(confusion[idx].sum().item())
        hits = int(confusion[idx, idx].item())
        per_class[label] = {
            "total": support,
            "correct": hits,
            "accuracy": safe_accuracy(hits, support),
        }
    return {
        "samples": total,
        "accuracy": safe_accuracy(correct, total),
        "confusion_matrix": confusion.tolist(),
        "class_summary": per_class,
    }


def print_summary(title: str, summary: dict) -> None:
    print(f"{title}:")
    print(f"- samples={summary['samples']} acc={format_acc(summary['accuracy'])}")
    for label, stats in summary["class_summary"].items():
        print(f"- {label}: correct={stats['correct']} total={stats['total']} acc={format_acc(stats['accuracy'])}")
    if "decoder_thresholds" in summary:
        print(f"- decoder_thresholds={summary['decoder_thresholds']}")
    if "span_activation" in summary:
        for label, stats in summary["span_activation"].items():
            avg = stats["avg_spans_per_positive_clip"]
            avg_text = f"{avg:.2f}" if avg is not None else "n/a"
            print(f"- {label} predicted positive clips={stats['clips']} total_spans={stats['total_spans']} avg_spans_per_positive_clip={avg_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/alignment/transition_time_projection_strict.jsonl")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    seed_everything(7)
    train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=7)
    if args.split == "train":
        indices = train_idx
    elif args.split == "val":
        indices = val_idx
    else:
        indices = list(range(len(rows)))
    subset_rows = [rows[i] for i in indices]

    whole_verse = evaluate_whole_verse(subset_rows)
    localized = evaluate_localized(subset_rows, indices)

    print_summary("Whole-verse transition", whole_verse)
    print("")
    print_summary("Localized transition", localized)

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "split": args.split,
        "whole_verse": whole_verse,
        "localized": localized,
    }

    if args.output_json:
        save_json(summary, PROJECT_ROOT / args.output_json)
        print("")
        print(f"Saved comparison JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

