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

from tajweed_assessment.data.real_duration_audio_dataset import (
    RealDurationAudioDataset,
    collate_real_duration_audio_batch,
    load_jsonl,
)
from tajweed_assessment.utils.seed import seed_everything


class BiLSTMMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
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
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_t = out.size(1)
        mask = torch.arange(max_t, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        summed = (out * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / denom

        return self.classifier(self.dropout(pooled))


def split_indices_by_reciter(rows, val_fraction: float = 0.2, seed: int = 7):
    by_group = {}
    for i, row in enumerate(rows):
        group = row.get("reciter_id") or "Unknown"
        by_group.setdefault(group, []).append(i)

    groups = list(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_rows = len(rows)
    target_val_rows = int(total_rows * val_fraction)

    val_groups = set()
    val_count = 0

    for g in groups:
        if val_count >= target_val_rows:
            break
        val_groups.add(g)
        val_count += len(by_group[g])

    train_idx = []
    val_idx = []

    for g, indices in by_group.items():
        if g in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if len(val_idx) == 0 or len(train_idx) == 0:
        split = max(1, int(0.8 * total_rows))
        train_idx = list(range(split))
        val_idx = list(range(split, total_rows))

    return train_idx, val_idx


def per_label_f1_from_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    preds = (probs >= threshold).int()
    targets = targets.int()

    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return float(f1.item())


def evaluate_with_thresholds(probs: torch.Tensor, targets: torch.Tensor, thresholds: list[float]):
    thresholds_t = torch.tensor(thresholds, dtype=torch.float32).unsqueeze(0)
    preds = (probs >= thresholds_t).int()
    targets = targets.int()

    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()

    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()

    micro_precision = micro_tp / (micro_tp + micro_fp).clamp(min=1.0)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp(min=1.0)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)

    exact_match = (preds == targets).all(dim=1).float().mean()

    return {
        "macro_f1": float(f1.mean().item()),
        "micro_f1": float(micro_f1.item()),
        "exact_match": float(exact_match.item()),
        "per_label_f1": f1.cpu().tolist(),
        "support": targets.sum(dim=0).cpu().tolist(),
    }


def collect_val_outputs(model, loader, device: str):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["input_lengths"].to(device)
            targets = batch["target"].cpu()

            logits = model(x, lengths)
            probs = torch.sigmoid(logits).cpu()

            all_probs.append(probs)
            all_targets.append(targets)

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_subset.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/real_duration_classifier.pt")
    parser.add_argument("--output", default="checkpoints/real_duration_thresholds.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-cache-dir", default="data/interim/retasy_mfcc_cache")
    parser.add_argument("--device", default="")
    parser.add_argument("--grid-start", type=float, default=0.05)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    manifest_path = PROJECT_ROOT / args.manifest
    out_path = PROJECT_ROOT / args.output

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    label_vocab = checkpoint["label_vocab"]

    seed = int(config.get("seed", 7))
    seed_everything(seed)

    rows = load_jsonl(manifest_path)
    _, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=seed)

    val_ds = RealDurationAudioDataset(
        manifest_path,
        indices=val_idx,
        rule_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_real_duration_audio_batch,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMMultiLabelClassifier(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs, targets = collect_val_outputs(model, val_loader, device)

    baseline_thresholds = [0.5 for _ in label_vocab]
    baseline_metrics = evaluate_with_thresholds(probs, targets, baseline_thresholds)

    thresholds = []
    per_label_best = []

    t = args.grid_start
    grid = []
    while t <= args.grid_stop + 1e-8:
        grid.append(round(t, 4))
        t += args.grid_step

    for j, label in enumerate(label_vocab):
        best_thr = 0.5
        best_f1 = -1.0

        for thr in grid:
            f1 = per_label_f1_from_probs(probs[:, j], targets[:, j], thr)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds.append(best_thr)
        per_label_best.append((label, best_thr, best_f1))

    tuned_metrics = evaluate_with_thresholds(probs, targets, thresholds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "manifest": str(manifest_path),
        "thresholds": {label: thr for label, thr in zip(label_vocab, thresholds)},
        "baseline_metrics_at_0_5": baseline_metrics,
        "tuned_metrics": tuned_metrics,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved thresholds to: {out_path}")
    print()
    print("Baseline @ 0.5")
    print(
        f"  macro_f1={baseline_metrics['macro_f1']:.3f} "
        f"micro_f1={baseline_metrics['micro_f1']:.3f} "
        f"exact={baseline_metrics['exact_match']:.3f}"
    )
    print()
    print("Tuned")
    print(
        f"  macro_f1={tuned_metrics['macro_f1']:.3f} "
        f"micro_f1={tuned_metrics['micro_f1']:.3f} "
        f"exact={tuned_metrics['exact_match']:.3f}"
    )
    print()
    print("Per-label thresholds")
    for label, thr, f1 in per_label_best:
        print(f"  {label:15s} thr={thr:.2f} best_f1={f1:.3f}")


if __name__ == "__main__":
    main()
