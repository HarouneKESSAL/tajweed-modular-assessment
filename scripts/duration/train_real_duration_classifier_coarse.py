from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
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


def compute_pos_weight(rows, label_vocab):
    num_labels = len(label_vocab)
    counts = torch.zeros(num_labels, dtype=torch.float32)

    for row in rows:
        labels = set(row.get("duration_rules", []))
        for i, label in enumerate(label_vocab):
            if label in labels:
                counts[i] += 1

    total = float(len(rows))
    neg = total - counts
    pos_weight = neg / counts.clamp(min=1.0)
    return pos_weight


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    targets = targets.int()

    tp = (preds & targets).sum(dim=0).float()
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


def run_epoch(model, loader, optimizer, criterion, device: str, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["input_lengths"].to(device)
        targets = batch["target"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x, lengths)
            loss = criterion(logits, targets)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = multilabel_metrics(logits, targets)

    return {
        "loss": total_loss / max(len(loader), 1),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/manifests/retasy_duration_subset_coarse.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-cache-dir", default="data/interim/retasy_mfcc_cache")
    parser.add_argument("--checkpoint", default="checkpoints/real_duration_classifier_coarse.pt")
    args = parser.parse_args()

    seed_everything(args.seed)

    manifest_path = PROJECT_ROOT / args.manifest
    rows = load_jsonl(manifest_path)

    train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=args.seed)

    full_ds = RealDurationAudioDataset(
        manifest_path,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    label_vocab = full_ds.label_vocab()

    train_ds = RealDurationAudioDataset(
        manifest_path,
        indices=train_idx,
        rule_vocab=label_vocab,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    val_ds = RealDurationAudioDataset(
        manifest_path,
        indices=val_idx,
        rule_vocab=label_vocab,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )

    print("Train rows:", len(train_ds))
    print("Val rows  :", len(val_ds))
    print("Labels    :", label_vocab)

    pos_weight = compute_pos_weight(train_ds.rows, label_vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMMultiLabelClassifier(
        input_dim=args.n_mfcc * 3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_labels=len(label_vocab),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_real_duration_audio_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_real_duration_audio_batch,
    )

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_vocab": label_vocab,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"saved best checkpoint to {checkpoint_path}")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.3f} "
            f"train_micro_f1={train_metrics['micro_f1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"val_micro_f1={val_metrics['micro_f1']:.3f} "
            f"val_exact={val_metrics['exact_match']:.3f}"
        )

    final_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    print("\nPer-label F1 on validation:")
    for label, f1, support in zip(label_vocab, final_metrics["per_label_f1"], final_metrics["support"]):
        print(f"  {label:10s} f1={f1:.3f} support={int(support)}")


if __name__ == "__main__":
    main()
