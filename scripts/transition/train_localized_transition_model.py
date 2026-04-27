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

from tajweed_assessment.data.localized_transition_dataset import (
    LocalizedTransitionDataset,
    collate_localized_transition_batch,
    load_jsonl,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.utils.seed import seed_everything


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


def compute_frame_pos_weight(dataset: LocalizedTransitionDataset, label_vocab):
    pos = torch.zeros(len(label_vocab), dtype=torch.float32)
    total = 0.0
    for i in range(len(dataset)):
        item = dataset[i]
        y = item["frame_targets_hard"]
        pos += y.sum(dim=0)
        total += y.shape[0]
    neg = total - pos
    return neg / pos.clamp(min=1.0)


def apply_pos_weight_overrides(pos_weight: torch.Tensor, label_vocab, overrides):
    if not overrides:
        return pos_weight
    label_to_idx = {label: i for i, label in enumerate(label_vocab)}
    adjusted = pos_weight.clone()
    for label, scale in overrides.items():
        idx = label_to_idx.get(label)
        if idx is None:
            continue
        adjusted[idx] = adjusted[idx] * float(scale)
    return adjusted


def parse_label_overrides(raw_values):
    overrides = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected LABEL=SCALE, got: {raw}")
        label, scale = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in override: {raw}")
        overrides[label] = float(scale)
    return overrides


def masked_bce_loss(logits, targets, frame_mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets)
    mask = frame_mask.unsqueeze(-1).float()
    loss = loss * mask
    denom = mask.sum() * logits.size(-1)
    return loss.sum() / denom.clamp(min=1.0)


def frame_multilabel_metrics_from_flat(logits_flat, targets_flat, threshold=0.5):
    probs = torch.sigmoid(logits_flat)
    preds = (probs >= threshold).int()
    targets = targets_flat.int()
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
    return {
        "frame_macro_f1": float(f1.mean().item()),
        "frame_micro_f1": float(micro_f1.item()),
        "per_label_frame_f1": f1.cpu().tolist(),
    }


def clip_metrics_from_probs(clip_probs, clip_targets, threshold=0.5):
    clip_preds = (clip_probs >= threshold).int()
    clip_targets = clip_targets.int()
    tp = ((clip_preds == 1) & (clip_targets == 1)).sum(dim=0).float()
    fp = ((clip_preds == 1) & (clip_targets == 0)).sum(dim=0).float()
    fn = ((clip_preds == 0) & (clip_targets == 1)).sum(dim=0).float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp).clamp(min=1.0)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp(min=1.0)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)
    exact_match = (clip_preds == clip_targets).all(dim=1).float().mean()
    return {
        "clip_macro_f1": float(f1.mean().item()),
        "clip_micro_f1": float(micro_f1.item()),
        "clip_exact_match": float(exact_match.item()),
    }


def run_epoch(model, loader, optimizer, pos_weight, device: str, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_frame_logits, all_frame_targets = [], []
    all_clip_probs, all_clip_targets = [], []
    for batch in loader:
        x = batch["x"].to(device)
        lengths = batch["input_lengths"].to(device)
        y_soft = batch["frame_targets_soft"].to(device)
        y_hard = batch["frame_targets_hard"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        clip_targets = batch["clip_targets"].to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x, lengths)
            loss = masked_bce_loss(logits, y_soft, frame_mask, pos_weight=pos_weight)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        valid_logits = logits.detach().cpu()[frame_mask.cpu()]
        valid_targets = y_hard.detach().cpu()[frame_mask.cpu()]
        all_frame_logits.append(valid_logits)
        all_frame_targets.append(valid_targets)
        probs = torch.sigmoid(logits.detach().cpu())
        mask = frame_mask.detach().cpu().unsqueeze(-1)
        probs = probs.masked_fill(~mask, 0.0)
        clip_probs = probs.max(dim=1).values
        all_clip_probs.append(clip_probs)
        all_clip_targets.append(clip_targets.detach().cpu())
    frame_logits = torch.cat(all_frame_logits, dim=0)
    frame_targets = torch.cat(all_frame_targets, dim=0)
    clip_probs = torch.cat(all_clip_probs, dim=0)
    clip_targets = torch.cat(all_clip_targets, dim=0)
    return {
        "loss": total_loss / max(len(loader), 1),
        **frame_multilabel_metrics_from_flat(frame_logits, frame_targets),
        **clip_metrics_from_probs(clip_probs, clip_targets),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/alignment/transition_time_projection_strict.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-cache-dir", default="data/interim/localized_transition_mfcc_cache")
    parser.add_argument("--checkpoint", default="checkpoints/localized_transition_model.pt")
    parser.add_argument("--label-pos-weight", action="append", default=[])
    parser.add_argument("--normalize-speed", action="store_true")
    parser.add_argument("--target-speech-rate", type=float, default=12.0)
    parser.add_argument("--max-speed-factor", type=float, default=1.35)
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest_path = PROJECT_ROOT / args.manifest
    rows = load_jsonl(manifest_path)
    train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=args.seed)
    speed_config = SpeedNormalizationConfig(
        enabled=bool(args.normalize_speed),
        target_speech_rate=float(args.target_speech_rate),
        max_speed_factor=float(args.max_speed_factor),
    )

    full_ds = LocalizedTransitionDataset(
        manifest_path,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        hop_length=args.hop_length,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        speed_config=speed_config,
    )
    label_vocab = full_ds.label_names()
    train_ds = LocalizedTransitionDataset(
        manifest_path,
        indices=train_idx,
        label_vocab=label_vocab,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        hop_length=args.hop_length,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        speed_config=speed_config,
    )
    val_ds = LocalizedTransitionDataset(
        manifest_path,
        indices=val_idx,
        label_vocab=label_vocab,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        hop_length=args.hop_length,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        speed_config=speed_config,
    )

    print("Train rows:", len(train_ds))
    print("Val rows  :", len(val_ds))
    print("Labels    :", label_vocab)
    print("Train summary:", train_ds.summary())

    label_overrides = parse_label_overrides(args.label_pos_weight)
    pos_weight = compute_frame_pos_weight(train_ds, label_vocab)
    pos_weight = apply_pos_weight_overrides(pos_weight, label_vocab, label_overrides)
    print("Frame pos_weight:", {label: round(float(weight), 3) for label, weight in zip(label_vocab, pos_weight.tolist())})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LocalizedTransitionBiLSTM(
        input_dim=args.n_mfcc * 3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_labels=len(label_vocab),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_localized_transition_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_localized_transition_batch)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, pos_weight.to(device), device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, pos_weight.to(device), device, train=False)
        score = val_metrics["frame_macro_f1"]
        if score > best_score:
            best_score = score
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
            f"train_frame_macro_f1={train_metrics['frame_macro_f1']:.3f} "
            f"train_frame_micro_f1={train_metrics['frame_micro_f1']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_frame_macro_f1={val_metrics['frame_macro_f1']:.3f} "
            f"val_frame_micro_f1={val_metrics['frame_micro_f1']:.3f} "
            f"val_clip_macro_f1={val_metrics['clip_macro_f1']:.3f} "
            f"val_clip_micro_f1={val_metrics['clip_micro_f1']:.3f} "
            f"val_clip_exact={val_metrics['clip_exact_match']:.3f}"
        )

    final_metrics = run_epoch(model, val_loader, optimizer, pos_weight.to(device), device, train=False)
    print("\nPer-label frame F1 on validation:")
    for label, f1 in zip(label_vocab, final_metrics["per_label_frame_f1"]):
        print(f"  {label:10s} f1={f1:.3f}")


if __name__ == "__main__":
    main()

