from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.models.transition.multilabel_transition_module import (
    TransitionMultiLabelModule,
    transition_multilabel_label_names,
)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_audio_mono(path: Path, target_sample_rate: int = 16000) -> torch.Tensor:
    audio, sample_rate = sf.read(str(path), always_2d=True)

    waveform = torch.tensor(audio, dtype=torch.float32).transpose(0, 1)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.transforms.Resample(int(sample_rate), target_sample_rate)(waveform)

    return waveform


def extract_mfcc(path: Path, sample_rate: int = 16000, n_mfcc: int = 39) -> torch.Tensor:
    waveform = load_audio_mono(path, target_sample_rate=sample_rate)

    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
            "center": True,
        },
    )

    mfcc = transform(waveform).squeeze(0).transpose(0, 1).contiguous()
    return mfcc


class TransitionMultiLabelDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], ssl_dim: int = 64) -> None:
        self.rows = rows
        self.ssl_dim = ssl_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = resolve_path(row["audio_path"])

        mfcc = extract_mfcc(audio_path)
        ssl = torch.zeros((mfcc.size(0), self.ssl_dim), dtype=torch.float32)

        target = torch.tensor(row["transition_multihot"], dtype=torch.float32)

        return {
            "id": str(row.get("id") or row.get("sample_id") or index),
            "mfcc": mfcc,
            "ssl": ssl,
            "length": int(mfcc.size(0)),
            "target": target,
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "combo": row.get("transition_combo", ""),
            "label_source": row.get("label_source", ""),
            "text": row.get("text", ""),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_len = max(item["length"] for item in items)
    mfcc_dim = int(items[0]["mfcc"].size(1))
    ssl_dim = int(items[0]["ssl"].size(1))

    mfcc = torch.zeros((batch_size, max_len, mfcc_dim), dtype=torch.float32)
    ssl = torch.zeros((batch_size, max_len, ssl_dim), dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    targets = torch.stack([item["target"] for item in items])
    sample_weights = torch.tensor(
        [float(item.get("sample_weight", 1.0)) for item in items],
        dtype=torch.float32,
    )

    for i, item in enumerate(items):
        length = item["length"]
        mfcc[i, :length] = item["mfcc"]
        ssl[i, :length] = item["ssl"]

    return {
        "ids": [item["id"] for item in items],
        "mfcc": mfcc,
        "ssl": ssl,
        "lengths": lengths,
        "targets": targets,
        "sample_weights": sample_weights,
        "combos": [item["combo"] for item in items],
        "label_sources": [item["label_source"] for item in items],
        "texts": [item["text"] for item in items],
    }


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    val_count = max(1, int(round(len(rows) * val_fraction)))
    val_idx = set(indices[:val_count])

    train_rows = [row for i, row in enumerate(rows) if i not in val_idx]
    val_rows = [row for i, row in enumerate(rows) if i in val_idx]

    return train_rows, val_rows


def predictions_from_logits(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, Any]:
    preds = predictions_from_logits(logits, threshold=threshold)

    exact = (preds == targets).all(dim=1).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()
    tn = ((preds == 0) & (targets == 0)).sum(dim=0).float()

    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)

    label_names = transition_multilabel_label_names()
    per_label = {}

    probs = torch.sigmoid(logits)
    for idx, label in enumerate(label_names):
        per_label[label] = {
            "accuracy": float(accuracy[idx].item()),
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
            "positives": int(targets[:, idx].sum().item()),
            "predicted_positive": int(preds[:, idx].sum().item()),
            "mean_prob": float(probs[:, idx].mean().item()),
        }

    combo_counts: dict[str, int] = {}
    for row in preds.int().tolist():
        if row == [0, 0]:
            combo = "none"
        elif row == [1, 0]:
            combo = "ikhfa"
        elif row == [0, 1]:
            combo = "idgham"
        else:
            combo = "ikhfa+idgham"
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return {
        "exact_match": exact,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "per_label": per_label,
        "predicted_combo_counts": combo_counts,
    }


def evaluate(model: TransitionMultiLabelModule, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, Any]:
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            mfcc = batch["mfcc"].to(device)
            ssl = batch["ssl"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            logits = model(mfcc, ssl, lengths)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_metrics(logits, targets, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_transition_multilabel_extended.jsonl")
    parser.add_argument("--output-checkpoint", default="checkpoints/transition_multilabel_extended_v1.pt")
    parser.add_argument("--output-json", default="data/analysis/transition_multilabel_extended_v1_train.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    rows = load_jsonl(resolve_path(args.manifest))
    train_rows, val_rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)

    train_loader = DataLoader(
        TransitionMultiLabelDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        TransitionMultiLabelDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model = TransitionMultiLabelModule(
        mfcc_dim=39,
        ssl_dim=64,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print("Multi-label transition training")
    print("-------------------------------")
    print(f"samples      : {len(rows)}")
    print(f"train        : {len(train_rows)}")
    print(f"val          : {len(val_rows)}")
    print(f"device       : {device}")
    print(f"threshold    : {args.threshold}")

    best_f1 = -1.0
    best_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            mfcc = batch["mfcc"].to(device)
            ssl = batch["ssl"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)
            sample_weights = batch["sample_weights"].to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mfcc, ssl, lengths)
            raw_loss = criterion(logits, targets)
            loss = (raw_loss * sample_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            threshold=args.threshold,
        )
        mean_loss = sum(losses) / max(1, len(losses))

        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"val_exact={val_metrics['exact_match']:.3f} "
            f"val_f1={val_metrics['macro_f1']:.3f} "
            f"val_precision={val_metrics['macro_precision']:.3f} "
            f"val_recall={val_metrics['macro_recall']:.3f} "
            f"predicted={val_metrics['predicted_combo_counts']}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_metrics = val_metrics

            out_path = resolve_path(args.output_checkpoint)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": transition_multilabel_label_names(),
                    "config": {
                        "mfcc_dim": 39,
                        "ssl_dim": 64,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "threshold": args.threshold,
                    },
                    "best_metrics": best_metrics,
                },
                out_path,
            )
            print(f"saved checkpoint: {out_path}")

    result = {
        "manifest": str(resolve_path(args.manifest)),
        "output_checkpoint": str(resolve_path(args.output_checkpoint)),
        "best_metrics": best_metrics,
    }

    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved summary: {out_json}")


if __name__ == "__main__":
    main()
