from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
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
    rows = []
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
    return transform(waveform).squeeze(0).transpose(0, 1).contiguous()


class TransitionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], ssl_dim: int = 64) -> None:
        self.rows = rows
        self.ssl_dim = ssl_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        mfcc = extract_mfcc(resolve_path(row["audio_path"]))
        ssl = torch.zeros((mfcc.size(0), self.ssl_dim), dtype=torch.float32)
        target = torch.tensor(row["transition_multihot"], dtype=torch.float32)

        return {
            "mfcc": mfcc,
            "ssl": ssl,
            "length": int(mfcc.size(0)),
            "target": target,
            "combo": row.get("transition_combo", ""),
            "label_source": row.get("label_source", ""),
        }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(items)
    max_len = max(item["length"] for item in items)
    mfcc_dim = items[0]["mfcc"].size(1)
    ssl_dim = items[0]["ssl"].size(1)

    mfcc = torch.zeros((batch, max_len, mfcc_dim), dtype=torch.float32)
    ssl = torch.zeros((batch, max_len, ssl_dim), dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.long)
    targets = torch.stack([item["target"] for item in items])

    for i, item in enumerate(items):
        length = item["length"]
        mfcc[i, :length] = item["mfcc"]
        ssl[i, :length] = item["ssl"]

    return {
        "mfcc": mfcc,
        "ssl": ssl,
        "lengths": lengths,
        "targets": targets,
    }


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(rows) * val_fraction)))
    val_idx = set(indices[:val_count])
    return [row for i, row in enumerate(rows) if i in val_idx]


def metrics_for_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, Any]:
    preds = (probs >= thresholds.unsqueeze(0)).float()

    exact = (preds == targets).all(dim=1).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()
    tn = ((preds == 0) & (targets == 0)).sum(dim=0).float()

    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)

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

    label_names = transition_multilabel_label_names()
    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = {
            "accuracy": float(accuracy[i].item()),
            "precision": float(precision[i].item()),
            "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
            "positives": int(targets[:, i].sum().item()),
            "predicted_positive": int(preds[:, i].sum().item()),
            "threshold": float(thresholds[i].item()),
        }

    return {
        "exact_match": float(exact),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "per_label": per_label,
        "predicted_combo_counts": combo_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_transition_multilabel_extended.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/transition_multilabel_extended_v1.pt")
    parser.add_argument("--output-json", default="data/analysis/transition_multilabel_extended_v1_thresholds.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.manifest))
    val_rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)

    ckpt = torch.load(resolve_path(args.checkpoint), map_location="cpu")
    cfg = ckpt["config"]

    model = TransitionMultiLabelModule(
        mfcc_dim=int(cfg.get("mfcc_dim", 39)),
        ssl_dim=int(cfg.get("ssl_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_labels=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    loader = DataLoader(
        TransitionDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["mfcc"].to(device),
                batch["ssl"].to(device),
                batch["lengths"].to(device),
            )
            all_logits.append(logits.cpu())
            all_targets.append(batch["targets"])

    probs = torch.sigmoid(torch.cat(all_logits, dim=0))
    targets = torch.cat(all_targets, dim=0)

    best = None
    all_results = []

    grid = [x / 100.0 for x in range(20, 81, 5)]

    for ikhfa_t in grid:
        for idgham_t in grid:
            thresholds = torch.tensor([ikhfa_t, idgham_t], dtype=torch.float32)
            metrics = metrics_for_thresholds(probs, targets, thresholds)
            record = {
                "thresholds": {
                    "ikhfa": ikhfa_t,
                    "idgham": idgham_t,
                },
                "metrics": metrics,
            }
            all_results.append(record)

            key = (
                metrics["macro_f1"],
                metrics["exact_match"],
                metrics["per_label"]["idgham"]["f1"],
            )
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "record": record,
                }

    result = {
        "manifest": str(resolve_path(args.manifest)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "samples": len(val_rows),
        "best": best["record"],
        "top_10": sorted(
            all_results,
            key=lambda r: (
                r["metrics"]["macro_f1"],
                r["metrics"]["exact_match"],
                r["metrics"]["per_label"]["idgham"]["f1"],
            ),
            reverse=True,
        )[:10],
    }

    out = resolve_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    best_record = result["best"]
    print("Threshold tuning complete")
    print("-------------------------")
    print(f"samples: {len(val_rows)}")
    print(f"best thresholds: {best_record['thresholds']}")
    print(f"best exact: {best_record['metrics']['exact_match']:.3f}")
    print(f"best macro_f1: {best_record['metrics']['macro_f1']:.3f}")
    print(f"predicted: {best_record['metrics']['predicted_combo_counts']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
