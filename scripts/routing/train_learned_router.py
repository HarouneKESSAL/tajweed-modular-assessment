from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.models.routing.learned_router import LearnedRoutingModule, routing_label_names


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


class RoutingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "features": torch.tensor(row["features"], dtype=torch.float32),
            "targets": torch.tensor(row["targets"], dtype=torch.float32),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
        }


def evaluate(model: LearnedRoutingModule, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            logits = model(features)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    exact = (preds == targets).all(dim=1).float().mean().item()

    labels = routing_label_names()
    per_label = {}
    f1s = []

    for idx, label in enumerate(labels):
        tp = ((preds[:, idx] == 1) & (targets[:, idx] == 1)).sum().item()
        fp = ((preds[:, idx] == 1) & (targets[:, idx] == 0)).sum().item()
        fn = ((preds[:, idx] == 0) & (targets[:, idx] == 1)).sum().item()
        tn = ((preds[:, idx] == 0) & (targets[:, idx] == 0)).sum().item()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)

        f1s.append(f1)
        per_label[label] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positives": int(targets[:, idx].sum().item()),
            "predicted_positive": int(preds[:, idx].sum().item()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    return {
        "exact_match": exact,
        "macro_f1": sum(f1s) / len(f1s),
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/manifests/learned_routing_dataset_v1.jsonl")
    parser.add_argument("--output-checkpoint", default="checkpoints/learned_router_v1.pt")
    parser.add_argument("--output-json", default="data/analysis/learned_router_v1_train.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(resolve_path(args.dataset))
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]

    if not train_rows or not val_rows:
        raise RuntimeError("dataset must contain train and val splits")

    input_dim = len(train_rows[0]["features"])
    target_names = train_rows[0]["target_names"]
    feature_names = train_rows[0]["feature_names"]

    train_loader = DataLoader(RoutingDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RoutingDataset(val_rows), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = LearnedRoutingModule(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_outputs=len(target_names),
        dropout=args.dropout,
    ).to(device)

    targets = torch.tensor([row["targets"] for row in train_rows], dtype=torch.float32)
    positives = targets.sum(dim=0)
    negatives = targets.size(0) - positives
    pos_weight = negatives / positives.clamp(min=1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Learned router training")
    print("-----------------------")
    print(f"train       : {len(train_rows)}")
    print(f"val         : {len(val_rows)}")
    print(f"input_dim   : {input_dim}")
    print(f"targets     : {target_names}")
    print(f"pos_weight  : {[round(float(x), 3) for x in pos_weight.tolist()]}")

    best_f1 = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            sample_weight = batch["sample_weight"].to(device).unsqueeze(1)

            logits = model(features)
            raw_loss = criterion(logits, targets)
            loss = (raw_loss * sample_weight).mean()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, device)
        mean_loss = sum(losses) / max(1, len(losses))

        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"val_exact={metrics['exact_match']:.3f} "
            f"val_macro_f1={metrics['macro_f1']:.3f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics

            out_path = resolve_path(args.output_checkpoint)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "thresholds": {name: 0.5 for name in target_names},
                    },
                    "best_metrics": best_metrics,
                },
                out_path,
            )
            print(f"saved checkpoint: {out_path}")

    result = {
        "dataset": str(resolve_path(args.dataset)),
        "output_checkpoint": str(resolve_path(args.output_checkpoint)),
        "best_metrics": best_metrics,
    }

    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved summary: {out_json}")


if __name__ == "__main__":
    main()
