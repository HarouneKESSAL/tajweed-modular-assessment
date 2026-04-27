from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from tajweed_assessment.data.collate import collate_sequence_classification_batch
from tajweed_assessment.data.dataset import ToyBurstDataset
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.engine import train_classifier_epoch, evaluate_classifier_epoch
from tajweed_assessment.utils.io import load_checkpoint
from tajweed_assessment.utils.seed import seed_everything


class ManifestBurstDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        speed_config: SpeedNormalizationConfig | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.speed_config = speed_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = extract_mfcc_features(
            row["audio_path"],
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            speed_config=self.speed_config,
        )
        return {
            "x": x,
            "label": torch.tensor(int(row["burst_label"]), dtype=torch.long),
        }


def forward_fn(model, batch, device):
    return model(batch["x"].to(device))


def split_burst_indices(rows, val_fraction: float = 0.2, seed: int = 7):
    groups = {}
    for idx, row in enumerate(rows):
        group_key = row.get("reciter_id") or "Unknown"
        groups.setdefault(group_key, []).append(idx)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    def labels_for_indices(indices):
        return {int(rows[i]["burst_label"]) for i in indices}

    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    covered_labels = set()
    required_labels = {int(row["burst_label"]) for row in rows}

    informative_groups = []
    background_groups = []
    for key, indices in group_items:
        labels = labels_for_indices(indices)
        item = (key, indices, labels)
        if 1 in labels:
            informative_groups.append(item)
        else:
            background_groups.append(item)

    informative_groups.sort(key=lambda item: (len(item[2]), len(item[1])), reverse=True)

    for key, indices, labels in informative_groups:
        if covered_labels >= required_labels:
            break
        if labels - covered_labels:
            val_groups.add(key)
            val_count += len(indices)
            covered_labels.update(labels)

    remaining_groups = [(k, idxs) for k, idxs, _ in informative_groups if k not in val_groups]
    remaining_groups.extend((k, idxs) for k, idxs, _ in background_groups)
    rng.shuffle(remaining_groups)

    for key, indices in remaining_groups:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(indices)

    train_idx = []
    val_idx = []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if len(train_idx) == 0 or len(val_idx) == 0:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))

    return train_idx, val_idx


def label_counts(rows, indices=None):
    selected = rows if indices is None else [rows[i] for i in indices]
    return Counter(int(row["burst_label"]) for row in selected)


@torch.no_grad()
def evaluate_burst_diagnostics(model, loader, device: str):
    model.eval()
    confusion = torch.zeros(2, 2, dtype=torch.long)

    for batch in loader:
        logits = forward_fn(model, batch, device).cpu()
        preds = logits.argmax(dim=-1)
        targets = batch["label"].cpu()
        for target, pred in zip(targets.tolist(), preds.tolist()):
            confusion[target, pred] += 1

    per_class_acc = {}
    macro_acc_values = []
    label_names = ["none", "qalqalah"]
    for idx, label in enumerate(label_names):
        support = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        acc = (correct / support) if support > 0 else 0.0
        per_class_acc[label] = {"support": support, "correct": correct, "accuracy": acc}
        if support > 0:
            macro_acc_values.append(acc)

    macro_acc = sum(macro_acc_values) / len(macro_acc_values) if macro_acc_values else 0.0
    return {"confusion": confusion, "per_class_acc": per_class_acc, "macro_acc": macro_acc}


def print_label_counts(title: str, counts: Counter):
    print(f"{title}: none={counts.get(0, 0)} qalqalah={counts.get(1, 0)}")


def print_confusion(confusion: torch.Tensor):
    print("Validation confusion matrix (rows=gold, cols=pred):")
    print("gold\\pred".ljust(12) + "    none qalqalah")
    print("none".ljust(12) + f"{int(confusion[0,0]):8d}{int(confusion[0,1]):9d}")
    print("qalqalah".ljust(12) + f"{int(confusion[1,0]):8d}{int(confusion[1,1]):9d}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_burst.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    seed_everything(train_cfg["seed"])
    paths = ProjectPaths(PROJECT_ROOT)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    use_toy = not bool(args.manifest) and bool(data_cfg.get("use_toy_data", True))
    if use_toy:
        full_dataset = ToyBurstDataset(n_samples=64, seed=train_cfg["seed"])
        train_len = max(1, int(0.8 * len(full_dataset)))
        val_len = len(full_dataset) - train_len
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(train_cfg["seed"]),
        )
    else:
        full_dataset = ManifestBurstDataset(
            args.manifest,
            sample_rate=data_cfg["sample_rate"],
            n_mfcc=data_cfg["n_mfcc"],
            speed_config=speed_config,
        )
        train_idx, val_idx = split_burst_indices(full_dataset.rows, val_fraction=0.2, seed=train_cfg["seed"])
        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        val_ds = torch.utils.data.Subset(full_dataset, val_idx)

        print_label_counts("Full label counts", label_counts(full_dataset.rows))
        print_label_counts("Train label counts", label_counts(full_dataset.rows, train_idx))
        print_label_counts("Val label counts", label_counts(full_dataset.rows, val_idx))
        print(f"Grouped split by reciter: train_rows={len(train_idx)} val_rows={len(val_idx)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_sequence_classification_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_sequence_classification_batch,
    )

    channels = tuple(model_cfg["channels"])
    model = QalqalahCNN(
        input_dim=model_cfg["input_dim"],
        channels=channels,
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    checkpoint = ModelCheckpoint(paths.checkpoints, filename="burst_module.pt")

    best_score = float("-inf")
    device = train_cfg.get("device", "cpu")
    model = model.to(device)
    checkpoint_path = paths.checkpoints / "burst_module.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_classifier_epoch(model, train_loader, optimizer, forward_fn, device=device)
        val_metrics = evaluate_classifier_epoch(model, val_loader, forward_fn, device=device)
        diagnostics = evaluate_burst_diagnostics(model, val_loader, device=device)
        score = diagnostics["macro_acc"]

        if score > best_score:
            best_score = score
            checkpoint.step(
                -score,
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"model": model_cfg, "train": train_cfg},
                },
            )
            print(f"saved best checkpoint to {checkpoint_path}")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.3f} "
            f"val_macro_acc={diagnostics['macro_acc']:.3f}"
        )

    state = load_checkpoint(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    final_diagnostics = evaluate_burst_diagnostics(model, val_loader, device=device)
    print(f"best checkpoint: {checkpoint_path}")
    print_confusion(final_diagnostics["confusion"])
    for label in ["none", "qalqalah"]:
        stats = final_diagnostics["per_class_acc"][label]
        print(f"{label}: support={stats['support']} correct={stats['correct']} acc={stats['accuracy']:.3f}")

if __name__ == "__main__":
    main()

