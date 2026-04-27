from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from tajweed_assessment.data.collate import collate_sequence_classification_batch
from tajweed_assessment.data.dataset import ManifestTransitionDataset, ToyTransitionDataset
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.engine import train_classifier_epoch, evaluate_classifier_epoch
from tajweed_assessment.utils.io import load_checkpoint, load_json
from tajweed_assessment.utils.seed import seed_everything


def forward_fn(model, batch, device):
    return model(
        batch["mfcc"].to(device),
        batch["ssl"].to(device),
        batch["lengths"].to(device),
    )


def transition_label_name(entry) -> str:
    raw_label = (entry.canonical_rules or ["none"])[0]
    label_name = normalize_rule_name(raw_label)
    return label_name if label_name in TRANSITION_RULES else "none"


def split_transition_indices(entries, val_fraction: float = 0.2, seed: int = 7):
    groups = {}
    for idx, entry in enumerate(entries):
        group_key = getattr(entry, "reciter_id", None) or "Unknown"
        groups.setdefault(group_key, []).append(idx)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    def labels_for_indices(indices):
        labels = set()
        for idx in indices:
            label = transition_label_name(entries[idx])
            if label != "none":
                labels.add(label)
        return labels

    target_val_rows = max(1, int(len(entries) * val_fraction))
    val_groups = set()
    val_count = 0
    covered_labels = set()
    required_labels = {
        transition_label_name(entry)
        for entry in entries
        if transition_label_name(entry) != "none"
    }

    informative_groups = []
    background_groups = []
    for key, indices in group_items:
        item = (key, indices, labels_for_indices(indices))
        if item[2]:
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
        split = max(1, int(0.8 * len(entries)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(entries)))

    return train_idx, val_idx


def label_counts(entries, indices=None):
    selected = entries if indices is None else [entries[i] for i in indices]
    return Counter(transition_label_name(entry) for entry in selected)


def compute_class_weights(entries, indices):
    counts = label_counts(entries, indices)
    values = torch.tensor(
        [float(counts.get(label, 0)) for label in TRANSITION_RULES],
        dtype=torch.float32,
    )
    total = float(values.sum().item())
    raw_weights = total / values.clamp(min=1.0)
    weights = raw_weights.sqrt()
    weights = weights.clamp(min=0.5, max=1.75)
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights


@torch.no_grad()
def evaluate_transition_diagnostics(model, loader, device: str):
    model.eval()
    num_rules = len(TRANSITION_RULES)
    confusion = torch.zeros(num_rules, num_rules, dtype=torch.long)

    for batch in loader:
        logits = forward_fn(model, batch, device).cpu()
        preds = logits.argmax(dim=-1)
        targets = batch["label"].cpu()
        for target, pred in zip(targets.tolist(), preds.tolist()):
            confusion[target, pred] += 1

    per_class_acc = {}
    macro_acc_values = []
    for idx, label in enumerate(TRANSITION_RULES):
        support = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        acc = (correct / support) if support > 0 else 0.0
        per_class_acc[label] = {
            "support": support,
            "correct": correct,
            "accuracy": acc,
        }
        if support > 0:
            macro_acc_values.append(acc)

    macro_acc = sum(macro_acc_values) / len(macro_acc_values) if macro_acc_values else 0.0
    return {
        "confusion": confusion,
        "per_class_acc": per_class_acc,
        "macro_acc": macro_acc,
    }


def print_label_counts(title: str, counts: Counter):
    rendered = " ".join(f"{label}={counts.get(label, 0)}" for label in TRANSITION_RULES)
    print(f"{title}: {rendered}")


def print_confusion(confusion: torch.Tensor):
    print("Validation confusion matrix (rows=gold, cols=pred):")
    header = "gold\\pred".ljust(12) + " ".join(label.rjust(8) for label in TRANSITION_RULES)
    print(header)
    for idx, label in enumerate(TRANSITION_RULES):
        row = " ".join(str(int(v)).rjust(8) for v in confusion[idx].tolist())
        print(label.ljust(12) + row)


def transition_entry_id(entry) -> str:
    return str(getattr(entry, "sample_id", None) or getattr(entry, "audio_path", None) or "sample")


def load_hardcase_weight_map(path: str | Path) -> dict[str, float]:
    data = load_json(path)
    if isinstance(data, dict):
        if isinstance(data.get("weights"), dict):
            return {str(key): float(value) for key, value in data["weights"].items()}
        if isinstance(data.get("hardcases"), list):
            return {
                str(item.get("sample_id") or item.get("id") or item.get("audio_path")): float(item.get("weight", 1.0))
                for item in data["hardcases"]
                if isinstance(item, dict) and (item.get("sample_id") or item.get("id") or item.get("audio_path"))
            }
    if isinstance(data, list):
        return {
            str(item.get("sample_id") or item.get("id") or item.get("audio_path")): float(item.get("weight", 1.0))
            for item in data
            if isinstance(item, dict) and (item.get("sample_id") or item.get("id") or item.get("audio_path"))
        }
    return {}


def build_transition_sample_weights(
    entries,
    indices,
    hardcase_weight_map: dict[str, float] | None = None,
) -> list[float]:
    hardcase_weight_map = hardcase_weight_map or {}
    weights: list[float] = []
    for idx in indices:
        entry = entries[idx]
        weight = float(hardcase_weight_map.get(transition_entry_id(entry), 1.0))
        weights.append(max(1.0, weight))
    return weights

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--hardcase-json", default="")
    parser.add_argument("--checkpoint-name", default="transition_module.pt")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
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
        full_dataset = ToyTransitionDataset(n_samples=64, seed=train_cfg["seed"])
        train_ds, val_ds = random_split(
            full_dataset,
            [max(1, int(0.8 * len(full_dataset))), len(full_dataset) - max(1, int(0.8 * len(full_dataset)))],
            generator=torch.Generator().manual_seed(train_cfg["seed"]),
        )
    else:
        full_dataset = ManifestTransitionDataset(
            args.manifest,
            sample_rate=data_cfg["sample_rate"],
            n_mfcc=data_cfg["n_mfcc"],
            speed_config=speed_config,
        )
        train_idx, val_idx = split_transition_indices(full_dataset.entries, val_fraction=0.2, seed=train_cfg["seed"])
        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        val_ds = torch.utils.data.Subset(full_dataset, val_idx)

        print_label_counts("Full label counts", label_counts(full_dataset.entries))
        print_label_counts("Train label counts", label_counts(full_dataset.entries, train_idx))
        print_label_counts("Val label counts", label_counts(full_dataset.entries, val_idx))
        print(f"Grouped split by reciter: train_rows={len(train_idx)} val_rows={len(val_idx)}")
        class_weights = compute_class_weights(full_dataset.entries, train_idx)
        print(
            "Class weights: "
            + " ".join(
                f"{label}={weight:.3f}"
                for label, weight in zip(TRANSITION_RULES, class_weights.tolist())
            )
        )
    if use_toy:
        class_weights = torch.ones(len(TRANSITION_RULES), dtype=torch.float32)

    train_sampler = None
    if not use_toy and args.hardcase_json:
        hardcase_weight_map = load_hardcase_weight_map(PROJECT_ROOT / args.hardcase_json)
        sample_weights = build_transition_sample_weights(full_dataset.entries, train_idx, hardcase_weight_map)
        boosted = sum(1 for weight in sample_weights if weight > 1.0)
        if boosted > 0:
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            mean_weight = sum(sample_weights) / max(len(sample_weights), 1)
            max_weight = max(sample_weights) if sample_weights else 1.0
            print(
                f"Loaded hardcase weights: boosted_train_rows={boosted}/{len(sample_weights)} "
                f"mean_weight={mean_weight:.2f} max_weight={max_weight:.2f}"
            )
        else:
            print("Loaded hardcase weights, but no train rows received weight > 1.0.")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_sequence_classification_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_sequence_classification_batch,
    )

    model = TransitionRuleModule(
        mfcc_dim=model_cfg["mfcc_dim"],
        ssl_dim=model_cfg["ssl_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_rules=model_cfg["num_rules"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    checkpoint = ModelCheckpoint(paths.checkpoints, filename=args.checkpoint_name)

    best_score = float("-inf")
    device = train_cfg.get("device", "cpu")
    model = model.to(device)
    checkpoint_path = paths.checkpoints / args.checkpoint_name
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_classifier_epoch(model, train_loader, optimizer, forward_fn, device=device, loss_fn=loss_fn)
        val_metrics = evaluate_classifier_epoch(model, val_loader, forward_fn, device=device, loss_fn=loss_fn)
        diagnostics = evaluate_transition_diagnostics(model, val_loader, device=device)
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
    final_diagnostics = evaluate_transition_diagnostics(model, val_loader, device=device)
    print(f"best checkpoint: {checkpoint_path}")
    print_confusion(final_diagnostics["confusion"])
    for label in TRANSITION_RULES:
        stats = final_diagnostics["per_class_acc"][label]
        print(
            f"{label}: support={stats['support']} "
            f"correct={stats['correct']} "
            f"acc={stats['accuracy']:.3f}"
        )

if __name__ == "__main__":
    main()

