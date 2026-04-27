from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from tajweed_assessment.data.collate import collate_duration_batch
from tajweed_assessment.data.dataset import ManifestDurationDataset, ToyDurationDataset
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.common.losses import DurationLoss, build_rule_class_weights
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.engine import train_duration_epoch, evaluate_duration_epoch
from tajweed_assessment.utils.io import load_json
from tajweed_assessment.utils.seed import seed_everything


def duration_entry_id(entry) -> str:
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


def build_duration_sample_weights(entries, indices, hardcase_weight_map: dict[str, float] | None = None) -> list[float]:
    hardcase_weight_map = hardcase_weight_map or {}
    weights: list[float] = []
    for idx in indices:
        entry = entries[idx]
        weight = float(hardcase_weight_map.get(duration_entry_id(entry), 1.0))
        weights.append(max(1.0, weight))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--hardcase-json", default="")
    parser.add_argument("--checkpoint-name", default="duration_module.pt")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
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
        full_dataset = ToyDurationDataset(
            n_samples=256,
            input_dim=model_cfg["input_dim"],
            seed=train_cfg["seed"],
        )
    else:
        full_dataset = ManifestDurationDataset(
            args.manifest,
            sample_rate=data_cfg["sample_rate"],
            n_mfcc=data_cfg["n_mfcc"],
            speed_config=speed_config,
        )

    train_size = max(1, int(0.8 * len(full_dataset)))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )

    train_sampler = None
    if not use_toy and args.hardcase_json:
        train_indices = list(train_ds.indices)
        hardcase_weight_map = load_hardcase_weight_map(PROJECT_ROOT / args.hardcase_json)
        sample_weights = build_duration_sample_weights(full_dataset.entries, train_indices, hardcase_weight_map)
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
        collate_fn=collate_duration_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_duration_batch,
    )

    model = DurationRuleModule(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_phonemes=model_cfg["num_phonemes"],
        num_rules=model_cfg["num_rules"],
    )

    lambda_ctc = model_cfg.get("lambda_ctc", 1.0)
    lambda_rule = model_cfg.get("lambda_rule", 1.0)
    phoneme_score_weight = float(model_cfg.get("phoneme_score_weight", 0.7))
    rule_score_weight = float(model_cfg.get("rule_score_weight", 0.3))
    rule_class_weights = build_rule_class_weights(model_cfg.get("rule_class_weights"))

    loss_fn = DurationLoss(
        lambda_ctc=lambda_ctc,
        lambda_rule=lambda_rule,
        rule_class_weights=rule_class_weights,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint = ModelCheckpoint(paths.checkpoints, filename=args.checkpoint_name)

    best_score = float("-inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_duration_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = evaluate_duration_epoch(model, val_loader, loss_fn)

        score = (
            phoneme_score_weight * (val_metrics.phoneme_accuracy or 0.0)
            + rule_score_weight * (val_metrics.rule_accuracy or 0.0)
        )

        if score > best_score:
            best_score = score
            checkpoint.step(
                -score,
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"model": model_cfg, "train": train_cfg},
                },
            )
            print(f"saved best checkpoint to {paths.checkpoints / args.checkpoint_name}")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_ctc={train_metrics.ctc_loss:.4f} "
            f"train_rule={train_metrics.rule_loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_ctc={val_metrics.ctc_loss:.4f} "
            f"val_rule={val_metrics.rule_loss:.4f} "
            f"val_phoneme_acc={val_metrics.phoneme_accuracy:.3f} "
            f"val_rule_acc={val_metrics.rule_accuracy:.3f} "
            f"val_score={score:.3f}"
        )

    print(f"best checkpoint: {paths.checkpoints / args.checkpoint_name}")


if __name__ == "__main__":
    main()

