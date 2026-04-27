from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import torch
from torch.utils.data import DataLoader, random_split

from tajweed_assessment.data.collate import collate_duration_batch
from tajweed_assessment.data.dataset import ManifestDurationDataset, ToyDurationDataset
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.common.losses import DurationLoss, build_rule_class_weights
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.training.engine import evaluate_duration_epoch
from tajweed_assessment.utils.io import load_checkpoint
from tajweed_assessment.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/duration_module.pt")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--split", choices=["val", "full"], default="val")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    seed_everything(train_cfg["seed"])
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

    if args.split == "val":
        train_size = max(1, int(0.8 * len(full_dataset)))
        val_size = len(full_dataset) - train_size
        _, eval_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(train_cfg["seed"]),
        )
    else:
        eval_ds = full_dataset

    loader = DataLoader(
        eval_ds,
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

    ckpt = PROJECT_ROOT / args.checkpoint
    state = load_checkpoint(ckpt)
    model.load_state_dict(state["model_state_dict"])

    lambda_ctc = model_cfg.get("lambda_ctc", 1.0)
    lambda_rule = model_cfg.get("lambda_rule", 1.0)
    rule_class_weights = build_rule_class_weights(model_cfg.get("rule_class_weights"))
    loss_fn = DurationLoss(
        lambda_ctc=lambda_ctc,
        lambda_rule=lambda_rule,
        rule_class_weights=rule_class_weights,
    )

    metrics = evaluate_duration_epoch(model, loader, loss_fn)
    print(f"Loaded checkpoint: {ckpt}")
    print(f"Evaluation split: {args.split}")
    print(metrics)


if __name__ == "__main__":
    main()

