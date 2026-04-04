from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
import torch
from torch.utils.data import DataLoader, random_split

from tajweed_assessment.data.collate import collate_duration_batch
from tajweed_assessment.data.dataset import ManifestDurationDataset, ToyDurationDataset
from tajweed_assessment.models.common.losses import DurationLoss
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.engine import train_duration_epoch, evaluate_duration_epoch
from tajweed_assessment.utils.seed import seed_everything

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    seed_everything(train_cfg["seed"])
    paths = ProjectPaths(PROJECT_ROOT)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)

    use_toy = bool(data_cfg.get("use_toy_data", True)) or not args.manifest
    if use_toy:
        full_dataset = ToyDurationDataset(n_samples=12, input_dim=model_cfg["input_dim"], seed=train_cfg["seed"])
    else:
        full_dataset = ManifestDurationDataset(args.manifest, sample_rate=data_cfg["sample_rate"], n_mfcc=data_cfg["n_mfcc"])

    train_size = max(1, int(0.75 * len(full_dataset)))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(train_cfg["seed"]))

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_duration_batch)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_duration_batch)

    model = DurationRuleModule(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_phonemes=model_cfg["num_phonemes"],
        num_rules=model_cfg["num_rules"],
    )
    loss_fn = DurationLoss(lambda_ctc=model_cfg["lambda_ctc"], lambda_rule=model_cfg["lambda_rule"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    checkpoint = ModelCheckpoint(paths.checkpoints, filename="duration_module.pt")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_duration_epoch(model, train_loader, optimizer, loss_fn)
        val_metrics = evaluate_duration_epoch(model, val_loader, loss_fn)
        checkpoint.step(val_metrics.loss, {"model_state_dict": model.state_dict(), "config": {"model": model_cfg, "train": train_cfg}})
        print(f"epoch={epoch} train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} val_phoneme_acc={val_metrics.phoneme_accuracy:.3f}")

    print(f"saved best checkpoint to {paths.checkpoints / 'duration_module.pt'}")

if __name__ == "__main__":
    main()
