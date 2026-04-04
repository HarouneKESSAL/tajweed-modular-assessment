from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, random_split

from tajweed_assessment.data.collate import collate_duration_batch
from tajweed_assessment.data.dataset import ToyDurationDataset
from tajweed_assessment.models.common.losses import DurationLoss
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.training.engine import train_duration_epoch, evaluate_duration_epoch
from tajweed_assessment.utils.seed import seed_everything

def main() -> None:
    seed_everything(7)
    dataset = ToyDurationDataset(n_samples=8, input_dim=39, seed=7)
    train_ds, val_ds = random_split(dataset, [6, 2], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_duration_batch)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_duration_batch)

    model = DurationRuleModule(input_dim=39, hidden_dim=32, num_layers=1, dropout=0.1, num_phonemes=11, num_rules=6)
    loss_fn = DurationLoss(lambda_ctc=1.0, lambda_rule=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_metrics = train_duration_epoch(model, train_loader, optimizer, loss_fn)
    val_metrics = evaluate_duration_epoch(model, val_loader, loss_fn)
    print(f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f}")

if __name__ == "__main__":
    main()
