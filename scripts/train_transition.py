from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, random_split

from tajweed_assessment.data.collate import collate_sequence_classification_batch
from tajweed_assessment.data.dataset import ToyTransitionDataset
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.training.engine import train_classifier_epoch, evaluate_classifier_epoch
from tajweed_assessment.utils.seed import seed_everything

def forward_fn(model, batch, device):
    return model(batch["mfcc"], batch["ssl"], batch["lengths"])

def main() -> None:
    seed_everything(7)
    dataset = ToyTransitionDataset(n_samples=12, seed=7)
    train_ds, val_ds = random_split(dataset, [9, 3], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, collate_fn=collate_sequence_classification_batch)
    val_loader = DataLoader(val_ds, batch_size=3, shuffle=False, collate_fn=collate_sequence_classification_batch)

    model = TransitionRuleModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_metrics = train_classifier_epoch(model, train_loader, optimizer, forward_fn)
    val_metrics = evaluate_classifier_epoch(model, val_loader, forward_fn)
    print(f"train_loss={train_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f}")

if __name__ == "__main__":
    main()
