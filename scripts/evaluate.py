from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import argparse
from torch.utils.data import DataLoader

from tajweed_assessment.data.collate import collate_duration_batch
from tajweed_assessment.data.dataset import ToyDurationDataset
from tajweed_assessment.models.common.losses import DurationLoss
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.training.engine import evaluate_duration_epoch
from tajweed_assessment.utils.io import load_checkpoint

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/duration_module.pt")
    args = parser.parse_args()

    dataset = ToyDurationDataset(n_samples=8, seed=11)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_duration_batch)

    model = DurationRuleModule()
    ckpt = PROJECT_ROOT / args.checkpoint
    if ckpt.exists():
        state = load_checkpoint(ckpt)
        model.load_state_dict(state["model_state_dict"])
    metrics = evaluate_duration_epoch(model, loader, DurationLoss(lambda_ctc=1.0, lambda_rule=0.7))
    print(metrics)

if __name__ == "__main__":
    main()
