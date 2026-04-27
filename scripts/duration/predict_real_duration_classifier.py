from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tajweed_assessment.data.real_duration_audio_dataset import (
    RealDurationAudioDataset,
    load_jsonl,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.utils.seed import seed_everything


class BiLSTMMultiLabelClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_t = out.size(1)
        mask = torch.arange(max_t, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        summed = (out * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / denom

        return self.classifier(self.dropout(pooled))


def split_indices_by_reciter(rows, val_fraction: float = 0.2, seed: int = 7):
    by_group = {}
    for i, row in enumerate(rows):
        group = row.get("reciter_id") or "Unknown"
        by_group.setdefault(group, []).append(i)

    groups = list(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_rows = len(rows)
    target_val_rows = int(total_rows * val_fraction)

    val_groups = set()
    val_count = 0

    for g in groups:
        if val_count >= target_val_rows:
            break
        val_groups.add(g)
        val_count += len(by_group[g])

    train_idx = []
    val_idx = []

    for g, indices in by_group.items():
        if g in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if len(val_idx) == 0 or len(train_idx) == 0:
        split = max(1, int(0.8 * total_rows))
        train_idx = list(range(split))
        val_idx = list(range(split, total_rows))

    return train_idx, val_idx


def load_thresholds(path: Path, label_vocab):
    if not path.exists():
        return {label: 0.5 for label in label_vocab}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "thresholds" in data:
        raw = data["thresholds"]
    else:
        raw = data

    thresholds = {}
    for label in label_vocab:
        thresholds[label] = float(raw.get(label, 0.5))
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_subset.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/real_duration_classifier.pt")
    parser.add_argument("--thresholds", default="checkpoints/real_duration_thresholds.json")
    parser.add_argument("--sample-index", type=int, default=0, help="Index inside the selected split")
    parser.add_argument("--split", choices=["full", "train", "val"], default="val")
    parser.add_argument("--device", default="")
    parser.add_argument("--feature-cache-dir", default="data/interim/retasy_mfcc_cache")
    parser.add_argument("--normalize-speed", action="store_true")
    parser.add_argument("--target-speech-rate", type=float, default=12.0)
    parser.add_argument("--max-speed-factor", type=float, default=1.35)
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    manifest_path = PROJECT_ROOT / args.manifest
    thresholds_path = PROJECT_ROOT / args.thresholds

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    label_vocab = checkpoint["label_vocab"]

    seed = int(config.get("seed", 7))
    seed_everything(seed)

    rows = load_jsonl(manifest_path)
    train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=seed)

    if args.split == "train":
        indices = train_idx
    elif args.split == "val":
        indices = val_idx
    else:
        indices = None

    speed_config = SpeedNormalizationConfig(
        enabled=bool(args.normalize_speed),
        target_speech_rate=float(args.target_speech_rate),
        max_speed_factor=float(args.max_speed_factor),
    )

    dataset = RealDurationAudioDataset(
        manifest_path,
        indices=indices,
        rule_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        speed_config=speed_config,
    )

    if len(dataset) == 0:
        raise RuntimeError("Selected split is empty.")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index out of range: 0 <= idx < {len(dataset)}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMMultiLabelClassifier(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = dataset[args.sample_index]
    x = sample["x"].unsqueeze(0).to(device)
    lengths = torch.tensor([sample["input_length"]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.sigmoid(logits)[0].cpu().tolist()

    thresholds = load_thresholds(thresholds_path, label_vocab)

    scored = []
    gold = set(sample["duration_rules"])
    pred = []

    for label, prob in zip(label_vocab, probs):
        thr = thresholds[label]
        is_pred = prob >= thr
        if is_pred:
            pred.append(label)
        scored.append(
            {
                "label": label,
                "probability": float(prob),
                "threshold": float(thr),
                "predicted": bool(is_pred),
                "gold": label in gold,
            }
        )

    scored = sorted(scored, key=lambda x: x["probability"], reverse=True)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Thresholds : {thresholds_path if thresholds_path.exists() else 'default 0.5'}")
    print(f"Split      : {args.split}")
    print(f"Sample idx : {args.sample_index}")
    print()
    print(f"ID         : {sample['id']}")
    print(f"Surah      : {sample['surah_name']}")
    print(f"Verse key  : {sample['quranjson_verse_key']}")
    print(f"Text       : {sample['aya_text']}")
    print(f"Gold       : {sample['duration_rules']}")
    print(f"Predicted  : {pred}")
    print()

    for item in scored:
        marker = "✓" if item["predicted"] else " "
        gold_marker = "*" if item["gold"] else " "
        print(
            f"[{marker}][{gold_marker}] "
            f"{item['label']:15s} "
            f"prob={item['probability']:.3f} "
            f"thr={item['threshold']:.2f}"
        )


if __name__ == "__main__":
    main()

