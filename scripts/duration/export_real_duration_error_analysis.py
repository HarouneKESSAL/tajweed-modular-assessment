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
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from tajweed_assessment.data.real_duration_audio_dataset import (
    RealDurationAudioDataset,
    collate_real_duration_audio_batch,
    load_jsonl,
)
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

    raw = data["thresholds"] if isinstance(data, dict) and "thresholds" in data else data
    return {label: float(raw.get(label, 0.5)) for label in label_vocab}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_subset.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/real_duration_classifier.pt")
    parser.add_argument("--thresholds", default="checkpoints/real_duration_thresholds.json")
    parser.add_argument("--output", default="data/analysis/real_duration_error_analysis_val.jsonl")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--feature-cache-dir", default="data/interim/retasy_mfcc_cache")
    parser.add_argument("--only-errors", action="store_true")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    manifest_path = PROJECT_ROOT / args.manifest
    thresholds_path = PROJECT_ROOT / args.thresholds
    output_path = PROJECT_ROOT / args.output

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

    dataset = RealDurationAudioDataset(
        manifest_path,
        indices=indices,
        rule_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_real_duration_audio_batch,
    )

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

    thresholds = load_thresholds(thresholds_path, label_vocab)

    rows_out = []
    exact_counter = 0
    fp_counter = Counter()
    fn_counter = Counter()

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["input_lengths"].to(device)

            logits = model(x, lengths)
            probs = torch.sigmoid(logits).cpu()

            for i in range(probs.size(0)):
                prob_vec = probs[i].tolist()
                gold = list(batch["duration_rules"][i])

                predicted = []
                for label, prob in zip(label_vocab, prob_vec):
                    if prob >= thresholds[label]:
                        predicted.append(label)

                gold_set = set(gold)
                pred_set = set(predicted)

                false_positives = sorted(pred_set - gold_set)
                false_negatives = sorted(gold_set - pred_set)
                exact_match = pred_set == gold_set

                if exact_match:
                    exact_counter += 1

                for label in false_positives:
                    fp_counter[label] += 1
                for label in false_negatives:
                    fn_counter[label] += 1

                row = {
                    "id": batch["id"][i],
                    "hf_index": batch["hf_index"][i],
                    "surah_name": batch["surah_name"][i],
                    "aya_text": batch["aya_text"][i],
                    "gold_labels": gold,
                    "predicted_labels": predicted,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "exact_match": exact_match,
                    "probabilities": {
                        label: float(prob)
                        for label, prob in zip(label_vocab, prob_vec)
                    },
                    "thresholds": {
                        label: float(thresholds[label])
                        for label in label_vocab
                    },
                }

                if args.only_errors and exact_match:
                    continue

                rows_out.append(row)

    write_jsonl(output_path, rows_out)

    total = len(dataset)
    exported = len(rows_out)

    print(f"Checkpoint      : {ckpt_path}")
    print(f"Thresholds      : {thresholds_path if thresholds_path.exists() else 'default 0.5'}")
    print(f"Split           : {args.split}")
    print(f"Samples in split: {total}")
    print(f"Exported rows   : {exported}")
    print(f"Output          : {output_path}")
    print()
    print(f"Exact matches   : {exact_counter}/{total} ({(exact_counter / max(total, 1)):.3f})")
    print("Top false positives:")
    for label, count in fp_counter.most_common():
        print(f"  {label:15s} {count}")
    print("Top false negatives:")
    for label, count in fn_counter.most_common():
        print(f"  {label:15s} {count}")


if __name__ == "__main__":
    main()
