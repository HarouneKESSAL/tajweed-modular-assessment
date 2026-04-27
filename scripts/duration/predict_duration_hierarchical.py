from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tajweed_assessment.data.real_duration_audio_dataset import load_jsonl
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


def load_thresholds(path: Path, label_vocab, default: float = 0.5):
    if not path.exists():
        return {label: default for label in label_vocab}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data["thresholds"] if isinstance(data, dict) and "thresholds" in data else data
    return {label: float(raw.get(label, default)) for label in label_vocab}


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    label_vocab = checkpoint["label_vocab"]

    model = BiLSTMMultiLabelClassifier(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_vocab, config


def load_local_audio(audio_path: str):
    data, sr = sf.read(audio_path, always_2d=True)
    waveform = torch.tensor(data.T, dtype=torch.float32)  # [C, T]
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, int(sr)


def maybe_resample(waveform: torch.Tensor, sr: int, target_sr: int):
    if sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    return resampler(waveform)


def build_feature_extractor(sample_rate: int, n_mfcc: int):
    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
        },
    )


def extract_features(waveform: torch.Tensor, mfcc_extractor):
    mfcc = mfcc_extractor(waveform)               # [1, 13, frames]
    mfcc = mfcc.squeeze(0).transpose(0, 1)       # [frames, 13]

    delta = torchaudio.functional.compute_deltas(mfcc.transpose(0, 1)).transpose(0, 1)
    delta2 = torchaudio.functional.compute_deltas(delta.transpose(0, 1)).transpose(0, 1)

    return torch.cat([mfcc, delta, delta2], dim=-1)  # [frames, 39]


def predict_probs(model, x: torch.Tensor, device: str):
    x = x.unsqueeze(0).to(device)
    lengths = torch.tensor([x.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.sigmoid(logits)[0].cpu().tolist()

    return probs


def select_row(rows, split: str, sample_index: int, seed: int):
    train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=seed)

    if split == "train":
        indices = train_idx
    elif split == "val":
        indices = val_idx
    else:
        indices = list(range(len(rows)))

    if sample_index < 0 or sample_index >= len(indices):
        raise IndexError(f"sample-index out of range for split '{split}': 0 <= idx < {len(indices)}")

    return rows[indices[sample_index]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_subset.jsonl")
    parser.add_argument("--coarse-checkpoint", default="checkpoints/real_duration_classifier_coarse.pt")
    parser.add_argument("--subtype-checkpoint", default="checkpoints/madd_subtype_classifier.pt")

    parser.add_argument("--coarse-thresholds", default="")
    parser.add_argument("--subtype-thresholds", default="")

    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--override-has-madd-threshold", type=float, default=None)
    parser.add_argument("--override-ghunnah-threshold", type=float, default=None)
    parser.add_argument("--override-subtype-threshold", type=float, default=None)

    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    manifest_path = PROJECT_ROOT / args.manifest
    rows = load_jsonl(manifest_path)
    row = select_row(rows, args.split, args.sample_index, args.seed)

    coarse_ckpt = PROJECT_ROOT / args.coarse_checkpoint
    subtype_ckpt = PROJECT_ROOT / args.subtype_checkpoint

    coarse_model, coarse_labels, coarse_cfg = load_model(coarse_ckpt, device)
    subtype_model, subtype_labels, subtype_cfg = load_model(subtype_ckpt, device)

    coarse_thr_path = PROJECT_ROOT / args.coarse_thresholds if args.coarse_thresholds else Path("__missing__.json")
    subtype_thr_path = PROJECT_ROOT / args.subtype_thresholds if args.subtype_thresholds else Path("__missing__.json")

    coarse_thresholds = load_thresholds(coarse_thr_path, coarse_labels, default=0.5)
    subtype_thresholds = load_thresholds(subtype_thr_path, subtype_labels, default=0.5)

    if args.override_has_madd_threshold is not None and "has_madd" in coarse_thresholds:
        coarse_thresholds["has_madd"] = args.override_has_madd_threshold
    if args.override_ghunnah_threshold is not None and "ghunnah" in coarse_thresholds:
        coarse_thresholds["ghunnah"] = args.override_ghunnah_threshold
    if args.override_subtype_threshold is not None:
        for label in subtype_thresholds:
            subtype_thresholds[label] = args.override_subtype_threshold

    audio_path = row.get("audio_path")
    if not audio_path:
        raise RuntimeError(
            f"Row {row['id']} has no local audio_path. Rebuild retasy_train.jsonl first."
        )

    sample_rate = int(coarse_cfg.get("sample_rate", 16000))
    n_mfcc = int(coarse_cfg.get("n_mfcc", 13))

    waveform, sr = load_local_audio(audio_path)
    waveform = maybe_resample(waveform, sr, sample_rate)
    mfcc_extractor = build_feature_extractor(sample_rate, n_mfcc)
    x = extract_features(waveform, mfcc_extractor)

    coarse_probs = predict_probs(coarse_model, x, device)
    coarse_scored = []
    coarse_pred = []

    for label, prob in zip(coarse_labels, coarse_probs):
        thr = coarse_thresholds[label]
        is_pred = prob >= thr
        if is_pred:
            coarse_pred.append(label)
        coarse_scored.append(
            {
                "label": label,
                "probability": float(prob),
                "threshold": float(thr),
                "predicted": bool(is_pred),
            }
        )

    has_madd_pred = "has_madd" in coarse_pred

    subtype_scored = []
    subtype_pred = []

    if has_madd_pred:
        subtype_probs = predict_probs(subtype_model, x, device)

        for label, prob in zip(subtype_labels, subtype_probs):
            thr = subtype_thresholds[label]
            is_pred = prob >= thr
            if is_pred:
                subtype_pred.append(label)
            subtype_scored.append(
                {
                    "label": label,
                    "probability": float(prob),
                    "threshold": float(thr),
                    "predicted": bool(is_pred),
                }
            )

    diagnosis = {
        "id": row["id"],
        "surah_name": row.get("surah_name"),
        "verse_key": row.get("quranjson_verse_key"),
        "ayah_text": row.get("aya_text"),
        "gold_duration_rules": row.get("duration_rules", []),

        "coarse_head": {
            "predicted_labels": coarse_pred,
            "scored_labels": coarse_scored,
        },

        "madd_subtype_head": {
            "ran": has_madd_pred,
            "predicted_labels": subtype_pred,
            "scored_labels": subtype_scored,
        },

        "final_duration_diagnosis": {
            "ghunnah_present": "ghunnah" in coarse_pred,
            "has_madd": has_madd_pred,
            "madd_subtypes": subtype_pred if has_madd_pred else [],
            "final_labels": (
                (["ghunnah"] if "ghunnah" in coarse_pred else [])
                + (subtype_pred if has_madd_pred else [])
            ),
        },
    }

    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(diagnosis, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
