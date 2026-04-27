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

from tajweed_assessment.data.localized_duration_dataset import (
    LocalizedDurationDataset,
    load_jsonl,
    labels_from_spans,
    normalize_duration_label,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.utils.seed import seed_everything


class LocalizedDurationBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.1) -> None:
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
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)


def split_indices_by_reciter(rows, val_fraction: float = 0.2, seed: int = 7):
    groups = {}
    for idx, row in enumerate(rows):
        key = row.get("reciter_id") or "Unknown"
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    for key in group_keys:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(groups[key])
    train_idx, val_idx = [], []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    if not train_idx or not val_idx:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))
    return train_idx, val_idx


def contiguous_spans_from_probs(probs: torch.Tensor, threshold: float, frame_hop_sec: float, label: str):
    pred = probs >= threshold
    spans = []
    start = None
    for i, flag in enumerate(pred.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            seg = probs[start:end]
            spans.append(
                {
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": float(start * frame_hop_sec),
                    "end_sec": float(end * frame_hop_sec),
                    "max_prob": float(seg.max().item()),
                    "mean_prob": float(seg.mean().item()),
                }
            )
            start = None
    if start is not None:
        end = len(pred)
        seg = probs[start:end]
        spans.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_sec": float(start * frame_hop_sec),
                "end_sec": float(end * frame_hop_sec),
                "max_prob": float(seg.max().item()),
                "mean_prob": float(seg.mean().item()),
            }
        )
    return spans


def find_matching_indices(rows, target_label: str):
    target_label = str(target_label).strip()
    if not target_label:
        return []
    if target_label == "none":
        return [i for i, row in enumerate(rows) if not labels_from_spans(row.get("duration_rule_time_spans", []))]
    return [i for i, row in enumerate(rows) if target_label in labels_from_spans(row.get("duration_rule_time_spans", []))]


def print_json(payload):
    ensure_ascii = False
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower() in {"cp1252", "charmap"}:
        ensure_ascii = True
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def parse_label_overrides(raw_values):
    overrides = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected LABEL=VALUE, got: {raw}")
        label, value = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in override: {raw}")
        overrides[label] = float(value)
    return overrides


def load_decoder_config(path: Path, label_vocab, default_threshold=0.5):
    if not path.exists():
        return {label: float(default_threshold) for label in label_vocab}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("decoder_config", data)
    return {label: float(raw.get(label, {}).get("threshold", default_threshold)) for label in label_vocab}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/alignment/duration_time_projection_strict.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/localized_duration_model.pt")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--find-label", default="")
    parser.add_argument("--match-offset", type=int, default=0)
    parser.add_argument("--list-matches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label-threshold", action="append", default=[])
    parser.add_argument("--decoder-config", default="checkpoints/localized_duration_decoder.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/localized_duration_mfcc_cache")
    parser.add_argument("--normalize-speed", action="store_true")
    parser.add_argument("--target-speech-rate", type=float, default=12.0)
    parser.add_argument("--max-speed-factor", type=float, default=1.35)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    label_vocab = checkpoint["label_vocab"]
    seed = int(config.get("seed", 7))
    seed_everything(seed)

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
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
    dataset = LocalizedDurationDataset(
        PROJECT_ROOT / args.manifest,
        indices=indices,
        label_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        hop_length=int(config.get("hop_length", 160)),
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        speed_config=speed_config,
    )

    selected_index = args.sample_index
    if args.find_label:
        matches = find_matching_indices(dataset.rows, args.find_label)
        if args.list_matches > 0:
            preview = []
            for i in matches[: args.list_matches]:
                row = dataset.rows[i]
                preview.append(
                    {
                        "sample_index": i,
                        "id": row["id"],
                        "labels": labels_from_spans(row.get("duration_rule_time_spans", [])),
                        "surah_name": row.get("surah_name"),
                        "quranjson_verse_key": row.get("quranjson_verse_key"),
                        "normalized_text": row.get("normalized_text"),
                    }
                )
            print_json({"find_label": args.find_label, "count": len(matches), "matches": preview})
            return
        if args.match_offset < 0 or args.match_offset >= len(matches):
            raise IndexError(f"match-offset out of range for label '{args.find_label}': 0 <= idx < {len(matches)}")
        selected_index = matches[args.match_offset]

    if selected_index < 0 or selected_index >= len(dataset):
        raise IndexError(f"sample-index out of range: 0 <= idx < {len(dataset)}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LocalizedDurationBiLSTM(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold_overrides = parse_label_overrides(args.label_threshold)
    decoder_thresholds = load_decoder_config(PROJECT_ROOT / args.decoder_config, label_vocab, default_threshold=args.threshold)
    thresholds = {
        label: float(threshold_overrides.get(label, decoder_thresholds.get(label, args.threshold)))
        for label in label_vocab
    }

    item = dataset[selected_index]
    row = dataset.rows[selected_index]
    x = item["x"].unsqueeze(0).to(device)
    lengths = torch.tensor([item["input_length"]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, lengths)[0].cpu()
        probs = torch.sigmoid(logits)

    frame_hop_sec = int(config.get("hop_length", 160)) / int(config.get("sample_rate", 16000))
    clip_probs = probs.max(dim=0).values.tolist()
    clip_pred = [label for label, p in zip(label_vocab, clip_probs) if p >= thresholds[label]]
    predicted_spans = {
        label: contiguous_spans_from_probs(probs[:, j], thresholds[label], frame_hop_sec, label)
        for j, label in enumerate(label_vocab)
    }
    gold_spans = {
        label: [span for span in row.get("duration_rule_time_spans", []) if normalize_duration_label(span) == label]
        for label in label_vocab
    }

    out = {
        "sample_index": selected_index,
        "id": row["id"],
        "surah_name": row.get("surah_name"),
        "quranjson_verse_key": row.get("quranjson_verse_key"),
        "normalized_text": row.get("normalized_text"),
        "gold_labels": labels_from_spans(row.get("duration_rule_time_spans", [])),
        "thresholds": thresholds,
        "clip_probabilities": {label: float(p) for label, p in zip(label_vocab, clip_probs)},
        "clip_predicted_labels": clip_pred,
        "gold_spans_by_label": gold_spans,
        "predicted_spans_by_label": predicted_spans,
    }
    print_json(out)


if __name__ == "__main__":
    main()

