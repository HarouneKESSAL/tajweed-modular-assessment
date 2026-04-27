from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from itertools import product

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from tajweed_assessment.data.localized_duration_dataset import (
    LocalizedDurationDataset,
    collate_localized_duration_batch,
    load_jsonl,
)
from tajweed_assessment.utils.seed import seed_everything

from duration.evaluate_localized_duration_spans import (
    split_indices_by_reciter,
    contiguous_spans_from_probs,
    gold_spans_for_label,
    expand_span,
    match_spans,
    f1_from_counts,
)


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


def collect_outputs(model, loader, dataset, device: str):
    outputs = []
    row_offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["input_lengths"].to(device)
            logits = model(x, lengths).cpu()
            probs = torch.sigmoid(logits)
            for b in range(probs.size(0)):
                outputs.append(
                    {
                        "row": dataset.rows[row_offset + b],
                        "probs": probs[b, : int(batch["input_lengths"][b].item())].clone(),
                        "clip_target": batch["clip_targets"][b].int().clone(),
                    }
                )
            row_offset += probs.size(0)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/alignment/duration_time_projection_strict.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/localized_duration_model.pt")
    parser.add_argument("--output", default="checkpoints/localized_duration_decoder.json")
    parser.add_argument("--split", choices=["val", "train", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--gold-pad-sec", type=float, default=0.03)
    parser.add_argument("--gold-min-span-sec", type=float, default=0.08)
    parser.add_argument("--feature-cache-dir", default="data/interim/localized_duration_mfcc_cache")
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    manifest_path = PROJECT_ROOT / args.manifest
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

    dataset = LocalizedDurationDataset(
        manifest_path,
        indices=indices,
        label_vocab=label_vocab,
        sample_rate=int(config.get("sample_rate", 16000)),
        n_mfcc=int(config.get("n_mfcc", 13)),
        hop_length=int(config.get("hop_length", 160)),
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_localized_duration_batch,
    )

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

    outputs = collect_outputs(model, loader, dataset, device)
    frame_hop_sec = int(config.get("hop_length", 160)) / int(config.get("sample_rate", 16000))
    threshold_grid = [float(x) for x in args.thresholds.split(",")]

    best = {
        "score": -1.0,
        "thresholds": {label: 0.5 for label in label_vocab},
        "clip_macro_f1": 0.0,
        "span_macro_f1": 0.0,
        "per_label_clip": {},
        "per_label_span": {},
    }

    for values in product(threshold_grid, repeat=len(label_vocab)):
        thresholds = {label: value for label, value in zip(label_vocab, values)}

        clip_tp = torch.zeros(len(label_vocab), dtype=torch.float32)
        clip_fp = torch.zeros(len(label_vocab), dtype=torch.float32)
        clip_fn = torch.zeros(len(label_vocab), dtype=torch.float32)
        span_counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in label_vocab}

        for item in outputs:
            probs = item["probs"]
            clip_probs = probs.max(dim=0).values
            clip_pred = torch.tensor(
                [1 if clip_probs[j].item() >= thresholds[label_vocab[j]] else 0 for j in range(len(label_vocab))],
                dtype=torch.int,
            )
            clip_target = item["clip_target"]
            clip_tp += ((clip_pred == 1) & (clip_target == 1)).float()
            clip_fp += ((clip_pred == 1) & (clip_target == 0)).float()
            clip_fn += ((clip_pred == 0) & (clip_target == 1)).float()

            for j, label in enumerate(label_vocab):
                pred_spans = contiguous_spans_from_probs(
                    probs[:, j],
                    threshold=thresholds[label],
                    frame_hop_sec=frame_hop_sec,
                    label=label,
                )
                gold_spans = [
                    expand_span(span, pad_sec=args.gold_pad_sec, min_span_sec=args.gold_min_span_sec)
                    for span in gold_spans_for_label(item["row"], label)
                ]
                tp, fp, fn, _ = match_spans(pred_spans, gold_spans, iou_threshold=args.iou_threshold)
                span_counts[label]["tp"] += tp
                span_counts[label]["fp"] += fp
                span_counts[label]["fn"] += fn

        clip_precision = clip_tp / (clip_tp + clip_fp).clamp(min=1.0)
        clip_recall = clip_tp / (clip_tp + clip_fn).clamp(min=1.0)
        clip_f1 = 2 * clip_precision * clip_recall / (clip_precision + clip_recall).clamp(min=1e-8)
        clip_macro_f1 = float(clip_f1.mean().item())

        span_results = {}
        span_f1s = []
        for label in label_vocab:
            tp = span_counts[label]["tp"]
            fp = span_counts[label]["fp"]
            fn = span_counts[label]["fn"]
            f1, precision, recall = f1_from_counts(tp, fp, fn)
            span_results[label] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            span_f1s.append(f1)
        span_macro_f1 = sum(span_f1s) / max(len(span_f1s), 1)

        score = span_macro_f1
        if score > best["score"]:
            best = {
                "score": score,
                "thresholds": thresholds,
                "clip_macro_f1": clip_macro_f1,
                "span_macro_f1": span_macro_f1,
                "per_label_clip": {
                    label: {
                        "f1": float(clip_f1[j].item()),
                        "precision": float(clip_precision[j].item()),
                        "recall": float(clip_recall[j].item()),
                        "tp": int(clip_tp[j].item()),
                        "fp": int(clip_fp[j].item()),
                        "fn": int(clip_fn[j].item()),
                    }
                    for j, label in enumerate(label_vocab)
                },
                "per_label_span": span_results,
            }

    payload = {
        "checkpoint": str(ckpt_path),
        "manifest": str(manifest_path),
        "split": args.split,
        "iou_threshold": float(args.iou_threshold),
        "gold_pad_sec": float(args.gold_pad_sec),
        "gold_min_span_sec": float(args.gold_min_span_sec),
        "decoder_config": {label: {"threshold": float(value)} for label, value in best["thresholds"].items()},
        "clip_macro_f1": best["clip_macro_f1"],
        "span_macro_f1": best["span_macro_f1"],
        "per_label_clip": best["per_label_clip"],
        "per_label_span": best["per_label_span"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved decoder config to: {output_path}")
    print(f"Span macro F1: {best['span_macro_f1']:.3f}")
    print(f"Clip macro F1: {best['clip_macro_f1']:.3f}")
    print("Thresholds:", {k: round(v, 2) for k, v in best["thresholds"].items()})
    print()
    for label in label_vocab:
        clip = best["per_label_clip"][label]
        span = best["per_label_span"][label]
        print(
            f"{label:10s} "
            f"clip_f1={clip['f1']:.3f} "
            f"span_f1={span['f1']:.3f} "
            f"thr={best['thresholds'][label]:.2f}"
        )


if __name__ == "__main__":
    main()

