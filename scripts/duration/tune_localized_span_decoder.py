from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random
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


class LocalizedDurationBiLSTM(nn.Module):
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
        out = self.dropout(out)
        return self.classifier(out)


def split_indices_by_id(rows, val_fraction: float = 0.2, seed: int = 7):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    split = max(1, int((1.0 - val_fraction) * len(indices)))
    train_idx = indices[:split]
    val_idx = indices[split:]

    if len(val_idx) == 0:
        val_idx = indices[-1:]
        train_idx = indices[:-1]

    return train_idx, val_idx


def interval_iou(a, b):
    inter = max(0.0, min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"]))
    union = max(a["end_sec"], b["end_sec"]) - min(a["start_sec"], b["start_sec"])
    if union <= 0:
        return 0.0
    return inter / union


def match_spans(pred_spans, gold_spans, iou_threshold=0.1):
    used_pred = set()
    used_gold = set()
    matches = 0

    candidates = []
    for i, p in enumerate(pred_spans):
        for j, g in enumerate(gold_spans):
            iou = interval_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)

    for iou, i, j in candidates:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        matches += 1

    tp = matches
    fp = len(pred_spans) - matches
    fn = len(gold_spans) - matches
    return tp, fp, fn


def f1_from_counts(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def gold_spans_for_label(row, label: str):
    spans = []
    for span in row.get("duration_rule_time_spans", []):
        if span.get("coarse_group") != label:
            continue
        if span.get("start_sec") is None or span.get("end_sec") is None:
            continue
        spans.append(
            {
                "label": label,
                "start_sec": float(span["start_sec"]),
                "end_sec": float(span["end_sec"]),
            }
        )
    return spans


def decode_spans_from_probs(
    probs: torch.Tensor,
    threshold: float,
    frame_hop_sec: float,
    label: str,
    min_span_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
):
    """
    probs: [T]
    Returns list of contiguous predicted spans after optional merge + min-duration filtering.
    """
    pred = probs >= threshold
    raw_spans = []

    start = None
    for i, flag in enumerate(pred.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            seg = probs[start:end]
            raw_spans.append(
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
        raw_spans.append(
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

    # merge nearby spans
    if raw_spans and merge_gap_sec > 0.0:
        merged = [raw_spans[0]]
        for cur in raw_spans[1:]:
            prev = merged[-1]
            gap = cur["start_sec"] - prev["end_sec"]
            if gap <= merge_gap_sec:
                prev["end_frame"] = cur["end_frame"]
                prev["end_sec"] = cur["end_sec"]
                prev["max_prob"] = max(prev["max_prob"], cur["max_prob"])
                # cheap combined mean
                prev["mean_prob"] = max(prev["mean_prob"], cur["mean_prob"])
            else:
                merged.append(cur)
        raw_spans = merged

    # filter short spans
    if min_span_sec > 0.0:
        raw_spans = [
            s for s in raw_spans
            if (s["end_sec"] - s["start_sec"]) >= min_span_sec
        ]

    return raw_spans


def collect_val_outputs(model, loader, device: str):
    """
    Returns one entry per clip:
      {
        "row": original row,
        "probs": [T, C] on cpu,
        "length": int
      }
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["input_lengths"].to(device)

            logits = model(x, lengths).cpu()
            probs = torch.sigmoid(logits)

            for b in range(probs.size(0)):
                outputs.append(
                    {
                        "row_offset": row_offset + b,
                        "probs": probs[b, : int(batch["input_lengths"][b].item())].clone(),
                    }
                )

            row_offset += probs.size(0)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/alignment/duration_time_projection_strict.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/localized_duration_model.pt",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/localized_span_decoder.json",
    )
    parser.add_argument("--split", choices=["val", "train", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument(
        "--feature-cache-dir",
        default="data/interim/localized_duration_mfcc_cache",
    )

    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min-spans", default="0.00,0.04,0.06,0.08,0.10")
    parser.add_argument("--merge-gaps", default="0.00,0.02,0.04,0.06")

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
    train_idx, val_idx = split_indices_by_id(rows, val_fraction=0.2, seed=seed)

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

    thresholds_grid = [float(x) for x in args.thresholds.split(",")]
    min_spans_grid = [float(x) for x in args.min_spans.split(",")]
    merge_gaps_grid = [float(x) for x in args.merge_gaps.split(",")]

    frame_hop_sec = int(config.get("hop_length", 160)) / int(config.get("sample_rate", 16000))

    outputs = collect_val_outputs(model, loader, device)

    best_configs = {}
    per_label_results = {}

    for j, label in enumerate(label_vocab):
        best = {
            "f1": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.5,
            "min_span_sec": 0.0,
            "merge_gap_sec": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

        for threshold, min_span_sec, merge_gap_sec in product(
            thresholds_grid,
            min_spans_grid,
            merge_gaps_grid,
        ):
            total_tp = total_fp = total_fn = 0

            for k, item in enumerate(outputs):
                row = dataset.rows[k]
                probs = item["probs"][:, j]

                pred_spans = decode_spans_from_probs(
                    probs,
                    threshold=threshold,
                    frame_hop_sec=frame_hop_sec,
                    label=label,
                    min_span_sec=min_span_sec,
                    merge_gap_sec=merge_gap_sec,
                )
                gold_spans = gold_spans_for_label(row, label)

                tp, fp, fn = match_spans(
                    pred_spans,
                    gold_spans,
                    iou_threshold=args.iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

            f1, precision, recall = f1_from_counts(total_tp, total_fp, total_fn)

            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "threshold": threshold,
                    "min_span_sec": min_span_sec,
                    "merge_gap_sec": merge_gap_sec,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                }

        best_configs[label] = {
            "threshold": best["threshold"],
            "min_span_sec": best["min_span_sec"],
            "merge_gap_sec": best["merge_gap_sec"],
        }
        per_label_results[label] = best

    macro_f1 = sum(x["f1"] for x in per_label_results.values()) / max(len(per_label_results), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "manifest": str(manifest_path),
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "decoder_config": best_configs,
        "per_label_results": per_label_results,
        "span_macro_f1": macro_f1,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved decoder config to: {output_path}")
    print(f"Span macro F1: {macro_f1:.3f}")
    print()

    for label in label_vocab:
        r = per_label_results[label]
        print(
            f"{label:10s} "
            f"f1={r['f1']:.3f} "
            f"precision={r['precision']:.3f} "
            f"recall={r['recall']:.3f} "
            f"thr={r['threshold']:.2f} "
            f"min_span={r['min_span_sec']:.2f} "
            f"merge_gap={r['merge_gap_sec']:.2f} "
            f"tp={r['tp']} fp={r['fp']} fn={r['fn']}"
        )


if __name__ == "__main__":
    main()
