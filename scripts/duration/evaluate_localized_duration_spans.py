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
from torch.utils.data import DataLoader

from tajweed_assessment.data.localized_duration_dataset import (
    LocalizedDurationDataset,
    collate_localized_duration_batch,
    load_jsonl,
    normalize_duration_label,
)
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
                "start_sec": float(start * frame_hop_sec),
                "end_sec": float(end * frame_hop_sec),
                "max_prob": float(seg.max().item()),
                "mean_prob": float(seg.mean().item()),
            }
        )
    return spans


def gold_spans_for_label(row, label: str):
    spans = []
    for span in row.get("duration_rule_time_spans", []):
        if normalize_duration_label(span) != label:
            continue
        if span.get("start_sec") is None or span.get("end_sec") is None:
            continue
        spans.append({"label": label, "start_sec": float(span["start_sec"]), "end_sec": float(span["end_sec"])})
    return spans


def expand_span(span, pad_sec: float, min_span_sec: float):
    start_sec = float(span["start_sec"])
    end_sec = float(span["end_sec"])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    start_sec = max(0.0, start_sec - pad_sec)
    end_sec = max(start_sec, end_sec + pad_sec)
    if end_sec - start_sec < min_span_sec:
        center = 0.5 * (start_sec + end_sec)
        half = 0.5 * min_span_sec
        start_sec = max(0.0, center - half)
        end_sec = center + half
    return {**span, "start_sec": start_sec, "end_sec": end_sec}


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
    total_iou = 0.0

    candidates = []
    for i, pred in enumerate(pred_spans):
        for j, gold in enumerate(gold_spans):
            iou = interval_iou(pred, gold)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)
    for iou, i, j in candidates:
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        matches += 1
        total_iou += iou

    tp = matches
    fp = len(pred_spans) - matches
    fn = len(gold_spans) - matches
    mean_iou = total_iou / matches if matches else 0.0
    return tp, fp, fn, mean_iou


def f1_from_counts(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--label-threshold", action="append", default=[])
    parser.add_argument("--decoder-config", default="checkpoints/localized_duration_decoder.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/localized_duration_mfcc_cache")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--gold-pad-sec", type=float, default=0.03)
    parser.add_argument("--gold-min-span-sec", type=float, default=0.08)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint
    manifest_path = PROJECT_ROOT / args.manifest
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
    threshold_overrides = parse_label_overrides(args.label_threshold)
    decoder_thresholds = load_decoder_config(PROJECT_ROOT / args.decoder_config, label_vocab, default_threshold=args.threshold)
    thresholds = {
        label: float(threshold_overrides.get(label, decoder_thresholds.get(label, args.threshold)))
        for label in label_vocab
    }

    frame_hop_sec = int(config.get("hop_length", 160)) / int(config.get("sample_rate", 16000))

    clip_tp = torch.zeros(len(label_vocab), dtype=torch.float32)
    clip_fp = torch.zeros(len(label_vocab), dtype=torch.float32)
    clip_fn = torch.zeros(len(label_vocab), dtype=torch.float32)
    clip_exact = 0
    total_clips = 0
    span_counts = {label: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0} for label in label_vocab}

    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            x = batch["x"].to(device)
            lengths = batch["input_lengths"].to(device)
            logits = model(x, lengths).cpu()
            probs = torch.sigmoid(logits)

            for b in range(probs.size(0)):
                row = dataset.rows[row_offset + b]
                length = int(batch["input_lengths"][b].item())

                clip_probs = probs[b, :length].max(dim=0).values
                clip_pred = torch.tensor(
                    [1 if clip_probs[j].item() >= thresholds[label_vocab[j]] else 0 for j in range(len(label_vocab))],
                    dtype=torch.int,
                )
                clip_target = batch["clip_targets"][b].int()

                clip_tp += ((clip_pred == 1) & (clip_target == 1)).float()
                clip_fp += ((clip_pred == 1) & (clip_target == 0)).float()
                clip_fn += ((clip_pred == 0) & (clip_target == 1)).float()
                if torch.equal(clip_pred, clip_target):
                    clip_exact += 1
                total_clips += 1

                for j, label in enumerate(label_vocab):
                    pred_spans = contiguous_spans_from_probs(
                        probs[b, :length, j],
                        threshold=thresholds[label],
                        frame_hop_sec=frame_hop_sec,
                        label=label,
                    )
                    gold_spans = [
                        expand_span(span, pad_sec=args.gold_pad_sec, min_span_sec=args.gold_min_span_sec)
                        for span in gold_spans_for_label(row, label)
                    ]
                    tp, fp, fn, mean_iou = match_spans(pred_spans, gold_spans, iou_threshold=args.iou_threshold)
                    span_counts[label]["tp"] += tp
                    span_counts[label]["fp"] += fp
                    span_counts[label]["fn"] += fn
                    span_counts[label]["iou_sum"] += mean_iou * tp

            row_offset += probs.size(0)

    clip_precision = clip_tp / (clip_tp + clip_fp).clamp(min=1.0)
    clip_recall = clip_tp / (clip_tp + clip_fn).clamp(min=1.0)
    clip_f1 = 2 * clip_precision * clip_recall / (clip_precision + clip_recall).clamp(min=1e-8)
    micro_tp = clip_tp.sum()
    micro_fp = clip_fp.sum()
    micro_fn = clip_fn.sum()
    micro_precision = micro_tp / (micro_tp + micro_fp).clamp(min=1.0)
    micro_recall = micro_tp / (micro_tp + micro_fn).clamp(min=1.0)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall).clamp(min=1e-8)

    print("Clip-level metrics")
    print(f"  macro_f1   = {clip_f1.mean().item():.3f}")
    print(f"  micro_f1   = {micro_f1.item():.3f}")
    print(f"  exact_match= {clip_exact / max(total_clips, 1):.3f}")
    print()

    print("Per-label clip F1")
    clip_summary = {}
    for label, f1, tp, fp, fn in zip(label_vocab, clip_f1.tolist(), clip_tp.tolist(), clip_fp.tolist(), clip_fn.tolist()):
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        clip_summary[label] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        print(
            f"  {label:10s} f1={f1:.3f} precision={precision:.3f} recall={recall:.3f} "
            f"tp={int(tp)} fp={int(fp)} fn={int(fn)}"
        )
    print()

    print(f"Span-level metrics @ IoU >= {args.iou_threshold}")
    span_summary = {}
    span_f1s = []
    for label in label_vocab:
        tp = span_counts[label]["tp"]
        fp = span_counts[label]["fp"]
        fn = span_counts[label]["fn"]
        f1, precision, recall = f1_from_counts(tp, fp, fn)
        mean_iou = span_counts[label]["iou_sum"] / tp if tp else 0.0
        span_f1s.append(f1)
        span_summary[label] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou": mean_iou,
        }
        print(
            f"  {label:10s} f1={f1:.3f} precision={precision:.3f} recall={recall:.3f} "
            f"tp={tp} fp={fp} fn={fn} mean_iou={mean_iou:.3f}"
        )
    print()
    print(f"Span macro F1 = {sum(span_f1s) / max(len(span_f1s), 1):.3f}")

    if args.output_json:
        payload = {
            "manifest": str(manifest_path),
            "checkpoint": str(ckpt_path),
            "split": args.split,
            "threshold": float(args.threshold),
            "thresholds": thresholds,
            "iou_threshold": float(args.iou_threshold),
            "gold_pad_sec": float(args.gold_pad_sec),
            "gold_min_span_sec": float(args.gold_min_span_sec),
            "clip": {
                "macro_f1": float(clip_f1.mean().item()),
                "micro_f1": float(micro_f1.item()),
                "exact_match": float(clip_exact / max(total_clips, 1)),
                "per_label": clip_summary,
            },
            "spans": {
                "macro_f1": float(sum(span_f1s) / max(len(span_f1s), 1)),
                "per_label": span_summary,
            },
        }
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Saved evaluation JSON to {output_path}")


if __name__ == "__main__":
    main()

