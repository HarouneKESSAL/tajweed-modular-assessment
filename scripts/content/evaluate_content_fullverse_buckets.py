from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule


BUCKETS = [
    ("001_020", 1, 20),
    ("021_040", 21, 40),
    ("041_060", 41, 60),
    ("061_090", 61, 90),
    ("091_140", 91, 140),
    ("141_plus", 141, 10000),
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def compact(text: str) -> str:
    return str(text).replace(" ", "")


def get_text(row: dict[str, Any]) -> str:
    return row.get("normalized_text") or row.get("text") or row.get("source_text") or ""


def bucket_name(text: str) -> str:
    n = len(compact(text))
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "unknown"


def edit_distance(a: str, b: str) -> int:
    a = compact(a)
    b = compact(b)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = old
    return dp[-1]


def char_accuracy(gold: str, pred: str) -> float:
    g = compact(gold)
    if not g:
        return 1.0 if not compact(pred) else 0.0
    return max(0.0, 1.0 - edit_distance(gold, pred) / len(g))


def to_int_id_to_char(raw: dict[Any, str]) -> dict[int, str]:
    if isinstance(next(iter(raw.keys())), str):
        return {int(k): v for k, v in raw.items()}
    return raw


def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor, id_to_char: dict[int, str], blank_penalty: float) -> list[str]:
    adjusted = logits.clone()
    adjusted[..., 0] -= blank_penalty

    ids = adjusted.argmax(dim=-1).detach().cpu()
    out: list[str] = []

    for seq, length in zip(ids, input_lengths.detach().cpu().tolist()):
        chars = []
        prev = 0
        for idx in seq[:length].tolist():
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(id_to_char.get(idx, ""))
            prev = idx
        out.append("".join(chars))

    return out


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    if total == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "char_accuracy": 0.0,
            "edit_distance": 0.0,
            "avg_gold_len": 0.0,
            "avg_pred_len": 0.0,
        }

    return {
        "samples": total,
        "exact_match": sum(1 for x in items if compact(x["gold"]) == compact(x["pred"])) / total,
        "char_accuracy": sum(x["char_accuracy"] for x in items) / total,
        "edit_distance": sum(x["edit_distance"] for x in items) / total,
        "avg_gold_len": sum(x["gold_len"] for x in items) / total,
        "avg_pred_len": sum(x["pred_len"] for x in items) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blank-penalty", type=float, default=0.4)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_fullverse_eval_ssl_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
    char_to_id = ckpt["char_to_id"]
    id_to_char = to_int_id_to_char(ckpt["id_to_char"])
    hidden_dim = int(ckpt.get("config", {}).get("model", {}).get("hidden_dim", 96))

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = load_jsonl(resolve_path(args.manifest))

    indices = [i for i, row in enumerate(rows) if row.get("split", "train") == args.split]
    if not indices:
        indices = list(range(len(rows)))
    if args.limit > 0:
        indices = indices[: args.limit]

    ds = ChunkedContentDataset(
        resolve_path(args.manifest),
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=resolve_path(args.feature_cache_dir),
        indices=indices,
        char_to_id=char_to_id,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_content_batch)

    examples = []
    by_bucket = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            texts = batch["texts"]
            ids = batch["ids"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, args.blank_penalty)

            for sample_id, gold, pred in zip(ids, texts, preds):
                item = {
                    "id": sample_id,
                    "gold": gold,
                    "pred": pred,
                    "bucket": bucket_name(gold),
                    "gold_len": len(compact(gold)),
                    "pred_len": len(compact(pred)),
                    "edit_distance": edit_distance(gold, pred),
                    "char_accuracy": char_accuracy(gold, pred),
                }
                examples.append(item)
                by_bucket[item["bucket"]].append(item)

    result = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "manifest": str(resolve_path(args.manifest)),
        "split": args.split,
        "blank_penalty": args.blank_penalty,
        "overall": summarize(examples),
        "buckets": {name: summarize(items) for name, items in sorted(by_bucket.items())},
        "worst_examples": sorted(examples, key=lambda x: x["char_accuracy"])[:50],
    }

    out_json = resolve_path(args.output_json)
    out_md = resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Full-verse content diagnostics")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- blank_penalty: {args.blank_penalty}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- samples: {result['overall']['samples']}")
    lines.append(f"- exact_match: {result['overall']['exact_match']:.3f}")
    lines.append(f"- char_accuracy: {result['overall']['char_accuracy']:.3f}")
    lines.append(f"- edit_distance: {result['overall']['edit_distance']:.3f}")
    lines.append(f"- avg_gold_len: {result['overall']['avg_gold_len']:.1f}")
    lines.append(f"- avg_pred_len: {result['overall']['avg_pred_len']:.1f}")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in result["buckets"].items():
        lines.append(
            f"| {name} | {m['samples']} | {m['exact_match']:.3f} | {m['char_accuracy']:.3f} | {m['edit_distance']:.3f} | {m['avg_gold_len']:.1f} | {m['avg_pred_len']:.1f} |"
        )
    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for e in result["worst_examples"][:30]:
        lines.append(f"### {e['id']}")
        lines.append(f"- bucket: {e['bucket']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['pred']}`")
        lines.append(f"- char_accuracy: {e['char_accuracy']:.3f}")
        lines.append(f"- lengths gold/pred: {e['gold_len']}/{e['pred_len']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Full-verse content diagnostics")
    print("-------------------------------")
    print("overall:", result["overall"])
    print("buckets:", result["buckets"])
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
