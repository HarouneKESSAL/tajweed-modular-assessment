from __future__ import annotations

import argparse
import json
import re
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


def resolve_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def compact(text: str) -> str:
    return str(text or "").replace(" ", "")


def edit_distance(a: str, b: str) -> int:
    a = compact(a)
    b = compact(b)
    dp = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if ca == cb else 1),
            )
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


def reciter_from_id(sample_id: str) -> str:
    m = re.match(r"^hf_quran_md_ayah_route_(.+?)_\d{3}_\d{3}_\d+", sample_id)
    return m.group(1) if m else "unknown"


def greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    id_to_char: dict[int, str],
    blank_penalty: float,
) -> list[str]:
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


def load_model(checkpoint_path: Path, device: str) -> tuple[ContentVerificationModule, dict[int, str], dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    char_to_id = ckpt["char_to_id"]
    id_to_char = to_int_id_to_char(ckpt["id_to_char"])
    hidden_dim = int(ckpt.get("config", {}).get("model", {}).get("hidden_dim", 96))

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, id_to_char, char_to_id


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "char_accuracy": 0.0,
            "edit_distance": 0.0,
            "avg_gold_len": 0.0,
            "avg_pred_len": 0.0,
        }

    return {
        "samples": n,
        "exact_match": sum(1 for x in items if compact(x["gold"]) == compact(x["pred"])) / n,
        "char_accuracy": sum(x["char_accuracy"] for x in items) / n,
        "edit_distance": sum(x["edit_distance"] for x in items) / n,
        "avg_gold_len": sum(x["gold_len"] for x in items) / n,
        "avg_pred_len": sum(x["pred_len"] for x in items) / n,
    }


def evaluate_checkpoint(
    name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    split: str,
    limit: int,
    batch_size: int,
    blank_penalty: float,
    feature_cache_dir: Path,
    device: str,
) -> dict[str, Any]:
    model, id_to_char, char_to_id = load_model(checkpoint_path, device)

    rows = load_jsonl(manifest_path)
    indices = [i for i, row in enumerate(rows) if row.get("split", "train") == split]
    if not indices:
        indices = list(range(len(rows)))
    if limit > 0:
        indices = indices[:limit]

    ds = ChunkedContentDataset(
        manifest_path,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=feature_cache_dir,
        indices=indices,
        char_to_id=char_to_id,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    examples: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            texts = batch["texts"]
            ids = batch["ids"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, blank_penalty)

            for sample_id, gold, pred in zip(ids, texts, preds):
                reciter = reciter_from_id(sample_id)
                ed = edit_distance(gold, pred)
                acc = char_accuracy(gold, pred)

                examples.append(
                    {
                        "id": sample_id,
                        "reciter": reciter,
                        "gold": compact(gold),
                        "pred": compact(pred),
                        "char_accuracy": acc,
                        "edit_distance": ed,
                        "gold_len": len(compact(gold)),
                        "pred_len": len(compact(pred)),
                    }
                )

    by_reciter = defaultdict(list)
    for e in examples:
        by_reciter[e["reciter"]].append(e)

    return {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "overall": summarize(examples),
        "by_reciter": {
            reciter: summarize(items)
            for reciter, items in sorted(by_reciter.items())
        },
        "worst_examples": sorted(examples, key=lambda x: x["char_accuracy"])[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--v2-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--blank-penalty", type=float, default=1.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_ayah_reciter_compare_ssl_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    common = {
        "manifest_path": resolve_path(args.manifest),
        "split": args.split,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "blank_penalty": args.blank_penalty,
        "feature_cache_dir": resolve_path(args.feature_cache_dir),
        "device": device,
    }

    print("Evaluating v1...")
    v1 = evaluate_checkpoint(
        name="v1",
        checkpoint_path=resolve_path(args.v1_checkpoint),
        **common,
    )

    print("Evaluating v2...")
    v2 = evaluate_checkpoint(
        name="v2",
        checkpoint_path=resolve_path(args.v2_checkpoint),
        **common,
    )

    reciters = sorted(set(v1["by_reciter"]) | set(v2["by_reciter"]))
    deltas = []

    for reciter in reciters:
        a = v1["by_reciter"].get(reciter, summarize([]))
        b = v2["by_reciter"].get(reciter, summarize([]))

        deltas.append(
            {
                "reciter": reciter,
                "samples": max(a["samples"], b["samples"]),
                "v1_char_accuracy": a["char_accuracy"],
                "v2_char_accuracy": b["char_accuracy"],
                "delta_char_accuracy": b["char_accuracy"] - a["char_accuracy"],
                "v1_edit_distance": a["edit_distance"],
                "v2_edit_distance": b["edit_distance"],
                "delta_edit_distance": b["edit_distance"] - a["edit_distance"],
                "v1_exact_match": a["exact_match"],
                "v2_exact_match": b["exact_match"],
                "delta_exact_match": b["exact_match"] - a["exact_match"],
            }
        )

    result = {
        "manifest": str(resolve_path(args.manifest)),
        "split": args.split,
        "limit": args.limit,
        "blank_penalty": args.blank_penalty,
        "v1": v1,
        "v2": v2,
        "deltas": sorted(deltas, key=lambda x: x["delta_char_accuracy"], reverse=True),
    }

    out_json = resolve_path(args.output_json)
    out_md = resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Ayah checkpoint comparison by reciter")
    lines.append("")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- blank_penalty: {args.blank_penalty}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| model | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for item in [v1, v2]:
        o = item["overall"]
        lines.append(
            f"| {item['name']} | {o['samples']} | {o['exact_match']:.3f} | {o['char_accuracy']:.3f} | {o['edit_distance']:.3f} | {o['avg_gold_len']:.1f} | {o['avg_pred_len']:.1f} |"
        )

    lines.append("")
    lines.append("## Reciter deltas")
    lines.append("")
    lines.append("| reciter | samples | v1 char | v2 char | Δ char | v1 edit | v2 edit | Δ edit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for d in result["deltas"]:
        lines.append(
            f"| {d['reciter']} | {d['samples']} | {d['v1_char_accuracy']:.3f} | {d['v2_char_accuracy']:.3f} | {d['delta_char_accuracy']:+.3f} | {d['v1_edit_distance']:.3f} | {d['v2_edit_distance']:.3f} | {d['delta_edit_distance']:+.3f} |"
        )

    lines.append("")
    lines.append("## Biggest v2 improvements")
    lines.append("")
    for d in result["deltas"][:8]:
        lines.append(
            f"- {d['reciter']}: char {d['v1_char_accuracy']:.3f} → {d['v2_char_accuracy']:.3f} ({d['delta_char_accuracy']:+.3f}), edit {d['v1_edit_distance']:.2f} → {d['v2_edit_distance']:.2f}"
        )

    lines.append("")
    lines.append("## Biggest v2 regressions")
    lines.append("")
    for d in sorted(result["deltas"], key=lambda x: x["delta_char_accuracy"])[:8]:
        lines.append(
            f"- {d['reciter']}: char {d['v1_char_accuracy']:.3f} → {d['v2_char_accuracy']:.3f} ({d['delta_char_accuracy']:+.3f}), edit {d['v1_edit_distance']:.2f} → {d['v2_edit_distance']:.2f}"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Ayah checkpoint comparison by reciter")
    print("-------------------------------------")
    print("v1 overall:", v1["overall"])
    print("v2 overall:", v2["overall"])
    print()
    print("Top improvements:")
    for d in result["deltas"][:8]:
        print(d)
    print()
    print("Top regressions:")
    for d in sorted(result["deltas"], key=lambda x: x["delta_char_accuracy"])[:8]:
        print(d)
    print()
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
