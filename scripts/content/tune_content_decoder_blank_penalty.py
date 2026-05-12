from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    split_content_indices,
)
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def compact(text: str) -> str:
    return str(text).replace(" ", "")


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


def evaluate_blank_penalty(model, loader, id_to_char, blank_penalty: float, device: str) -> dict[str, Any]:
    model.eval()

    total = 0
    exact = 0
    char_sum = 0.0
    edit_sum = 0.0
    pred_len_sum = 0
    gold_len_sum = 0
    examples = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            texts = batch["texts"]
            ids = batch["ids"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, blank_penalty)

            for sample_id, gold, pred in zip(ids, texts, preds):
                total += 1
                ed = edit_distance(gold, pred)
                acc = char_accuracy(gold, pred)

                exact += int(compact(gold) == compact(pred))
                char_sum += acc
                edit_sum += ed
                pred_len_sum += len(compact(pred))
                gold_len_sum += len(compact(gold))

                if len(examples) < 30 and acc < 0.7:
                    examples.append(
                        {
                            "id": sample_id,
                            "gold": gold,
                            "pred": pred,
                            "char_accuracy": acc,
                            "edit_distance": ed,
                            "gold_len": len(compact(gold)),
                            "pred_len": len(compact(pred)),
                        }
                    )

    return {
        "blank_penalty": blank_penalty,
        "samples": total,
        "exact_match": exact / max(1, total),
        "char_accuracy": char_sum / max(1, total),
        "edit_distance": edit_sum / max(1, total),
        "avg_gold_len": gold_len_sum / max(1, total),
        "avg_pred_len": pred_len_sum / max(1, total),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--split-mode", default="text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_decoder_tune_ssl_cache")
    parser.add_argument("--blank-penalties", nargs="*", type=float, default=[0.0, 0.4, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8])
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

    rows = load_jsonl(resolve_path(args.manifest))
    train_idx, val_idx = split_content_indices(
        rows,
        val_fraction=0.2,
        seed=7,
        split_mode=args.split_mode,
    )
    indices = val_idx if args.split == "val" else train_idx
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

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    results = []
    for penalty in args.blank_penalties:
        metrics = evaluate_blank_penalty(model, loader, id_to_char, penalty, device)
        results.append(metrics)
        print(
            f"blank_penalty={penalty:.2f} "
            f"exact={metrics['exact_match']:.3f} "
            f"char={metrics['char_accuracy']:.3f} "
            f"edit={metrics['edit_distance']:.3f} "
            f"pred_len={metrics['avg_pred_len']:.2f}"
        )

    best = max(results, key=lambda x: (x["char_accuracy"], x["exact_match"]))

    output = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "manifest": str(resolve_path(args.manifest)),
        "split": args.split,
        "split_mode": args.split_mode,
        "samples": len(ds),
        "best": best,
        "results": results,
    }

    out_json = resolve_path(args.output_json)
    out_md = resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Content decoder tuning")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- split_mode: `{args.split_mode}`")
    lines.append(f"- samples: {len(ds)}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| blank_penalty | exact | char_accuracy | edit_distance | avg_pred_len |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['blank_penalty']:.2f} | {r['exact_match']:.3f} | {r['char_accuracy']:.3f} | {r['edit_distance']:.3f} | {r['avg_pred_len']:.2f} |"
        )
    lines.append("")
    lines.append("## Best")
    lines.append("")
    lines.append(f"- blank_penalty: {best['blank_penalty']}")
    lines.append(f"- exact_match: {best['exact_match']:.3f}")
    lines.append(f"- char_accuracy: {best['char_accuracy']:.3f}")
    lines.append(f"- edit_distance: {best['edit_distance']:.3f}")
    lines.append("")
    lines.append("## Example errors")
    lines.append("")
    for e in best["examples"]:
        lines.append(f"### {e['id']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['pred']}`")
        lines.append(f"- char_accuracy: {e['char_accuracy']:.3f}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print()
    print("best:", best)
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
