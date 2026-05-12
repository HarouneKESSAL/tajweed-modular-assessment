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

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from scripts.system.run_ayah_content_inference import (
    char_accuracy,
    compact,
    edit_distance,
    greedy_decode,
    load_decoder_config,
    load_model,
    resolve_path,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]



def select_indices(rows: list[dict[str, Any]], split: str, limit: int) -> list[int]:
    indices = [
        i for i, row in enumerate(rows)
        if split == "all" or row.get("split", "train") == split
    ]

    if not indices:
        indices = list(range(len(rows)))

    if limit > 0:
        indices = indices[:limit]

    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_v6c_export_ssl_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    manifest_path = resolve_path(args.manifest)
    rows = load_jsonl(manifest_path)
    indices = select_indices(rows, args.split, args.limit)

    decoder_cfg = load_decoder_config(resolve_path(args.decoder_config))
    blank_penalty = float(decoder_cfg.get("blank_penalty", 0.0))

    model, id_to_char, char_to_id = load_model(resolve_path(args.checkpoint), device)

    ds = ChunkedContentDataset(
        manifest_path,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=resolve_path(args.feature_cache_dir),
        indices=indices,
        char_to_id=char_to_id,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_content_batch)

    results = []

    print("Exporting tuned chunk content predictions")
    print("----------------------------------------")
    print("manifest:", manifest_path)
    print("samples:", len(ds))
    print("checkpoint:", resolve_path(args.checkpoint))
    print("decoder config:", resolve_path(args.decoder_config))
    print("blank penalty:", blank_penalty)
    print("device:", device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            ids = batch["ids"]
            golds = batch["texts"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, blank_penalty)

            for sample_id, gold, pred in zip(ids, golds, preds):
                gold_c = compact(gold)
                pred_c = compact(pred)
                ed = edit_distance(gold_c, pred_c)
                acc = char_accuracy(gold_c, pred_c)

                results.append({
                    "id": str(sample_id),
                    "gold": gold_c,
                    "pred": pred_c,
                    "exact": gold_c == pred_c,
                    "char_accuracy": acc,
                    "edit_distance": ed,
                    "gold_len": len(gold_c),
                    "pred_len": len(pred_c),
                    "len_delta": len(pred_c) - len(gold_c),
                })

            if batch_idx % 25 == 0:
                print(f"processed {min(batch_idx * args.batch_size, len(ds))}/{len(ds)} samples...")

    overall = {
        "samples": len(results),
        "exact_match": sum(1 for r in results if r["exact"]) / max(1, len(results)),
        "char_accuracy": sum(r["char_accuracy"] for r in results) / max(1, len(results)),
        "edit_distance": sum(r["edit_distance"] for r in results) / max(1, len(results)),
        "avg_gold_len": sum(r["gold_len"] for r in results) / max(1, len(results)),
        "avg_pred_len": sum(r["pred_len"] for r in results) / max(1, len(results)),
    }

    output = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "manifest": str(manifest_path),
        "split": args.split,
        "decoder_config": str(resolve_path(args.decoder_config)),
        "blank_penalty": blank_penalty,
        "overall": overall,
        "examples": results,
        "worst_examples": sorted(results, key=lambda r: r["char_accuracy"])[:50],
    }

    out_json = resolve_path(args.output_json)
    out_md = resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Tuned chunk content predictions")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    for k, v in overall.items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")

    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for e in output["worst_examples"][:30]:
        lines.append(f"### {e['id']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['pred']}`")
        lines.append(f"- char_accuracy: {e['char_accuracy']:.3f}")
        lines.append(f"- edit_distance: {e['edit_distance']}")
        lines.append(f"- lengths gold/pred: {e['gold_len']}/{e['pred_len']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("overall:", overall)
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
