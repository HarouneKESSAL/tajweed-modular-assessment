from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from scripts.system.run_ayah_content_inference import (
    ayah_acceptance_verdict,
    ayah_feedback,
    ayah_quality_label,
    char_accuracy,
    compact,
    edit_distance,
    greedy_decode,
    load_decoder_config,
    load_jsonl,
    load_model,
    resolve_path,
)


def reciter_from_id(sample_id: str) -> str:
    m = re.match(r"^hf_quran_md_ayah_route_(.+?)_\d{3}_\d{3}_\d+", sample_id)
    return m.group(1) if m else "unknown"


def row_text(row: dict[str, Any]) -> str:
    return row.get("normalized_text") or row.get("text") or row.get("source_text") or ""


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {
            "samples": 0,
            "avg_score": 0.0,
            "avg_char_accuracy": 0.0,
            "avg_edit_distance": 0.0,
            "exact_rate": 0.0,
            "accepted_rate": 0.0,
            "acceptance_counts": {},
            "quality_counts": {},
        }

    accepted = [x for x in items if str(x["acceptance_verdict"]).startswith("accepted")]
    exact = [x for x in items if x["exact_match"]]

    return {
        "samples": n,
        "avg_score": sum(x["score"] for x in items) / n,
        "avg_char_accuracy": sum(x["char_accuracy"] for x in items) / n,
        "avg_edit_distance": sum(x["edit_distance"] for x in items) / n,
        "avg_edit_rate": sum(x["edit_rate"] for x in items) / n,
        "exact_rate": len(exact) / n,
        "accepted_rate": len(accepted) / n,
        "acceptance_counts": dict(Counter(x["acceptance_verdict"] for x in items)),
        "quality_counts": dict(Counter(x["quality"] for x in items)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/content_ayah_hf_v2_balanced_hd96.pt")
    parser.add_argument("--decoder-config", default="configs/content_ayah_decoder_bp12.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/batch_ayah_content_ssl_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    manifest_path = resolve_path(args.manifest)
    rows_all = load_jsonl(manifest_path)

    indices = [
        i for i, row in enumerate(rows_all)
        if args.split == "all" or row.get("split", "train") == args.split
    ]
    if not indices:
        indices = list(range(len(rows_all)))
    if args.limit > 0:
        indices = indices[: args.limit]

    selected_rows = [rows_all[i] for i in indices]
    rows_by_id = {
        str(row.get("id") or row.get("sample_id") or f"sample_{i}"): row
        for i, row in enumerate(selected_rows)
    }

    decoder_cfg = load_decoder_config(resolve_path(args.decoder_config))
    blank_penalty = float(decoder_cfg.get("blank_penalty", 1.2))

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

    results: list[dict[str, Any]] = []

    print("Batch ayah content scoring")
    print("--------------------------")
    print(f"manifest      : {manifest_path}")
    print(f"samples       : {len(ds)}")
    print(f"checkpoint    : {resolve_path(args.checkpoint)}")
    print(f"decoder config: {resolve_path(args.decoder_config)}")
    print(f"blank penalty : {blank_penalty}")
    print(f"device        : {device}")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            ids = batch["ids"]
            golds = batch["texts"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, blank_penalty)

            for sample_id, gold, pred in zip(ids, golds, preds):
                sample_id = str(sample_id)
                row = rows_by_id.get(sample_id, {})

                gold_compact = compact(gold)
                pred_compact = compact(pred)
                acc = char_accuracy(gold_compact, pred_compact)
                ed = edit_distance(gold_compact, pred_compact)
                gold_len = len(gold_compact)
                pred_len = len(pred_compact)
                edit_rate = ed / max(1, gold_len)
                exact_match = gold_compact == pred_compact
                quality = ayah_quality_label(acc, ed, exact_match)
                verdict = ayah_acceptance_verdict(acc, ed, exact_match)

                results.append(
                    {
                        "id": sample_id,
                        "surah": row.get("surah") or row.get("surah_name") or "",
                        "verse_key": row.get("verse_key") or row.get("ayah_key") or "",
                        "reciter": row.get("reciter") or reciter_from_id(sample_id),
                        "audio_path": row.get("audio_path", ""),
                        "gold": gold_compact,
                        "prediction": pred_compact,
                        "score": round(acc * 100.0, 2),
                        "quality": quality,
                        "acceptance_verdict": verdict,
                        "exact_match": exact_match,
                        "char_accuracy": acc,
                        "edit_distance": ed,
                        "edit_rate": edit_rate,
                        "gold_len": gold_len,
                        "pred_len": pred_len,
                        "feedback": ayah_feedback(acc, ed, gold_len, pred_len, exact_match),
                    }
                )

            if batch_index % 25 == 0:
                print(f"processed {min(batch_index * args.batch_size, len(ds))}/{len(ds)} samples...")

    out_jsonl = resolve_path(args.output_jsonl)
    out_summary_json = resolve_path(args.output_summary_json)
    out_md = resolve_path(args.output_md)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_reciter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_verdict: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in results:
        by_reciter[row["reciter"]].append(row)
        by_quality[row["quality"]].append(row)
        by_verdict[row["acceptance_verdict"]].append(row)

    reciter_summary = {
        reciter: summarize(items)
        for reciter, items in sorted(by_reciter.items())
    }

    summary = {
        "manifest": str(manifest_path),
        "split": args.split,
        "limit": args.limit,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "decoder_config": str(resolve_path(args.decoder_config)),
        "blank_penalty": blank_penalty,
        "overall": summarize(results),
        "by_reciter": reciter_summary,
        "worst_examples": sorted(results, key=lambda x: x["char_accuracy"])[:50],
        "best_examples": sorted(results, key=lambda x: x["char_accuracy"], reverse=True)[:50],
    }

    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Batch ayah content scoring")
    lines.append("")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- decoder_config: `{args.decoder_config}`")
    lines.append(f"- blank_penalty: {blank_penalty}")
    lines.append("")
    lines.append("## Overall strict acceptance")
    lines.append("")
    o = summary["overall"]
    lines.append(f"- samples: {o['samples']}")
    lines.append(f"- avg_score: {o['avg_score']:.2f}")
    lines.append(f"- avg_char_accuracy: {o['avg_char_accuracy']:.3f}")
    lines.append(f"- avg_edit_distance: {o['avg_edit_distance']:.3f}")
    lines.append(f"- exact_rate: {o['exact_rate']:.3f}")
    lines.append(f"- accepted_rate: {o['accepted_rate']:.3f}")
    lines.append(f"- acceptance_counts: `{o['acceptance_counts']}`")
    lines.append(f"- quality_counts: `{o['quality_counts']}`")
    lines.append("")
    lines.append("## By reciter")
    lines.append("")
    lines.append("| reciter | samples | avg_score | char_acc | edit | exact_rate | accepted_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for reciter, item in sorted(reciter_summary.items(), key=lambda kv: kv[1]["avg_char_accuracy"]):
        lines.append(
            f"| {reciter} | {item['samples']} | {item['avg_score']:.2f} | {item['avg_char_accuracy']:.3f} | {item['avg_edit_distance']:.3f} | {item['exact_rate']:.3f} | {item['accepted_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for e in summary["worst_examples"][:30]:
        lines.append(f"### {e['id']}")
        lines.append(f"- reciter: {e['reciter']}")
        lines.append(f"- score: {e['score']}")
        lines.append(f"- quality: {e['quality']}")
        lines.append(f"- verdict: {e['acceptance_verdict']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['prediction']}`")
        lines.append(f"- edit_distance: {e['edit_distance']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("overall:", summary["overall"])
    print("saved jsonl:", out_jsonl)
    print("saved summary:", out_summary_json)
    print("saved md:", out_md)


if __name__ == "__main__":
    main()
