from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
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


def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    ids = []
    missing = []

    for ch in compact(text):
        idx = char_to_id.get(ch)
        if idx is None:
            missing.append(ch)
        else:
            ids.append(idx)

    if missing:
        raise ValueError(f"Text contains chars missing from checkpoint vocab: {sorted(set(missing))}")

    return ids


def ctc_loss_for_text(
    logits: torch.Tensor,
    input_length: int,
    text: str,
    char_to_id: dict[str, int],
    criterion: nn.CTCLoss,
) -> dict[str, Any]:
    target_ids = encode_text(text, char_to_id)

    if not target_ids:
        return {
            "ctc_loss": None,
            "ctc_loss_per_char": None,
            "ctc_confidence": 0.0,
            "target_len": 0,
        }

    one_logits = logits.unsqueeze(1)  # T, 1, C
    log_probs = one_logits.log_softmax(dim=-1)

    targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    input_lengths = torch.tensor([input_length], dtype=torch.long, device=logits.device)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=logits.device)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    loss_value = float(loss.detach().cpu().item())
    per_char = loss_value / max(1, len(target_ids))

    # This is not a calibrated probability. It is a convenient monotonic confidence proxy:
    # lower per-character CTC loss => higher confidence.
    confidence = math.exp(-per_char)

    return {
        "ctc_loss": loss_value,
        "ctc_loss_per_char": per_char,
        "ctc_confidence": confidence,
        "target_len": len(target_ids),
    }


def expected_text_verdict(
    free_exact: bool,
    free_char_acc: float,
    expected_loss_per_char: float | None,
    expected_confidence: float,
    pred_loss_per_char: float | None,
    edit_dist: int,
) -> str:
    # Keep Qur'an acceptance strict.
    if free_exact:
        return "accepted_free_decode_exact"

    # This threshold is intentionally conservative and should be calibrated.
    # It can mark strong expected-text evidence, but still review-required unless calibrated further.
    if (
        expected_loss_per_char is not None
        and expected_loss_per_char <= 0.35
        and expected_confidence >= 0.70
        and edit_dist <= 1
    ):
        return "accepted_expected_text_near_exact_review_recommended"

    if expected_loss_per_char is not None and expected_loss_per_char <= 0.65:
        return "expected_text_strong_but_review_required"

    if expected_loss_per_char is not None and expected_loss_per_char <= 1.00:
        return "expected_text_plausible_review_required"

    if free_char_acc >= 0.70:
        return "free_decode_similarity_review_required"

    return "not_supported"


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {}

    accepted = [x for x in items if str(x["expected_text_verdict"]).startswith("accepted")]
    strong = [x for x in items if x["expected_text_verdict"] == "expected_text_strong_but_review_required"]
    plausible = [x for x in items if x["expected_text_verdict"] == "expected_text_plausible_review_required"]
    exact = [x for x in items if x["free_exact_match"]]

    return {
        "samples": n,
        "free_exact_rate": len(exact) / n,
        "expected_text_accepted_rate": len(accepted) / n,
        "expected_text_strong_review_rate": len(strong) / n,
        "expected_text_plausible_review_rate": len(plausible) / n,
        "avg_free_char_accuracy": sum(x["free_char_accuracy"] for x in items) / n,
        "avg_expected_ctc_loss_per_char": sum(x["expected_ctc_loss_per_char"] for x in items if x["expected_ctc_loss_per_char"] is not None) / max(1, sum(1 for x in items if x["expected_ctc_loss_per_char"] is not None)),
        "avg_expected_ctc_confidence": sum(x["expected_ctc_confidence"] for x in items) / n,
        "verdict_counts": dict(Counter(x["expected_text_verdict"] for x in items)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/content_ayah_hf_v2_balanced_hd96.pt")
    parser.add_argument("--decoder-config", default="configs/content_ayah_decoder_bp12.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/expected_ayah_ctc_ssl_cache")
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

    rows_by_id = {
        str(rows_all[i].get("id") or rows_all[i].get("sample_id") or f"sample_{i}"): rows_all[i]
        for i in indices
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
    criterion = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)

    results: list[dict[str, Any]] = []

    print("Expected ayah CTC scoring")
    print("-------------------------")
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

            logits_batch = model(x, input_lengths)  # B, T, C
            preds = greedy_decode(logits_batch, input_lengths, id_to_char, blank_penalty)

            for row_i, (sample_id, gold, pred) in enumerate(zip(ids, golds, preds)):
                sample_id = str(sample_id)
                row = rows_by_id.get(sample_id, {})

                gold_compact = compact(gold)
                pred_compact = compact(pred)
                free_acc = char_accuracy(gold_compact, pred_compact)
                ed = edit_distance(gold_compact, pred_compact)
                free_exact = gold_compact == pred_compact

                logits = logits_batch[row_i, : int(input_lengths[row_i].item()), :]
                input_len = int(input_lengths[row_i].item())

                expected_stats = ctc_loss_for_text(
                    logits=logits,
                    input_length=input_len,
                    text=gold_compact,
                    char_to_id=char_to_id,
                    criterion=criterion,
                )

                pred_stats = None
                if pred_compact:
                    pred_stats = ctc_loss_for_text(
                        logits=logits,
                        input_length=input_len,
                        text=pred_compact,
                        char_to_id=char_to_id,
                        criterion=criterion,
                    )

                pred_loss_per_char = None if pred_stats is None else pred_stats["ctc_loss_per_char"]

                verdict = expected_text_verdict(
                    free_exact=free_exact,
                    free_char_acc=free_acc,
                    expected_loss_per_char=expected_stats["ctc_loss_per_char"],
                    expected_confidence=expected_stats["ctc_confidence"],
                    pred_loss_per_char=pred_loss_per_char,
                    edit_dist=ed,
                )

                strict_quality = ayah_quality_label(free_acc, ed, free_exact)
                strict_acceptance = ayah_acceptance_verdict(free_acc, ed, free_exact)

                results.append(
                    {
                        "id": sample_id,
                        "surah": row.get("surah") or row.get("surah_name") or "",
                        "verse_key": row.get("verse_key") or row.get("ayah_key") or "",
                        "reciter": row.get("reciter") or reciter_from_id(sample_id),
                        "gold": gold_compact,
                        "free_prediction": pred_compact,
                        "free_score": round(free_acc * 100.0, 2),
                        "free_quality": strict_quality,
                        "free_acceptance_verdict": strict_acceptance,
                        "free_exact_match": free_exact,
                        "free_char_accuracy": free_acc,
                        "free_edit_distance": ed,
                        "expected_ctc_loss": expected_stats["ctc_loss"],
                        "expected_ctc_loss_per_char": expected_stats["ctc_loss_per_char"],
                        "expected_ctc_confidence": expected_stats["ctc_confidence"],
                        "pred_ctc_loss_per_char": pred_loss_per_char,
                        "expected_text_verdict": verdict,
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
    for row in results:
        by_reciter[row["reciter"]].append(row)

    summary = {
        "manifest": str(manifest_path),
        "split": args.split,
        "limit": args.limit,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "decoder_config": str(resolve_path(args.decoder_config)),
        "blank_penalty": blank_penalty,
        "overall": summarize(results),
        "by_reciter": {reciter: summarize(items) for reciter, items in sorted(by_reciter.items())},
        "best_expected_text": sorted(results, key=lambda x: (x["expected_ctc_loss_per_char"] if x["expected_ctc_loss_per_char"] is not None else 999))[:50],
        "worst_expected_text": sorted(results, key=lambda x: (x["expected_ctc_loss_per_char"] if x["expected_ctc_loss_per_char"] is not None else -1), reverse=True)[:50],
    }

    out_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Expected ayah CTC scoring")
    lines.append("")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- decoder_config: `{args.decoder_config}`")
    lines.append(f"- blank_penalty: {blank_penalty}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    o = summary["overall"]
    for k, v in o.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## By reciter")
    lines.append("")
    lines.append("| reciter | samples | free exact | exp accepted | exp strong review | free char | exp loss/char | exp confidence |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for reciter, item in sorted(summary["by_reciter"].items(), key=lambda kv: kv[1].get("avg_expected_ctc_loss_per_char", 999)):
        lines.append(
            f"| {reciter} | {item['samples']} | {item['free_exact_rate']:.3f} | {item['expected_text_accepted_rate']:.3f} | {item['expected_text_strong_review_rate']:.3f} | {item['avg_free_char_accuracy']:.3f} | {item['avg_expected_ctc_loss_per_char']:.3f} | {item['avg_expected_ctc_confidence']:.3f} |"
        )

    lines.append("")
    lines.append("## Best expected-text support")
    lines.append("")
    for e in summary["best_expected_text"][:20]:
        lines.append(f"### {e['id']}")
        lines.append(f"- reciter: {e['reciter']}")
        lines.append(f"- verdict: {e['expected_text_verdict']}")
        lines.append(f"- free_score: {e['free_score']}")
        lines.append(f"- free_exact: {e['free_exact_match']}")
        lines.append(f"- expected_loss_per_char: {e['expected_ctc_loss_per_char']}")
        lines.append(f"- expected_confidence: {e['expected_ctc_confidence']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- free_prediction: `{e['free_prediction']}`")
        lines.append("")

    lines.append("")
    lines.append("## Worst expected-text support")
    lines.append("")
    for e in summary["worst_expected_text"][:20]:
        lines.append(f"### {e['id']}")
        lines.append(f"- reciter: {e['reciter']}")
        lines.append(f"- verdict: {e['expected_text_verdict']}")
        lines.append(f"- free_score: {e['free_score']}")
        lines.append(f"- expected_loss_per_char: {e['expected_ctc_loss_per_char']}")
        lines.append(f"- expected_confidence: {e['expected_ctc_confidence']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- free_prediction: `{e['free_prediction']}`")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("overall:", summary["overall"])
    print("saved jsonl:", out_jsonl)
    print("saved summary:", out_summary_json)
    print("saved md:", out_md)


if __name__ == "__main__":
    main()
