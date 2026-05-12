from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def to_int_id_to_char(raw):
    if isinstance(next(iter(raw.keys())), str):
        return {int(k): v for k, v in raw.items()}
    return raw


def load_model(checkpoint_path: Path, device: str):
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

    return model, char_to_id, id_to_char


def greedy_decode(logits, input_lengths, id_to_char):
    pred_ids = logits.argmax(dim=-1).detach().cpu()
    out = []

    for seq, length in zip(pred_ids, input_lengths.detach().cpu().tolist()):
        chars = []
        prev = 0
        for idx in seq[:length].tolist():
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(id_to_char.get(idx, ""))
            prev = idx
        out.append("".join(chars))

    return out


def edit_distance(a: str, b: str) -> int:
    a = a.replace(" ", "")
    b = b.replace(" ", "")

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


def char_acc(gold: str, pred: str) -> float:
    gold_compact = gold.replace(" ", "")
    if not gold_compact:
        return 1.0 if not pred.replace(" ", "") else 0.0
    return max(0.0, 1.0 - edit_distance(gold, pred) / len(gold_compact))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--split-mode", default="text")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_compare_side_by_side_ssl_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    baseline_model, baseline_char_to_id, baseline_id_to_char = load_model(
        PROJECT_ROOT / args.baseline_checkpoint,
        device,
    )
    candidate_model, candidate_char_to_id, candidate_id_to_char = load_model(
        PROJECT_ROOT / args.candidate_checkpoint,
        device,
    )

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    train_idx, val_idx = split_content_indices(
        rows,
        val_fraction=0.2,
        seed=7,
        split_mode=args.split_mode,
    )
    indices = val_idx if args.split == "val" else train_idx
    indices = indices[: args.limit]

    # Use candidate vocab for dataset targets because it is the full vocab.
    # The audio features/text are independent of the vocab.
    ds = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=indices,
        char_to_id=candidate_char_to_id,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    examples = []
    stats = {
        "baseline_char_acc_sum": 0.0,
        "candidate_char_acc_sum": 0.0,
        "baseline_exact": 0,
        "candidate_exact": 0,
        "samples": 0,
    }

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            texts = batch["texts"]
            ids = batch["ids"]

            baseline_logits = baseline_model(x, input_lengths)
            candidate_logits = candidate_model(x, input_lengths)

            baseline_preds = greedy_decode(baseline_logits, input_lengths, baseline_id_to_char)
            candidate_preds = greedy_decode(candidate_logits, input_lengths, candidate_id_to_char)

            for sample_id, gold, base_pred, cand_pred in zip(
                ids, texts, baseline_preds, candidate_preds
            ):
                base_acc = char_acc(gold, base_pred)
                cand_acc = char_acc(gold, cand_pred)

                stats["samples"] += 1
                stats["baseline_char_acc_sum"] += base_acc
                stats["candidate_char_acc_sum"] += cand_acc
                stats["baseline_exact"] += int(base_pred.replace(" ", "") == gold.replace(" ", ""))
                stats["candidate_exact"] += int(cand_pred.replace(" ", "") == gold.replace(" ", ""))

                examples.append(
                    {
                        "id": sample_id,
                        "gold": gold,
                        "baseline_pred": base_pred,
                        "candidate_pred": cand_pred,
                        "baseline_char_acc": base_acc,
                        "candidate_char_acc": cand_acc,
                        "baseline_len": len(base_pred.replace(" ", "")),
                        "candidate_len": len(cand_pred.replace(" ", "")),
                        "gold_len": len(gold.replace(" ", "")),
                    }
                )

    n = max(1, stats["samples"])
    summary = {
        "samples": stats["samples"],
        "baseline_checkpoint": args.baseline_checkpoint,
        "candidate_checkpoint": args.candidate_checkpoint,
        "baseline_vocab_size_including_blank": len(baseline_char_to_id) + 1,
        "candidate_vocab_size_including_blank": len(candidate_char_to_id) + 1,
        "baseline_char_accuracy": stats["baseline_char_acc_sum"] / n,
        "candidate_char_accuracy": stats["candidate_char_acc_sum"] / n,
        "baseline_exact": stats["baseline_exact"] / n,
        "candidate_exact": stats["candidate_exact"] / n,
        "examples": examples,
    }

    out_json = PROJECT_ROOT / args.output_json
    out_md = PROJECT_ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Content checkpoint side-by-side comparison")
    lines.append("")
    lines.append(f"- samples: {summary['samples']}")
    lines.append(f"- baseline vocab incl blank: {summary['baseline_vocab_size_including_blank']}")
    lines.append(f"- candidate vocab incl blank: {summary['candidate_vocab_size_including_blank']}")
    lines.append(f"- baseline char accuracy: {summary['baseline_char_accuracy']:.3f}")
    lines.append(f"- candidate char accuracy: {summary['candidate_char_accuracy']:.3f}")
    lines.append(f"- baseline exact: {summary['baseline_exact']:.3f}")
    lines.append(f"- candidate exact: {summary['candidate_exact']:.3f}")
    lines.append("")
    lines.append("## Worst candidate examples")
    lines.append("")

    for e in sorted(examples, key=lambda x: x["candidate_char_acc"])[:40]:
        lines.append(f"### {e['id']}")
        lines.append("")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- baseline: `{e['baseline_pred']}`")
        lines.append(f"- candidate: `{e['candidate_pred']}`")
        lines.append(f"- baseline_char_acc: {e['baseline_char_acc']:.3f}")
        lines.append(f"- candidate_char_acc: {e['candidate_char_acc']:.3f}")
        lines.append(f"- lengths gold/baseline/candidate: {e['gold_len']}/{e['baseline_len']}/{e['candidate_len']}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Content side-by-side comparison")
    print("-------------------------------")
    print(f"samples: {summary['samples']}")
    print(f"baseline vocab incl blank: {summary['baseline_vocab_size_including_blank']}")
    print(f"candidate vocab incl blank: {summary['candidate_vocab_size_including_blank']}")
    print(f"baseline char accuracy: {summary['baseline_char_accuracy']:.3f}")
    print(f"candidate char accuracy: {summary['candidate_char_accuracy']:.3f}")
    print(f"baseline exact: {summary['baseline_exact']:.3f}")
    print(f"candidate exact: {summary['candidate_exact']:.3f}")
    print(f"saved json: {out_json}")
    print(f"saved md: {out_md}")


if __name__ == "__main__":
    main()
