from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    normalize_text_target,
    split_content_indices,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.data.labels import BLANK_ID
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs
from tajweed_assessment.utils.io import load_checkpoint


def decode_ids(ids, id_to_char):
    return "".join(id_to_char.get(int(i), "") for i in ids)


def safe_text(value: str | None) -> str:
    if value is None:
        return ""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower() in {"cp1252", "charmap"}:
        return str(value).encode("ascii", "backslashreplace").decode("ascii")
    return str(value)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    dist = levenshtein(gold, pred)
    return max(0.0, 1.0 - (dist / len(gold)))


def collapse_excess_repetitions(text: str, max_run: int = 2) -> str:
    if not text:
        return text
    out = [text[0]]
    run_char = text[0]
    run_len = 1
    for ch in text[1:]:
        if ch == run_char:
            run_len += 1
            if run_len <= max_run:
                out.append(ch)
        else:
            run_char = ch
            run_len = 1
            out.append(ch)
    return "".join(out)


def content_postprocess(text: str) -> str:
    return collapse_excess_repetitions(text, max_run=2)


def apply_blank_penalty(log_probs: torch.Tensor, blank_penalty: float) -> torch.Tensor:
    if blank_penalty == 0.0:
        return log_probs
    adjusted = log_probs.clone()
    adjusted[..., BLANK_ID] -= blank_penalty
    return adjusted


def repeated_char_error_count(gold: str, pred: str) -> int:
    count = 0
    i = 1
    while i < len(pred):
        if pred[i] == pred[i - 1]:
            if i >= len(gold) or pred[i] != gold[i]:
                count += 1
        i += 1
    return count


def align_ops(gold: str, pred: str):
    m, n = len(gold), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gold[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if gold[i - 1] == pred[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    ops.append(("match", gold[i - 1], pred[j - 1]))
                else:
                    ops.append(("substitution", gold[i - 1], pred[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("deletion", gold[i - 1], ""))
            i -= 1
            continue
        ops.append(("insertion", "", pred[j - 1]))
        j -= 1

    ops.reverse()
    return ops


def length_bucket(length: int) -> str:
    if length <= 3:
        return "01-03"
    if length <= 5:
        return "04-05"
    if length <= 8:
        return "06-08"
    if length <= 12:
        return "09-12"
    return "13+"


def edit_bucket(distance: int) -> str:
    if distance == 0:
        return "0"
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if distance <= 4:
        return "3-4"
    return "5+"


@dataclass
class EvalRow:
    gold: str
    pred: str
    exact: bool
    char_acc: float
    edit_distance: int
    repeated_errors: int


def summarize_rows(rows: list[EvalRow]) -> dict:
    total = len(rows)
    exact = sum(1 for row in rows if row.exact)
    return {
        "count": total,
        "exact_match": exact / max(total, 1),
        "char_accuracy": sum(row.char_acc for row in rows) / max(total, 1),
        "edit_distance": sum(row.edit_distance for row in rows) / max(total, 1),
        "repeated_errors": sum(row.repeated_errors for row in rows) / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module.pt")
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--split-mode", choices=["reciter", "text"], default="reciter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--use-cleanup", action="store_true")
    parser.add_argument("--blank-penalty", type=float, default=0.0)
    parser.add_argument("--hardest-count", type=int, default=10)
    parser.add_argument("--example-count", type=int, default=10)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    train_idx, val_idx = split_content_indices(
        dataset.rows,
        val_fraction=0.2,
        seed=train_cfg["seed"],
        split_mode=args.split_mode,
    )
    if args.split == "train":
        indices = train_idx
    elif args.split == "val":
        indices = val_idx
    else:
        indices = list(range(len(dataset)))
    if args.limit > 0:
        indices = indices[: args.limit]

    dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(train_cfg["batch_size"], max(1, len(dataset) or 1)),
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    checkpoint = load_checkpoint(PROJECT_ROOT / args.checkpoint)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    ckpt_model_cfg = checkpoint.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(checkpoint["char_to_id"]) + 1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[EvalRow] = []
    length_groups: defaultdict[str, list[EvalRow]] = defaultdict(list)
    edit_groups: defaultdict[str, list[EvalRow]] = defaultdict(list)
    op_counts = Counter()
    deleted_chars = Counter()
    inserted_chars = Counter()
    substituted_pairs = Counter()
    phrase_groups: dict[str, list[EvalRow]] = defaultdict(list)
    examples = []

    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"])
            decode_log_probs = apply_blank_penalty(log_probs.cpu(), args.blank_penalty)
            decoded = greedy_decode_from_log_probs(decode_log_probs, batch["input_lengths"])
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                if args.use_cleanup:
                    pred = content_postprocess(pred)
                row = EvalRow(
                    gold=gold,
                    pred=pred,
                    exact=(pred == gold),
                    char_acc=char_accuracy(gold, pred),
                    edit_distance=levenshtein(gold, pred),
                    repeated_errors=repeated_char_error_count(gold, pred),
                )
                rows.append(row)
                length_groups[length_bucket(len(gold))].append(row)
                edit_groups[edit_bucket(row.edit_distance)].append(row)
                phrase_groups[gold].append(row)

                for op, ref, hyp in align_ops(gold, pred):
                    if op == "match":
                        continue
                    op_counts[op] += 1
                    if op == "deletion":
                        deleted_chars[ref] += 1
                    elif op == "insertion":
                        inserted_chars[hyp] += 1
                    elif op == "substitution":
                        substituted_pairs[f"{ref}->{hyp}"] += 1

                if not row.exact:
                    examples.append(
                        {
                            "gold": gold,
                            "pred": pred,
                            "edit_distance": row.edit_distance,
                            "char_accuracy": row.char_acc,
                            "repeated_errors": row.repeated_errors,
                            "length": len(gold),
                        }
                    )

    overall = summarize_rows(rows)
    length_summary = {bucket: summarize_rows(group) for bucket, group in sorted(length_groups.items())}
    edit_summary = {bucket: summarize_rows(group) for bucket, group in sorted(edit_groups.items(), key=lambda item: item[0])}

    hardest_phrases = []
    for phrase, group in phrase_groups.items():
        summary = summarize_rows(group)
        hardest_phrases.append(
            {
                "text": phrase,
                "count": len(group),
                "exact_match": summary["exact_match"],
                "char_accuracy": summary["char_accuracy"],
                "edit_distance": summary["edit_distance"],
                "repeated_errors": summary["repeated_errors"],
            }
        )
    hardest_phrases.sort(key=lambda item: (item["char_accuracy"], -item["count"], -item["edit_distance"]))
    examples.sort(key=lambda item: (-item["edit_distance"], item["char_accuracy"], -item["length"]))

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "split": args.split,
        "split_mode": args.split_mode,
        "samples": len(rows),
        "use_cleanup": bool(args.use_cleanup),
        "blank_penalty": float(args.blank_penalty),
        "overall": overall,
        "by_length_bucket": length_summary,
        "by_edit_distance_bucket": edit_summary,
        "error_type_counts": dict(op_counts),
        "top_deleted_chars": deleted_chars.most_common(args.hardest_count),
        "top_inserted_chars": inserted_chars.most_common(args.hardest_count),
        "top_substitutions": substituted_pairs.most_common(args.hardest_count),
        "hardest_phrases": hardest_phrases[: args.hardest_count],
        "representative_failures": examples[: args.example_count],
    }

    print("Chunked content failure summary:")
    print(
        f"- samples={summary['samples']} exact_match={overall['exact_match']:.3f} "
        f"char_accuracy={overall['char_accuracy']:.3f} edit_distance={overall['edit_distance']:.3f} "
        f"repeat_errors={overall['repeated_errors']:.3f}"
    )
    print("")
    print("By length bucket:")
    for bucket, stats in summary["by_length_bucket"].items():
        print(
            f"- {bucket}: count={stats['count']} exact_match={stats['exact_match']:.3f} "
            f"char_accuracy={stats['char_accuracy']:.3f} edit_distance={stats['edit_distance']:.3f}"
        )
    print("")
    print("By edit-distance bucket:")
    for bucket, stats in summary["by_edit_distance_bucket"].items():
        print(
            f"- {bucket}: count={stats['count']} exact_match={stats['exact_match']:.3f} "
            f"char_accuracy={stats['char_accuracy']:.3f}"
        )
    print("")
    print("Error types:")
    for label in ["deletion", "insertion", "substitution"]:
        print(f"- {label}: {op_counts[label]}")
    print("")
    print(f"Top substitutions (top {args.hardest_count}):")
    for pair, count in summary["top_substitutions"]:
        print(f"- {safe_text(pair)}: {count}")
    print("")
    print(f"Top deleted chars (top {args.hardest_count}):")
    for ch, count in summary["top_deleted_chars"]:
        print(f"- {safe_text(ch)}: {count}")
    print("")
    print(f"Top inserted chars (top {args.hardest_count}):")
    for ch, count in summary["top_inserted_chars"]:
        print(f"- {safe_text(ch)}: {count}")
    print("")
    print(f"Hardest phrases (top {min(args.hardest_count, len(hardest_phrases))}):")
    for item in summary["hardest_phrases"]:
        print(
            f"- {safe_text(item['text'])}: count={item['count']} exact_match={item['exact_match']:.3f} "
            f"char_accuracy={item['char_accuracy']:.3f} edit_distance={item['edit_distance']:.3f}"
        )
    print("")
    print(f"Representative failures (top {min(args.example_count, len(examples))}):")
    if not summary["representative_failures"]:
        print("- None")
    else:
        for item in summary["representative_failures"]:
            print(
                f"- EDIT={item['edit_distance']} CHAR_ACC={item['char_accuracy']:.3f} "
                f"REPEAT_ERR={item['repeated_errors']} LEN={item['length']}"
            )
            print(f"  GOLD: {safe_text(item['gold'])}")
            print(f"  PRED: {safe_text(item['pred'])}")

    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("")
        print(f"Saved analysis JSON to {output_path}")


if __name__ == "__main__":
    main()

