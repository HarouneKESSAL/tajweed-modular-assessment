from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0
    sm = SequenceMatcher(a=gold, b=pred)
    matches = sum(block.size for block in sm.get_matching_blocks())
    return matches / max(1, len(gold))


def length_bucket(n: int) -> str:
    if n <= 3:
        return "001_003"
    if n <= 6:
        return "004_006"
    if n <= 10:
        return "007_010"
    if n <= 15:
        return "011_015"
    return "016_plus"


def edit_char_stats(gold: str, pred: str) -> dict[str, Counter]:
    deletes = Counter()
    inserts = Counter()
    replaces = Counter()

    sm = SequenceMatcher(a=gold, b=pred)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            deletes.update(gold[i1:i2])
        elif tag == "insert":
            inserts.update(pred[j1:j2])
        elif tag == "replace":
            replaces.update(gold[i1:i2])
            inserts.update(pred[j1:j2])

    return {
        "deleted_gold_chars": deletes,
        "inserted_pred_chars": inserts,
        "replaced_gold_chars": replaces,
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    if n == 0:
        return {
            "samples": 0,
            "exact_rate": 0.0,
            "avg_char_accuracy": 0.0,
            "empty_pred_rate": 0.0,
            "avg_gold_len": 0.0,
            "avg_pred_len": 0.0,
        }

    return {
        "samples": n,
        "exact_rate": sum(1 for x in items if x.get("exact")) / n,
        "avg_char_accuracy": sum(float(x.get("char_accuracy", 0.0)) for x in items) / n,
        "empty_pred_rate": sum(1 for x in items if not x.get("pred", "")) / n,
        "avg_gold_len": sum(int(x.get("gold_len", len(x.get("gold", "")))) for x in items) / n,
        "avg_pred_len": sum(int(x.get("pred_len", len(x.get("pred", "")))) for x in items) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    data = load_json(Path(args.input_json))

    if (
        isinstance(data, dict)
        and isinstance(data.get("content"), dict)
        and isinstance(data["content"].get("examples"), list)
    ):
        examples = data["content"]["examples"]
    elif isinstance(data, dict) and isinstance(data.get("examples"), list):
        examples = data["examples"]
    else:
        raise RuntimeError("No content.examples found. Use the modular suite JSON with saved examples.")

    errors = [e for e in examples if not e.get("exact")]
    empty_preds = [e for e in errors if not e.get("pred", "")]
    too_short = [e for e in errors if int(e.get("len_delta", 0)) <= -2]
    too_long = [e for e in errors if int(e.get("len_delta", 0)) >= 2]

    by_len = defaultdict(list)
    by_gold = defaultdict(list)

    deleted_chars = Counter()
    inserted_chars = Counter()
    replaced_chars = Counter()

    for e in errors:
        gold = str(e.get("gold", ""))
        pred = str(e.get("pred", ""))
        by_len[length_bucket(len(gold))].append(e)
        by_gold[gold].append(e)

        stats = edit_char_stats(gold, pred)
        deleted_chars.update(stats["deleted_gold_chars"])
        inserted_chars.update(stats["inserted_pred_chars"])
        replaced_chars.update(stats["replaced_gold_chars"])

    summary = {
        "overall": summarize(examples),
        "errors": summarize(errors),
        "num_errors": len(errors),
        "num_empty_predictions": len(empty_preds),
        "num_too_short": len(too_short),
        "num_too_long": len(too_long),
        "by_gold_length": {k: summarize(v) for k, v in sorted(by_len.items())},
        "top_repeated_gold_failures": [
            {
                "gold": gold,
                "count": len(items),
                "avg_char_accuracy": sum(float(x.get("char_accuracy", 0.0)) for x in items) / len(items),
                "example_predictions": Counter(str(x.get("pred", "")) for x in items).most_common(5),
            }
            for gold, items in sorted(by_gold.items(), key=lambda kv: len(kv[1]), reverse=True)[:30]
        ],
        "top_deleted_gold_chars": deleted_chars.most_common(20),
        "top_inserted_pred_chars": inserted_chars.most_common(20),
        "top_replaced_gold_chars": replaced_chars.most_common(20),
        "worst_examples": sorted(errors, key=lambda x: float(x.get("char_accuracy", 0.0)))[:50],
        "near_misses": sorted(
            [e for e in errors if float(e.get("char_accuracy", 0.0)) >= 0.75],
            key=lambda x: float(x.get("char_accuracy", 0.0)),
            reverse=True,
        )[:50],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Chunk content error diagnostics")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k, v in summary["overall"].items():
        lines.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines.append(f"| num_errors | {summary['num_errors']} |")
    lines.append(f"| num_empty_predictions | {summary['num_empty_predictions']} |")
    lines.append(f"| num_too_short | {summary['num_too_short']} |")
    lines.append(f"| num_too_long | {summary['num_too_long']} |")

    lines.append("")
    lines.append("## By gold length")
    lines.append("")
    lines.append("| bucket | samples | exact_rate | char_acc | empty_pred_rate | avg_gold_len | avg_pred_len |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for bucket, stats in summary["by_gold_length"].items():
        lines.append(
            f"| {bucket} | {stats['samples']} | {stats['exact_rate']:.3f} | "
            f"{stats['avg_char_accuracy']:.3f} | {stats['empty_pred_rate']:.3f} | "
            f"{stats['avg_gold_len']:.2f} | {stats['avg_pred_len']:.2f} |"
        )

    lines.append("")
    lines.append("## Top repeated gold failures")
    lines.append("")
    lines.append("| gold | count | avg_char_acc | common predictions |")
    lines.append("|---|---:|---:|---|")
    for item in summary["top_repeated_gold_failures"]:
        preds = "; ".join(f"`{p}` x{c}" for p, c in item["example_predictions"])
        lines.append(f"| `{item['gold']}` | {item['count']} | {item['avg_char_accuracy']:.3f} | {preds} |")

    lines.append("")
    lines.append("## Character-level error clues")
    lines.append("")
    lines.append(f"- top deleted gold chars: `{summary['top_deleted_gold_chars']}`")
    lines.append(f"- top inserted pred chars: `{summary['top_inserted_pred_chars']}`")
    lines.append(f"- top replaced gold chars: `{summary['top_replaced_gold_chars']}`")

    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for e in summary["worst_examples"][:30]:
        lines.append(f"### {e.get('id', '')}")
        lines.append(f"- gold: `{e.get('gold', '')}`")
        lines.append(f"- pred: `{e.get('pred', '')}`")
        lines.append(f"- char_accuracy: {float(e.get('char_accuracy', 0.0)):.3f}")
        lines.append(f"- edit_distance: {e.get('edit_distance', '')}")
        lines.append(f"- lengths gold/pred: {e.get('gold_len', '')}/{e.get('pred_len', '')}")
        lines.append("")

    lines.append("")
    lines.append("## Near misses")
    lines.append("")
    for e in summary["near_misses"][:30]:
        lines.append(f"### {e.get('id', '')}")
        lines.append(f"- gold: `{e.get('gold', '')}`")
        lines.append(f"- pred: `{e.get('pred', '')}`")
        lines.append(f"- char_accuracy: {float(e.get('char_accuracy', 0.0)):.3f}")
        lines.append(f"- edit_distance: {e.get('edit_distance', '')}")
        lines.append(f"- lengths gold/pred: {e.get('gold_len', '')}/{e.get('pred_len', '')}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8")[:5000])


if __name__ == "__main__":
    main()
