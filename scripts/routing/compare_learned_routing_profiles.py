from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.inference.learned_routing import (
    load_learned_routing_predictor_from_config,
)


TARGET_NAMES = ["use_duration", "use_transition", "use_burst"]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def combo(plan: dict[str, bool]) -> str:
    active = [name for name in TARGET_NAMES if plan.get(name, False)]
    return "+".join(active) if active else "none"


def collect_rule_strings(row: dict[str, Any]) -> list[str]:
    rules: list[str] = []

    for key in [
        "canonical_rules",
        "transition_rules",
        "transition_multilabel_rules",
        "rules",
    ]:
        value = row.get(key)
        if isinstance(value, list):
            rules.extend(str(item).lower() for item in value)
        elif value:
            rules.append(str(value).lower())

    spans = row.get("rule_spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict) and span.get("rule"):
                rules.append(str(span["rule"]).lower())

    return rules


def current_plan_from_row(row: dict[str, Any]) -> dict[str, bool]:
    # Preferred: trusted/current routing targets already materialized.
    if isinstance(row.get("target_names"), list) and isinstance(row.get("targets"), list):
        return {
            str(name): bool(int(value))
            for name, value in zip(row["target_names"], row["targets"])
            if str(name) in TARGET_NAMES
        }

    # If a previous tool saved a routing plan directly.
    if isinstance(row.get("routing_plan"), dict):
        return {
            name: bool(row["routing_plan"].get(name, False))
            for name in TARGET_NAMES
        }

    # Fallback: derive from row metadata. This is useful for older manifests.
    rules = collect_rule_strings(row)

    use_duration = any(("madd" in rule or "ghunnah" in rule) for rule in rules)

    multihot = row.get("transition_multihot")
    use_transition = False
    if isinstance(multihot, list) and any(float(x) >= 0.5 for x in multihot):
        use_transition = True
    if row.get("transition_rules"):
        use_transition = True
    if any(("ikhfa" in rule or "idgham" in rule) for rule in rules):
        use_transition = True

    use_burst = False
    if "burst_label" in row:
        try:
            use_burst = bool(int(row.get("burst_label") or 0))
        except (TypeError, ValueError):
            use_burst = False
    if any("qalqalah" in rule for rule in rules):
        use_burst = True

    return {
        "use_duration": use_duration,
        "use_transition": use_transition,
        "use_burst": use_burst,
    }


def normalize_plan(plan: dict[str, bool]) -> dict[str, bool]:
    return {name: bool(plan.get(name, False)) for name in TARGET_NAMES}


def metrics_against_current(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    max_examples: int,
) -> dict[str, Any]:
    counts = {
        name: {"tp": 0, "fp_extra": 0, "fn_missed": 0, "tn": 0}
        for name in TARGET_NAMES
    }

    exact = 0
    current_combo_counts = Counter()
    learned_combo_counts = Counter()
    disagreement_type_counts = Counter()
    examples = []

    for row, pred_payload in zip(rows, predictions):
        current = normalize_plan(current_plan_from_row(row))
        learned = normalize_plan(pred_payload["routing_plan"])

        current_combo_counts[combo(current)] += 1
        learned_combo_counts[combo(learned)] += 1

        if current == learned:
            exact += 1
        else:
            missed = [name for name in TARGET_NAMES if current[name] and not learned[name]]
            extra = [name for name in TARGET_NAMES if learned[name] and not current[name]]

            if missed and extra:
                disagreement_type = "missed_and_extra"
            elif missed:
                disagreement_type = "missed_official_module"
            elif extra:
                disagreement_type = "extra_learned_module"
            else:
                disagreement_type = "other"

            disagreement_type_counts[disagreement_type] += 1

            if len(examples) < max_examples:
                examples.append(
                    {
                        "id": row.get("id") or row.get("sample_id"),
                        "text": row.get("text") or row.get("normalized_text") or row.get("source_text") or "",
                        "label_source": row.get("label_source", ""),
                        "sources": row.get("sources", []),
                        "current": current,
                        "learned": learned,
                        "probabilities": pred_payload.get("probabilities", {}),
                        "missed": missed,
                        "extra": extra,
                    }
                )

        for name in TARGET_NAMES:
            c = current[name]
            p = learned[name]

            if c and p:
                counts[name]["tp"] += 1
            elif (not c) and p:
                counts[name]["fp_extra"] += 1
            elif c and (not p):
                counts[name]["fn_missed"] += 1
            else:
                counts[name]["tn"] += 1

    per_label = {}
    f1s = []

    for name, c in counts.items():
        tp = c["tp"]
        fp = c["fp_extra"]
        fn = c["fn_missed"]
        tn = c["tn"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / max(1, tp + fp + fn + tn)

        f1s.append(f1)
        per_label[name] = {
            "accuracy": accuracy,
            "precision_vs_current": precision,
            "recall_vs_current": recall,
            "f1_vs_current": f1,
            "official_positive": tp + fn,
            "learned_positive": tp + fp,
            "tp_agree_on": tp,
            "fp_extra": fp,
            "fn_missed": fn,
            "tn_agree_off": tn,
        }

    return {
        "samples": len(rows),
        "exact_agreement": exact / max(1, len(rows)),
        "macro_f1_vs_current": sum(f1s) / len(f1s),
        "per_label": per_label,
        "current_combo_counts": dict(current_combo_counts),
        "learned_combo_counts": dict(learned_combo_counts),
        "disagreement_type_counts": dict(disagreement_type_counts),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--threshold-config", default="configs/learned_router_v5_thresholds.yaml")
    parser.add_argument("--profiles", nargs="+", default=["trusted_retasy_calibrated", "weak_policy_tuned"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.dataset))
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]

    result: dict[str, Any] = {
        "dataset": str(resolve_path(args.dataset)),
        "split": args.split,
        "threshold_config": str(resolve_path(args.threshold_config)),
        "profiles": args.profiles,
        "samples": len(rows),
        "results": {},
    }

    for profile in args.profiles:
        predictor = load_learned_routing_predictor_from_config(
            config_path=args.threshold_config,
            profile=profile,
            device=args.device,
        )

        predictions = []
        for row in rows:
            text = row.get("text") or row.get("normalized_text") or row.get("source_text") or ""
            pred = predictor.predict(
                audio_path=row.get("audio_path", ""),
                text=text,
            )
            predictions.append(pred.to_dict())

        result["results"][profile] = metrics_against_current(
            rows=rows,
            predictions=predictions,
            max_examples=args.max_examples,
        )

    out = resolve_path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Learned routing profile comparison")
    print("----------------------------------")
    print(f"dataset: {resolve_path(args.dataset)}")
    print(f"samples: {len(rows)}")
    for profile, metrics in result["results"].items():
        print()
        print(f"Profile: {profile}")
        print(f"- exact_agreement={metrics['exact_agreement']:.3f}")
        print(f"- macro_f1_vs_current={metrics['macro_f1_vs_current']:.3f}")
        print(f"- current_combo_counts={metrics['current_combo_counts']}")
        print(f"- learned_combo_counts={metrics['learned_combo_counts']}")
        print(f"- disagreement_type_counts={metrics['disagreement_type_counts']}")
        for name, item in metrics["per_label"].items():
            print(
                f"  - {name}: "
                f"f1={item['f1_vs_current']:.3f} "
                f"precision={item['precision_vs_current']:.3f} "
                f"recall={item['recall_vs_current']:.3f} "
                f"extra={item['fp_extra']} missed={item['fn_missed']}"
            )

    print()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
