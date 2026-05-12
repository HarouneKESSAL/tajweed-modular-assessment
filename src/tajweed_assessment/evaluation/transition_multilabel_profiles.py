from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tajweed_assessment.inference.transition_multilabel import (
    TransitionMultiLabelPredictor,
    labels_to_combo,
    load_threshold_profile,
)
from tajweed_assessment.models.transition.multilabel_transition_module import (
    transition_multilabel_label_names,
    transition_rules_to_multihot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_gold_rules(row: dict[str, Any]) -> list[str]:
    rules = row.get("transition_multilabel_rules")
    if isinstance(rules, list):
        return [str(rule) for rule in rules]

    rules = row.get("transition_rules")
    if isinstance(rules, list):
        return [str(rule) for rule in rules]

    multihot = row.get("transition_multihot")
    if isinstance(multihot, list):
        label_names = transition_multilabel_label_names()
        return [
            label
            for label, value in zip(label_names, multihot)
            if float(value) >= 0.5
        ]

    return []


def rules_to_multihot(rules: list[str]) -> list[int]:
    values = transition_rules_to_multihot(
        rules,
        label_names=transition_multilabel_label_names(),
    )
    return [1 if float(value) >= 0.5 else 0 for value in values]


def compute_multilabel_metrics(
    gold_vectors: list[list[int]],
    pred_vectors: list[list[int]],
    pred_combos: list[str],
    gold_combos: list[str],
) -> dict[str, Any]:
    label_names = transition_multilabel_label_names()
    n = len(gold_vectors)

    if n == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_label": {},
            "predicted_combo_counts": {},
            "gold_combo_counts": {},
        }

    exact = sum(1 for gold, pred in zip(gold_vectors, pred_vectors) if gold == pred) / n

    per_label: dict[str, dict[str, Any]] = {}
    precisions = []
    recalls = []
    f1s = []

    for idx, label in enumerate(label_names):
        tp = sum(1 for g, p in zip(gold_vectors, pred_vectors) if g[idx] == 1 and p[idx] == 1)
        fp = sum(1 for g, p in zip(gold_vectors, pred_vectors) if g[idx] == 0 and p[idx] == 1)
        fn = sum(1 for g, p in zip(gold_vectors, pred_vectors) if g[idx] == 1 and p[idx] == 0)
        tn = sum(1 for g, p in zip(gold_vectors, pred_vectors) if g[idx] == 0 and p[idx] == 0)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_label[label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positives": sum(g[idx] for g in gold_vectors),
            "predicted_positive": sum(p[idx] for p in pred_vectors),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return {
        "samples": n,
        "exact_match": exact,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "per_label": per_label,
        "predicted_combo_counts": dict(Counter(pred_combos)),
        "gold_combo_counts": dict(Counter(gold_combos)),
    }


def evaluate_transition_multilabel_profiles(
    *,
    manifest_path: str | Path,
    threshold_config: str | Path = "configs/transition_multilabel_thresholds.yaml",
    profiles: list[str] | None = None,
    limit: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)
    if limit and limit > 0:
        rows = rows[:limit]

    if profiles is None or not profiles:
        profiles = ["gold_safe", "ikhfa_recall_safe", "merged_best", "retasy_extended_best"]

    label_names = transition_multilabel_label_names()
    results: dict[str, Any] = {}

    gold_rules_by_row = [row_gold_rules(row) for row in rows]
    gold_vectors = [rules_to_multihot(rules) for rules in gold_rules_by_row]
    gold_combos = [labels_to_combo(rules) for rules in gold_rules_by_row]

    for profile in profiles:
        checkpoint, thresholds = load_threshold_profile(
            threshold_config,
            profile=profile,
        )
        predictor = TransitionMultiLabelPredictor(
            checkpoint_path=checkpoint,
            thresholds=thresholds,
            device=device,
        )

        pred_vectors: list[list[int]] = []
        pred_combos: list[str] = []
        examples: list[dict[str, Any]] = []

        for index, row in enumerate(rows):
            prediction = predictor.predict(row["audio_path"])
            pred_rules = prediction.predicted_rules
            pred_vector = rules_to_multihot(pred_rules)

            pred_vectors.append(pred_vector)
            pred_combos.append(prediction.predicted_combo)

            if index < 20:
                examples.append(
                    {
                        "id": row.get("id") or row.get("sample_id"),
                        "text": row.get("text") or row.get("normalized_text"),
                        "gold_rules": gold_rules_by_row[index],
                        "predicted_rules": pred_rules,
                        "gold_combo": gold_combos[index],
                        "predicted_combo": prediction.predicted_combo,
                        "probabilities": prediction.probabilities,
                        "thresholds": prediction.thresholds,
                    }
                )

        metrics = compute_multilabel_metrics(
            gold_vectors=gold_vectors,
            pred_vectors=pred_vectors,
            pred_combos=pred_combos,
            gold_combos=gold_combos,
        )
        metrics["thresholds"] = thresholds
        metrics["checkpoint"] = checkpoint
        metrics["examples"] = examples
        results[profile] = metrics

    return {
        "manifest": str(resolve_path(manifest_path)),
        "threshold_config": str(resolve_path(threshold_config)),
        "profiles": profiles,
        "label_names": label_names,
        "samples": len(rows),
        "results": results,
    }


def save_transition_multilabel_profile_report(path: str | Path, result: dict[str, Any]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
