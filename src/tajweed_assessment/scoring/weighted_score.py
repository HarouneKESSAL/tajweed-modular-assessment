from __future__ import annotations

from pathlib import Path
from typing import Any
from tajweed_assessment.scoring.error_types import TajweedError
import yaml


def load_error_weights(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("error weight config must be a mapping")

    return config


def _clamp_confidence(value: Any) -> float:
    if value is None:
        return 1.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, confidence))


def _lookup_weight(config: dict[str, Any], module: str, error_type: str) -> tuple[float, str, str]:
    item = (
        config.get("categories", {})
        .get(module, {})
        .get(error_type, {})
    )

    if not isinstance(item, dict):
        return 1.0, "unknown", "unknown"

    return (
        float(item.get("weight", 1.0)),
        str(item.get("severity", "unknown")),
        str(item.get("lahn_type", "unknown")),
    )


def classify_rule_judgment(judgment: dict[str, Any]) -> tuple[str, str]:
    module = str(judgment.get("source_module") or "rule")
    expected = str(judgment.get("rule") or "none")
    predicted = str(judgment.get("predicted_rule") or "none")

    if module == "duration":
        if expected == "madd":
            return "duration", "severe_madd_error"
        if expected == "ghunnah":
            return "duration", "ghunnah_duration_error"
        return "duration", "minor_madd_duration_error"

    if module == "transition":
        if expected == "ikhfa" and predicted == "none":
            return "transition", "weak_ikhfa"
        if expected in {"idgham", "ikhfa"}:
            return "transition", "wrong_transition_rule"
        return "transition", "wrong_transition_rule"

    if module == "burst":
        if expected == "qalqalah" and predicted == "none":
            return "burst", "missing_qalqalah"
        return "burst", "weak_qalqalah"

    return module, "unknown_rule_error"


def classify_content_error(error: dict[str, Any]) -> tuple[str, str]:
    extra = error.get("extra") or {}
    if isinstance(extra, dict) and extra.get("weighted_error_type"):
        return "content", str(extra["weighted_error_type"])

    expected = str(error.get("expected") or "")
    predicted = str(error.get("predicted") or "")

    if expected == "<none>":
        return "content", "extra_word"
    if predicted == "<deleted>":
        return "content", "missing_word"
    return "content", "letter_substitution"


def score_inference_result(
    *,
    report: dict[str, Any],
    module_judgments: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    weighted_errors: list[dict[str, Any]] = []

    # Content errors come from the diagnosis report.
    for error in report.get("errors", []):
        if error.get("type") != "content_error":
            continue

        module, error_type = classify_content_error(error)
        extra = error.get("extra") or {}
        confidence = _clamp_confidence(extra.get("confidence") if isinstance(extra, dict) else 1.0)
        weight, severity, lahn_type = _lookup_weight(config, module, error_type)

        weighted_errors.append(
            {
                "module": module,
                "error_type": error_type,
                "severity": severity,
                "lahn_type": lahn_type,
                "position": error.get("position"),
                "expected": error.get("expected"),
                "predicted": error.get("predicted"),
                "confidence": confidence,
                "weight": weight,
                "weighted_penalty": weight * confidence,
            }
        )

    # Rule errors come from module_judgments, because they contain source_module/confidence.
    for judgment in module_judgments:
        if judgment.get("is_correct", False):
            continue

        module, error_type = classify_rule_judgment(judgment)
        confidence = _clamp_confidence(judgment.get("confidence"))
        weight, severity, lahn_type = _lookup_weight(config, module, error_type)

        weighted_errors.append(
            {
                "module": module,
                "error_type": error_type,
                "severity": severity,
                "lahn_type": lahn_type,
                "position": judgment.get("position"),
                "expected": judgment.get("rule"),
                "predicted": judgment.get("predicted_rule"),
                "confidence": confidence,
                "weight": weight,
                "weighted_penalty": weight * confidence,
                "detail": judgment.get("detail", ""),
            }
        )

    weighted_error_sum = sum(float(item["weighted_penalty"]) for item in weighted_errors)
    scale = float(config.get("scale", 3.0))
    final_score = max(0.0, 100.0 - weighted_error_sum * scale)

    severity_counts: dict[str, int] = {}
    for item in weighted_errors:
        severity = str(item.get("severity", "unknown"))
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    return {
        "score": final_score,
        "weighted_error_sum": weighted_error_sum,
        "scale": scale,
        "num_errors": len(weighted_errors),
        "severity_counts": severity_counts,
        "errors": weighted_errors,
    }
    
    
    



def get_error_weight(config: dict[str, Any], error: TajweedError) -> float:
    weight, _severity, _lahn_type = _lookup_weight(
        config,
        error.module,
        error.error_type,
    )
    return weight


def get_error_severity(config: dict[str, Any], error: TajweedError) -> str:
    _weight, severity, _lahn_type = _lookup_weight(
        config,
        error.module,
        error.error_type,
    )
    return severity


def get_error_lahn_type(config: dict[str, Any], error: TajweedError) -> str:
    _weight, _severity, lahn_type = _lookup_weight(
        config,
        error.module,
        error.error_type,
    )
    return lahn_type


def weighted_error_sum(errors: list[TajweedError], config: dict[str, Any]) -> float:
    total = 0.0

    for error in errors:
        confidence = _clamp_confidence(error.confidence)
        total += get_error_weight(config, error) * confidence

    return total


def final_score(errors: list[TajweedError], config: dict[str, Any]) -> float:
    scale = float(config.get("scale", 3.0))
    penalty = weighted_error_sum(errors, config) * scale
    return max(0.0, 100.0 - penalty)


def summarize_weighted_errors(
    errors: list[TajweedError],
    config: dict[str, Any],
) -> dict[str, Any]:
    weighted_errors: list[dict[str, Any]] = []
    severity_counts: dict[str, int] = {}

    for error in errors:
        confidence = _clamp_confidence(error.confidence)
        weight = get_error_weight(config, error)
        severity = get_error_severity(config, error)
        lahn_type = get_error_lahn_type(config, error)
        weighted_penalty = weight * confidence

        severity_counts[severity] = severity_counts.get(severity, 0) + 1

        weighted_errors.append(
            {
                "module": error.module,
                "error_type": error.error_type,
                "severity": severity,
                "lahn_type": lahn_type,
                "location": error.location,
                "expected": error.expected,
                "predicted": error.predicted,
                "message": error.message,
                "confidence": confidence,
                "weight": weight,
                "weighted_penalty": weighted_penalty,
            }
        )

    weighted_sum = sum(float(item["weighted_penalty"]) for item in weighted_errors)
    scale = float(config.get("scale", 3.0))
    score = max(0.0, 100.0 - weighted_sum * scale)

    return {
        "score": score,
        "weighted_error_sum": weighted_sum,
        "scale": scale,
        "error_count": len(weighted_errors),
        "num_errors": len(weighted_errors),
        "severity_counts": severity_counts,
        "errors": weighted_errors,
    }