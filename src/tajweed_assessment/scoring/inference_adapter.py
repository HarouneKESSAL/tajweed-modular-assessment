from __future__ import annotations

from typing import Any

from tajweed_assessment.scoring.error_types import TajweedError
from tajweed_assessment.scoring.weighted_score import summarize_weighted_errors


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _infer_module(raw_error: Any) -> str:
    explicit = _get_value(raw_error, "module")
    if explicit:
        return str(explicit)

    rule = str(
        _get_value(raw_error, "rule", "")
        or _get_value(raw_error, "rule_type", "")
        or _get_value(raw_error, "expected_rule", "")
        or _get_value(raw_error, "error_type", "")
        or _get_value(raw_error, "type", "")
    ).lower()

    if any(token in rule for token in ("madd", "ghunnah", "duration")):
        return "duration"

    if any(token in rule for token in ("ikhfa", "idgham", "iqlab", "izhar", "transition")):
        return "transition"

    if any(token in rule for token in ("qalqalah", "burst")):
        return "burst"

    if any(token in rule for token in ("text", "word", "content", "letter")):
        return "content"

    return "unknown"


def _infer_error_type(raw_error: Any, module: str) -> str:
    explicit = (
        _get_value(raw_error, "error_type")
        or _get_value(raw_error, "type")
        or _get_value(raw_error, "label")
    )
    if explicit:
        return str(explicit)

    rule = str(
        _get_value(raw_error, "rule", "")
        or _get_value(raw_error, "rule_type", "")
        or _get_value(raw_error, "expected_rule", "")
    ).lower()

    if module == "content":
        return "wrong_word"

    if module == "duration":
        if "ghunnah" in rule:
            return "ghunnah_duration_error"
        if "madd" in rule:
            return "minor_madd_duration_error"
        return "minor_madd_duration_error"

    if module == "transition":
        if "ikhfa" in rule:
            return "weak_ikhfa"
        return "wrong_transition_rule"

    if module == "burst":
        return "missing_qalqalah"

    return "unknown"


def tajweed_errors_from_diagnosis(diagnosis_report: Any) -> list[TajweedError]:
    """
    Convert the existing inference diagnosis report into weighted-scoring errors.

    This is intentionally defensive because diagnosis errors may be dicts,
    dataclasses, or simple strings depending on the module.
    """
    raw_errors = _get_value(diagnosis_report, "errors", [])

    if raw_errors is None:
        raw_errors = []

    errors: list[TajweedError] = []

    for raw_error in raw_errors:
        if isinstance(raw_error, str):
            errors.append(
                TajweedError(
                    module="unknown",
                    error_type="unknown",
                    confidence=1.0,
                    message=raw_error,
                )
            )
            continue

        module = _infer_module(raw_error)
        error_type = _infer_error_type(raw_error, module)

        confidence = (
            _get_value(raw_error, "confidence")
            or _get_value(raw_error, "probability")
            or _get_value(raw_error, "score")
            or 1.0
        )

        expected = _get_value(raw_error, "expected")
        predicted = _get_value(raw_error, "predicted")
        location = (
            _get_value(raw_error, "location")
            or _get_value(raw_error, "position")
            or _get_value(raw_error, "span")
        )
        message = (
            _get_value(raw_error, "message")
            or _get_value(raw_error, "description")
            or _get_value(raw_error, "detail")
        )

        errors.append(
            TajweedError(
                module=str(module),
                error_type=str(error_type),
                confidence=float(confidence),
                location=None if location is None else str(location),
                expected=None if expected is None else str(expected),
                predicted=None if predicted is None else str(predicted),
                message=None if message is None else str(message),
            )
        )

    return errors


def score_diagnosis_report(
    diagnosis_report: Any,
    error_weight_config: dict[str, Any],
) -> dict[str, Any]:
    errors = tajweed_errors_from_diagnosis(diagnosis_report)
    return summarize_weighted_errors(errors, error_weight_config)