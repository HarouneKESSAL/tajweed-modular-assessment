from tajweed_assessment.models.fusion.schemas import DiagnosisReport


def _render_duration_localizer_hint(error) -> str:
    extra = error.extra or {}
    if extra.get("source_module") != "duration":
        return ""
    if extra.get("decision_source") not in {
        "sequence_with_localized_evidence",
        "sequence_overridden_by_localized_evidence",
        "learned_duration_fusion",
    }:
        return ""

    localized_labels = extra.get("localized_predicted_labels") or []
    localized_span_count = extra.get("localized_predicted_span_count")
    localized_prob = extra.get("localized_clip_probability")
    expected_rule = error.rule

    if localized_labels:
        labels_text = ", ".join(str(label) for label in localized_labels)
        hint = f" Localized evidence suggested {labels_text}"
        if isinstance(localized_span_count, int):
            hint += f" with {localized_span_count} span"
            if localized_span_count != 1:
                hint += "s"
        if isinstance(localized_prob, (int, float)):
            hint += f" for {expected_rule}={localized_prob:.2f}"
        return hint + "."

    if isinstance(localized_prob, (int, float)):
        return f" Localized evidence found no {expected_rule} span (p={localized_prob:.2f})."

    return ""


def _render_transition_localizer_hint(error) -> str:
    extra = error.extra or {}
    if extra.get("source_module") != "transition":
        return ""
    if extra.get("decision_source") != "whole_verse_with_localized_evidence":
        return ""

    localized_labels = extra.get("localized_predicted_labels") or []
    localized_span_count = extra.get("localized_predicted_span_count")
    localized_prob = extra.get("localized_clip_probability")
    expected_rule = error.rule

    if localized_labels:
        labels_text = ", ".join(str(label) for label in localized_labels)
        hint = f" Localized evidence suggested {labels_text}"
        if isinstance(localized_span_count, int):
            hint += f" with {localized_span_count} span"
            if localized_span_count != 1:
                hint += "s"
        if isinstance(localized_prob, (int, float)):
            hint += f" for {expected_rule}={localized_prob:.2f}"
        return hint + "."

    if isinstance(localized_prob, (int, float)):
        return f" Localized evidence found no {expected_rule} span (p={localized_prob:.2f})."

    return ""


def render_feedback(report: DiagnosisReport) -> list[str]:
    messages: list[str] = []
    for error in report.errors:
        if error.type == "content_error":
            messages.append(
                f"At position {error.position}, the expected phoneme was {error.expected} but the recitation produced {error.predicted}."
            )
        elif error.type == "rule_error":
            detail = f" {error.detail}" if error.detail else ""
            char = error.extra.get("char") if error.extra else None
            char_text = f' ("{char}")' if char else ""
            confidence = error.extra.get("confidence") if error.extra else None
            confidence_text = f" Confidence={confidence:.2f}." if isinstance(confidence, (int, float)) else ""
            duration_hint = _render_duration_localizer_hint(error)
            transition_hint = _render_transition_localizer_hint(error)
            messages.append(
                f"At position {error.position}{char_text}, the rule {error.rule} was not applied correctly.{detail}{confidence_text}{duration_hint}{transition_hint}"
            )
    if not messages:
        messages.append("No errors were detected.")
    return messages
