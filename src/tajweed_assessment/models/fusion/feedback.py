from tajweed_assessment.models.fusion.schemas import DiagnosisReport

def render_feedback(report: DiagnosisReport) -> list[str]:
    messages: list[str] = []
    for error in report.errors:
        if error.type == "content_error":
            messages.append(
                f"At position {error.position}, the expected phoneme was {error.expected} but the recitation produced {error.predicted}."
            )
        elif error.type == "rule_error":
            detail = f" {error.detail}" if error.detail else ""
            messages.append(
                f"At position {error.position}, the rule {error.rule} was not applied correctly.{detail}"
            )
    if not messages:
        messages.append("No errors were detected.")
    return messages
