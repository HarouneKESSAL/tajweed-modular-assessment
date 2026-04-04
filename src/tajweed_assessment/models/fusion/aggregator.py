from typing import Iterable, List
from tajweed_assessment.data.labels import id_to_phoneme, id_to_rule
from tajweed_assessment.models.content.aligner import align_sequences
from tajweed_assessment.models.fusion.schemas import DiagnosisError, DiagnosisReport

def aggregate_diagnosis(word: str, canonical_phonemes: List[int], predicted_phonemes: List[int], canonical_rules: List[int], module_judgments: Iterable[dict]) -> DiagnosisReport:
    errors: List[DiagnosisError] = []
    alignment = align_sequences(canonical_phonemes, predicted_phonemes)

    surviving_positions = set()
    ref_pos = -1
    for step in alignment:
        if step["ref"] is not None:
            ref_pos += 1
        if step["type"] == "match":
            surviving_positions.add(ref_pos)
            continue
        if step["type"] == "substitution":
            errors.append(DiagnosisError(position=ref_pos, type="content_error", expected=id_to_phoneme[step["ref"]], predicted=id_to_phoneme[step["hyp"]]))
        elif step["type"] == "deletion":
            errors.append(DiagnosisError(position=ref_pos, type="content_error", expected=id_to_phoneme[step["ref"]], predicted="<deleted>"))
        elif step["type"] == "insertion":
            errors.append(DiagnosisError(position=ref_pos + 1, type="content_error", expected="<none>", predicted=id_to_phoneme[step["hyp"]]))

    for j in module_judgments:
        pos = int(j["position"])
        if pos not in surviving_positions:
            continue
        if j.get("is_correct", False):
            continue
        errors.append(
            DiagnosisError(
                position=pos,
                type="rule_error",
                rule=j["rule"],
                detail=j.get("detail", ""),
                expected=id_to_rule.get(canonical_rules[pos], "none") if pos < len(canonical_rules) else None,
                predicted=j.get("predicted_rule"),
                extra={k: v for k, v in j.items() if k not in {"position", "rule", "detail", "predicted_rule", "is_correct"}},
            )
        )

    return DiagnosisReport(
        word=word,
        canonical_phonemes=[id_to_phoneme[p] for p in canonical_phonemes],
        predicted_phonemes=[id_to_phoneme[p] for p in predicted_phonemes],
        errors=errors,
    )
