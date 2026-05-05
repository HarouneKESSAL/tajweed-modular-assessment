
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from tajweed_assessment.scoring.inference_adapter import (
    score_diagnosis_report,
    tajweed_errors_from_diagnosis,
)


def test_empty_diagnosis_scores_100():
    config = {"scale": 3.0, "categories": {}}
    diagnosis = {"errors": []}

    summary = score_diagnosis_report(diagnosis, config)

    assert summary["score"] == 100.0
    assert summary["weighted_error_sum"] == 0.0
    assert summary["error_count"] == 0


def test_content_error_is_converted_and_weighted():
    config = {
        "scale": 3.0,
        "categories": {
            "content": {
                "wrong_word": {
                    "weight": 10.0,
                    "severity": "critical",
                }
            }
        },
    }

    diagnosis = {
        "errors": [
            {
                "module": "content",
                "error_type": "wrong_word",
                "expected": "الرحمن",
                "predicted": "الرحيم",
                "confidence": 1.0,
            }
        ]
    }

    errors = tajweed_errors_from_diagnosis(diagnosis)
    summary = score_diagnosis_report(diagnosis, config)

    assert len(errors) == 1
    assert errors[0].module == "content"
    assert errors[0].error_type == "wrong_word"
    assert summary["score"] == 70.0
    assert summary["weighted_error_sum"] == 10.0
    assert summary["severity_counts"]["critical"] == 1


def test_transition_error_is_inferred_from_rule():
    diagnosis = {
        "errors": [
            {
                "rule": "ikhfa",
                "expected": "ikhfa",
                "predicted": "none",
                "confidence": 0.5,
            }
        ]
    }

    errors = tajweed_errors_from_diagnosis(diagnosis)

    assert len(errors) == 1
    assert errors[0].module == "transition"
    assert errors[0].error_type == "weak_ikhfa"
    assert errors[0].confidence == 0.5