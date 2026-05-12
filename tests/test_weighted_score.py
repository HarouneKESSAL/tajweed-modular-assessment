from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.scoring.error_types import TajweedError


import pytest

from tajweed_assessment.scoring.weighted_score import score_inference_result


def test_content_error_penalizes_more_than_rule_error():
    config = {
        "scale": 3.0,
        "categories": {
            "content": {
                "letter_substitution": {
                    "weight": 7.0,
                    "severity": "major",
                    "lahn_type": "jali",
                }
            },
            "duration": {
                "minor_madd_duration_error": {
                    "weight": 2.0,
                    "severity": "minor",
                    "lahn_type": "khafi",
                }
            },
        },
    }

    content_result = score_inference_result(
        report={
            "errors": [
                {
                    "type": "content_error",
                    "position": 0,
                    "expected": "a",
                    "predicted": "b",
                    "extra": {"weighted_error_type": "letter_substitution"},
                }
            ]
        },
        module_judgments=[],
        config=config,
    )

    rule_result = score_inference_result(
        report={"errors": []},
        module_judgments=[
            {
                "source_module": "duration",
                "position": 0,
                "rule": "madd",
                "predicted_rule": "none",
                "is_correct": False,
                "confidence": 1.0,
            }
        ],
        config=config,
    )

    assert content_result["weighted_error_sum"] > rule_result["weighted_error_sum"]
    assert content_result["score"] < rule_result["score"]


def test_confidence_reduces_penalty():
    config = {
        "scale": 3.0,
        "categories": {
            "transition": {
                "wrong_transition_rule": {
                    "weight": 4.0,
                    "severity": "medium",
                    "lahn_type": "khafi",
                }
            }
        },
    }

    high = score_inference_result(
        report={"errors": []},
        module_judgments=[
            {
                "source_module": "transition",
                "position": 0,
                "rule": "ikhfa",
                "predicted_rule": "idgham",
                "is_correct": False,
                "confidence": 1.0,
            }
        ],
        config=config,
    )

    low = score_inference_result(
        report={"errors": []},
        module_judgments=[
            {
                "source_module": "transition",
                "position": 0,
                "rule": "ikhfa",
                "predicted_rule": "idgham",
                "is_correct": False,
                "confidence": 0.25,
            }
        ],
        config=config,
    )

    assert high["weighted_error_sum"] > low["weighted_error_sum"]
    assert high["score"] < low["score"]


def test_no_errors_gives_100():
    result = score_inference_result(
        report={"errors": []},
        module_judgments=[],
        config={"scale": 3.0, "categories": {}},
    )

    assert result["score"] == pytest.approx(100.0)
    assert result["weighted_error_sum"] == pytest.approx(0.0)