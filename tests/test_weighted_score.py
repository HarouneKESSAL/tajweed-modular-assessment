from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.scoring.error_types import TajweedError
from tajweed_assessment.scoring.weighted_score import final_score, weighted_error_sum


def test_content_error_penalizes_more_than_minor_duration():
    config = {
        "scale": 3.0,
        "categories": {
            "content": {"wrong_word": {"weight": 10.0}},
            "duration": {"minor_madd_duration_error": {"weight": 2.0}},
        },
    }

    content = [TajweedError(module="content", error_type="wrong_word", confidence=1.0)]
    duration = [TajweedError(module="duration", error_type="minor_madd_duration_error", confidence=1.0)]

    assert weighted_error_sum(content, config) > weighted_error_sum(duration, config)
    assert final_score(content, config) < final_score(duration, config)


def test_confidence_reduces_penalty():
    config = {"categories": {"content": {"wrong_word": {"weight": 10.0}}}}
    high = [TajweedError(module="content", error_type="wrong_word", confidence=1.0)]
    low = [TajweedError(module="content", error_type="wrong_word", confidence=0.2)]
    assert weighted_error_sum(high, config) > weighted_error_sum(low, config)