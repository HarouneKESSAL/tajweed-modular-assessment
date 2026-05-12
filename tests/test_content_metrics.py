from pathlib import Path
import sys

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.evaluation.content_metrics import (
    char_accuracy,
    compute_content_metrics,
    levenshtein,
)


def test_levenshtein_basic():
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("abc", "ab") == 1
    assert levenshtein("abc", "adc") == 1



def test_char_accuracy():
    assert char_accuracy("abc", "abc") == 1.0
    assert char_accuracy("ab", "abc") == pytest.approx(2 / 3)


def test_normalized_exact_can_be_higher_than_strict():
    metrics = compute_content_metrics(["الرحمن"], ["الرَّحْمَٰن"])
    assert metrics.exact_match == 0.0
    assert metrics.normalized_exact_match == 1.0