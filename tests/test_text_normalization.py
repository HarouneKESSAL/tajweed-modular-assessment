from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.text.normalization import normalize_arabic_text


def test_strip_diacritics():
    assert normalize_arabic_text("الرَّحْمَٰن") == "الرحمن"


def test_normalize_hamza_forms():
    assert normalize_arabic_text("إياك") == "اياك"
    assert normalize_arabic_text("أعوذ") == "اعوذ"


def test_normalize_spaces():
    assert normalize_arabic_text("من   شر  ما خلق") == "من شر ما خلق"


def test_keep_ta_marbuta_by_default():
    assert normalize_arabic_text("رحمة") == "رحمة"