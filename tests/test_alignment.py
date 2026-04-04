from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.labels import phoneme_to_id
from tajweed_assessment.models.content.aligner import align_sequences

def test_alignment_has_substitution():
    ref = [phoneme_to_id["m"], phoneme_to_id["a"], phoneme_to_id["l"]]
    hyp = [phoneme_to_id["m"], phoneme_to_id["i"], phoneme_to_id["l"]]
    ops = align_sequences(ref, hyp)
    assert any(step["type"] == "substitution" for step in ops)
