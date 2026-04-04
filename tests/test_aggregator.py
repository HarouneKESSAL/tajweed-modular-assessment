from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.labels import phoneme_to_id, rule_to_id
from tajweed_assessment.models.fusion.aggregator import aggregate_diagnosis

def test_aggregator_records_content_error():
    report = aggregate_diagnosis(
        word="Malik",
        canonical_phonemes=[phoneme_to_id["m"], phoneme_to_id["a"]],
        predicted_phonemes=[phoneme_to_id["m"], phoneme_to_id["i"]],
        canonical_rules=[rule_to_id["none"], rule_to_id["madd"]],
        module_judgments=[{"position": 1, "rule": "madd", "predicted_rule": "none", "is_correct": False, "detail": "too short"}],
    )
    assert any(e.type == "content_error" for e in report.errors)
