from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.dataset import ToyDurationDataset, ToyTransitionDataset, ToyBurstDataset
from tajweed_assessment.data.localized_duration_dataset import normalize_duration_label

def test_toy_duration_shapes():
    s = ToyDurationDataset(n_samples=2)[0]
    assert s["x"].dim() == 2
    assert s["phoneme_targets"].dim() == 1
    assert s["rule_targets"].dim() == 1

def test_toy_transition_shapes():
    s = ToyTransitionDataset(n_samples=2)[0]
    assert s["mfcc"].dim() == 2
    assert s["ssl"].dim() == 2

def test_toy_burst_shapes():
    s = ToyBurstDataset(n_samples=2)[0]
    assert s["x"].dim() == 2

def test_normalize_duration_label_maps_madd_and_ghunnah():
    assert normalize_duration_label({"rule": "ghunnah", "coarse_group": "ghunnah"}) == "ghunnah"
    assert normalize_duration_label({"rule": "madd_246", "coarse_group": "has_madd"}) == "madd"
    assert normalize_duration_label({"rule": "none"}) is None
