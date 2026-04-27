from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from tajweed_assessment.data.manifests import ManifestEntry
from tajweed_assessment.data.labels import rule_to_id
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.models.common.losses import build_rule_class_weights
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from duration.train_duration import build_duration_sample_weights
from transition.train_transition import build_transition_sample_weights

def test_duration_forward():
    model = DurationRuleModule()
    x = torch.randn(2, 12, 39)
    lengths = torch.tensor([12, 10])
    log_probs, rule_logits = model(x, lengths)
    assert log_probs.shape[:2] == rule_logits.shape[:2]
    assert log_probs.size(-1) == 11

def test_transition_forward():
    model = TransitionRuleModule()
    mfcc = torch.randn(2, 24, 39)
    ssl = torch.randn(2, 24, 64)
    lengths = torch.tensor([24, 24])
    logits = model(mfcc, ssl, lengths)
    assert logits.shape == (2, 3)

def test_burst_forward():
    model = QalqalahCNN()
    x = torch.randn(2, 24, 39)
    logits = model(x)
    assert logits.shape == (2, 2)

def test_build_rule_class_weights_applies_overrides():
    weights = build_rule_class_weights({"ghunnah": 4.0, "madd": 1.5})
    assert weights is not None
    assert weights[rule_to_id["ghunnah"]].item() == 4.0
    assert weights[rule_to_id["madd"]].item() == 1.5
    assert weights[rule_to_id["none"]].item() == 1.0

def test_dummy_ssl_feature_extractor_is_deterministic():
    mfcc = torch.randn(12, 13)
    extractor_a = DummySSLFeatureExtractor(output_dim=64)
    extractor_b = DummySSLFeatureExtractor(output_dim=64)
    out_a = extractor_a.from_mfcc(mfcc)
    out_b = extractor_b.from_mfcc(mfcc)
    assert torch.allclose(out_a, out_b)

def test_build_transition_sample_weights_applies_hardcase_overrides():
    entries = [
        ManifestEntry(sample_id="a", audio_path="a.wav", canonical_rules=["none"]),
        ManifestEntry(sample_id="b", audio_path="b.wav", canonical_rules=["ikhfa"]),
    ]
    weights = build_transition_sample_weights(entries, [0, 1], {"b": 4.5})
    assert weights == [1.0, 4.5]

def test_build_duration_sample_weights_applies_hardcase_overrides():
    entries = [
        ManifestEntry(sample_id="a", audio_path="a.wav", canonical_rules=["madd"]),
        ManifestEntry(sample_id="b", audio_path="b.wav", canonical_rules=["ghunnah"]),
    ]
    weights = build_duration_sample_weights(entries, [0, 1], {"b": 5.0})
    assert weights == [1.0, 5.0]
