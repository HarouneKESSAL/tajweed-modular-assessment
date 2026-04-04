from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN

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
