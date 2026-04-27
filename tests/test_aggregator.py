from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.data.labels import phoneme_to_id, rule_to_id
from tajweed_assessment.models.fusion.aggregator import aggregate_diagnosis
from tajweed_assessment.models.fusion.feedback import render_feedback
from tajweed_assessment.features.routing import build_routing_plan
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
import torch
import torch.nn as nn

def test_aggregator_records_content_error():
    report = aggregate_diagnosis(
        word="Malik",
        canonical_phonemes=[phoneme_to_id["m"], phoneme_to_id["a"]],
        predicted_phonemes=[phoneme_to_id["m"], phoneme_to_id["i"]],
        canonical_rules=[rule_to_id["none"], rule_to_id["madd"]],
        module_judgments=[{"position": 1, "rule": "madd", "predicted_rule": "none", "is_correct": False, "detail": "too short"}],
    )
    assert any(e.type == "content_error" for e in report.errors)

def test_routing_plan_selects_transition_and_burst():
    plan = build_routing_plan([rule_to_id["ikhfa"], rule_to_id["qalqalah"]])
    assert plan.use_transition is True
    assert plan.use_burst is True

def test_aggregator_keeps_rule_errors_without_canonical_content_track():
    report = aggregate_diagnosis(
        word="sample",
        canonical_phonemes=[],
        predicted_phonemes=[],
        canonical_rules=[rule_to_id["none"], rule_to_id["madd"]],
        module_judgments=[
            {"position": 1, "rule": "madd", "predicted_rule": "none", "is_correct": False, "detail": "expected madd but got none", "confidence": 0.42}
        ],
        canonical_chars=["ا", "م"],
    )
    assert len(report.errors) == 1
    assert report.errors[0].type == "rule_error"
    assert report.errors[0].extra["char"] == "م"
    assert report.errors[0].extra["confidence"] == 0.42

def test_feedback_includes_confidence_for_rule_errors():
    report = aggregate_diagnosis(
        word="sample",
        canonical_phonemes=[],
        predicted_phonemes=[],
        canonical_rules=[rule_to_id["madd"]],
        module_judgments=[
            {"position": 0, "rule": "madd", "predicted_rule": "none", "is_correct": False, "detail": "expected madd but got none", "confidence": 0.75}
        ],
        canonical_chars=["ن"],
    )
    feedback = render_feedback(report)
    assert '("ن")' in feedback[0]
    assert "Confidence=0.75." in feedback[0]


def test_feedback_includes_transition_localizer_disagreement_hint():
    report = aggregate_diagnosis(
        word="sample",
        canonical_phonemes=[],
        predicted_phonemes=[],
        canonical_rules=[rule_to_id["ikhfa"]],
        module_judgments=[
            {
                "position": 0,
                "rule": "ikhfa",
                "predicted_rule": "none",
                "is_correct": False,
                "detail": "expected ikhfa but got none",
                "confidence": 0.91,
                "source_module": "transition",
                "decision_source": "whole_verse_with_localized_evidence",
                "localized_clip_probability": 0.18,
                "localized_predicted_span_count": 0,
                "localized_predicted_labels": ["idgham"],
            }
        ],
        canonical_chars=["م"],
    )
    feedback = render_feedback(report)
    assert "Localized evidence suggested idgham" in feedback[0]
    assert "for ikhfa=0.18" in feedback[0]


def test_feedback_includes_duration_localizer_disagreement_hint():
    report = aggregate_diagnosis(
        word="sample",
        canonical_phonemes=[],
        predicted_phonemes=[],
        canonical_rules=[rule_to_id["ghunnah"]],
        module_judgments=[
            {
                "position": 0,
                "rule": "ghunnah",
                "predicted_rule": "madd",
                "is_correct": False,
                "detail": "expected ghunnah but got madd",
                "confidence": 0.88,
                "source_module": "duration",
                "decision_source": "sequence_with_localized_evidence",
                "localized_clip_probability": 0.11,
                "localized_predicted_span_count": 0,
                "localized_predicted_labels": ["madd"],
            }
        ],
        canonical_chars=["ن"],
    )
    feedback = render_feedback(report)
    assert "Localized evidence suggested madd" in feedback[0]
    assert "for ghunnah=0.11" in feedback[0]


class _FakeDurationModule(nn.Module):
    def forward(self, x, lengths):
        batch, time = x.size(0), x.size(1)
        log_probs = torch.zeros(batch, time, 11)
        rule_logits = torch.zeros(batch, time, len(rule_to_id))
        return log_probs, rule_logits


class _FakeProjectedDurationModule(nn.Module):
    def forward(self, x, lengths):
        batch, time = x.size(0), x.size(1)
        log_probs = torch.zeros(batch, time, 11)
        rule_logits = torch.zeros(batch, time, len(rule_to_id))
        rule_logits[:, :, rule_to_id["madd"]] = 5.0
        return log_probs, rule_logits


class _FakeWholeVerseTransition(nn.Module):
    def forward(self, mfcc, ssl, lengths):
        return torch.tensor([[0.1, 0.2, 3.0]], dtype=torch.float32)


class _FakeLocalizedDuration(nn.Module):
    def forward(self, x, lengths):
        logits = torch.full((1, x.size(1), 2), -5.0, dtype=torch.float32)
        logits[0, 2:5, 0] = 5.0
        return logits


class _FakeDurationFusionCalibrator(nn.Module):
    def forward(self, numeric_features, prev_char_ids, curr_char_ids, next_char_ids):
        return torch.tensor([[0.1, 3.0]], dtype=torch.float32)


class _FakeLocalizedTransition(nn.Module):
    def forward(self, x, lengths):
        logits = torch.full((1, x.size(1), 2), -5.0, dtype=torch.float32)
        logits[0, 2:5, 0] = 5.0
        return logits


def test_pipeline_keeps_whole_verse_transition_decision_and_adds_localized_evidence():
    pipeline = TajweedInferencePipeline(
        duration_module=_FakeDurationModule(),
        transition_module=_FakeWholeVerseTransition(),
        localized_transition_module=_FakeLocalizedTransition(),
        localized_transition_thresholds={"idgham": 0.7, "ikhfa": 0.75},
        localized_transition_labels=("idgham", "ikhfa"),
        device="cpu",
    )

    result = pipeline.run_modular(
        canonical_phonemes=[],
        canonical_rules=[rule_to_id["idgham"]],
        canonical_chars=["م"],
        word="sample",
        transition_mfcc=torch.randn(10, 39),
        transition_ssl=torch.randn(10, 64),
        localized_transition_x=torch.randn(10, 39),
    )

    assert result["routing_plan"]["use_transition"] is True
    assert result["routing_plan"]["use_transition_localizer"] is True
    judgment = result["module_judgments"][0]
    assert judgment["predicted_rule"] == "idgham"
    assert judgment["decision_source"] == "whole_verse_with_localized_evidence"
    assert judgment["localized_predicted_span_count"] == 1
    assert judgment["localized_clip_probability"] > 0.9


def test_pipeline_keeps_sequence_duration_decision_and_adds_localized_evidence():
    pipeline = TajweedInferencePipeline(
        duration_module=_FakeProjectedDurationModule(),
        localized_duration_module=_FakeLocalizedDuration(),
        localized_duration_thresholds={"ghunnah": 0.85, "madd": 0.45},
        localized_duration_labels=("ghunnah", "madd"),
        duration_localizer_override_threshold=1.1,
        device="cpu",
    )

    result = pipeline.run_modular(
        canonical_phonemes=[],
        canonical_rules=[rule_to_id["ghunnah"]],
        canonical_chars=["ن"],
        word="sample",
        duration_x=torch.randn(10, 39),
        localized_duration_x=torch.randn(10, 39),
    )

    assert result["routing_plan"]["use_duration"] is True
    assert result["routing_plan"]["use_duration_localizer"] is True
    judgment = result["module_judgments"][0]
    assert judgment["predicted_rule"] == "madd"
    assert judgment["decision_source"] == "sequence_with_localized_evidence"
    assert judgment["localized_predicted_span_count"] == 1
    assert judgment["localized_clip_probability"] > 0.9
    assert judgment["localized_predicted_labels"] == ["ghunnah"]


def test_pipeline_applies_conservative_duration_localizer_override_for_noon():
    pipeline = TajweedInferencePipeline(
        duration_module=_FakeProjectedDurationModule(),
        localized_duration_module=_FakeLocalizedDuration(),
        localized_duration_thresholds={"ghunnah": 0.85, "madd": 0.45},
        localized_duration_labels=("ghunnah", "madd"),
        duration_localizer_override_threshold=0.90,
        duration_localizer_override_chars=("ن",),
        device="cpu",
    )

    result = pipeline.run_modular(
        canonical_phonemes=[],
        canonical_rules=[rule_to_id["ghunnah"]],
        canonical_chars=["ن"],
        word="sample",
        duration_x=torch.randn(10, 39),
        localized_duration_x=torch.randn(10, 39),
    )

    judgment = result["module_judgments"][0]
    assert judgment["sequence_predicted_rule"] == "madd"
    assert judgment["predicted_rule"] == "ghunnah"
    assert judgment["decision_source"] == "sequence_overridden_by_localized_evidence"
    assert judgment["fusion_applied"] is True


def test_pipeline_prefers_learned_duration_fusion_over_rule_override():
    pipeline = TajweedInferencePipeline(
        duration_module=_FakeProjectedDurationModule(),
        duration_fusion_calibrator=_FakeDurationFusionCalibrator(),
        duration_fusion_char_vocab={"<pad>": 0, "<unk>": 1, "ن": 2},
        localized_duration_module=_FakeLocalizedDuration(),
        localized_duration_thresholds={"ghunnah": 0.85, "madd": 0.45},
        localized_duration_labels=("ghunnah", "madd"),
        duration_localizer_override_threshold=1.1,
        device="cpu",
    )

    result = pipeline.run_modular(
        canonical_phonemes=[],
        canonical_rules=[rule_to_id["ghunnah"]],
        canonical_chars=["ن"],
        word="sample",
        duration_x=torch.randn(10, 39),
        localized_duration_x=torch.randn(10, 39),
    )

    judgment = result["module_judgments"][0]
    assert judgment["predicted_rule"] == "ghunnah"
    assert judgment["sequence_predicted_rule"] == "madd"
    assert judgment["decision_source"] == "learned_duration_fusion"
    assert judgment["fusion_applied"] is True
