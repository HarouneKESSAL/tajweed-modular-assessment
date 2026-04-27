from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from tajweed_assessment.data.labels import TRANSITION_RULES, id_to_rule
from tajweed_assessment.models.common.decoding import decode_with_majority_rules
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.features.routing import build_routing_plan
from tajweed_assessment.models.fusion.aggregator import aggregate_diagnosis
from tajweed_assessment.models.fusion.duration_fusion_calibrator import (
    DURATION_FUSION_LABELS,
    build_duration_fusion_numeric_features,
    encode_duration_context_chars,
)
from tajweed_assessment.models.fusion.feedback import render_feedback


def _decode_duration_rule_confidences(
    log_probs: torch.Tensor,
    rule_logits: torch.Tensor,
    length: int,
) -> tuple[list[int], list[int], list[float]]:
    pred_phonemes, pred_rules = decode_with_majority_rules(log_probs, rule_logits, length)
    path = log_probs.argmax(dim=-1)[:length].tolist()
    rule_path = rule_logits.argmax(dim=-1)[:length].tolist()
    rule_probs = rule_logits.softmax(dim=-1)[:length]

    grouped_scores: list[list[float]] = []
    prev_blank = True
    prev_pid = None
    for frame_idx, (pid, rid) in enumerate(zip(path, rule_path)):
        if pid == 0:
            prev_blank = True
            prev_pid = None
            continue
        if prev_blank or pid != prev_pid:
            grouped_scores.append([float(rule_probs[frame_idx, rid].item())])
        else:
            grouped_scores[-1].append(float(rule_probs[frame_idx, rid].item()))
        prev_blank = False
        prev_pid = pid

    confidences = [sum(scores) / len(scores) for scores in grouped_scores if scores]
    return pred_phonemes, pred_rules, confidences


def _project_duration_rules_to_positions(
    rule_logits: torch.Tensor,
    length: int,
    num_positions: int,
) -> tuple[list[int], list[float]]:
    if num_positions <= 0:
        return [], []

    frame_probs = rule_logits.softmax(dim=-1)[:length]
    projected_rules: list[int] = []
    projected_confidences: list[float] = []

    for idx in range(num_positions):
        start = int(idx * length / num_positions)
        end = int((idx + 1) * length / num_positions)
        if end <= start:
            end = min(length, start + 1)
        span = frame_probs[start:end]
        if span.numel() == 0:
            projected_rules.append(0)
            projected_confidences.append(0.0)
            continue
        mean_probs = span.mean(dim=0)
        pred = int(mean_probs.argmax().item())
        projected_rules.append(pred)
        projected_confidences.append(float(mean_probs[pred].item()))

    return projected_rules, projected_confidences


def _decode_transition_prediction(
    logits: torch.Tensor,
    thresholds: dict[str, float] | None = None,
) -> tuple[int, float]:
    probs = logits.softmax(dim=-1)[0]
    if not thresholds:
        pred_idx = int(logits.argmax(dim=-1)[0].item())
        return pred_idx, float(probs[pred_idx].item())

    non_none_probs = probs[1:]
    best_offset = int(non_none_probs.argmax().item())
    pred_idx = best_offset + 1
    pred_name = TRANSITION_RULES[pred_idx] if 0 <= pred_idx < len(TRANSITION_RULES) else "none"
    pred_confidence = float(probs[pred_idx].item())
    threshold = float(thresholds.get(pred_name, 0.5))

    if pred_confidence >= threshold:
        return pred_idx, pred_confidence
    return 0, float(probs[0].item())


def _contiguous_transition_spans_from_probs(
    probs: torch.Tensor,
    threshold: float,
    frame_hop_sec: float,
    label: str,
) -> list[dict]:
    pred = probs >= threshold
    spans: list[dict] = []
    start = None
    for i, flag in enumerate(pred.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            seg = probs[start:end]
            spans.append(
                {
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": float(start * frame_hop_sec),
                    "end_sec": float(end * frame_hop_sec),
                    "max_prob": float(seg.max().item()),
                    "mean_prob": float(seg.mean().item()),
                }
            )
            start = None
    if start is not None:
        end = len(pred)
        seg = probs[start:end]
        spans.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_sec": float(start * frame_hop_sec),
                "end_sec": float(end * frame_hop_sec),
                "max_prob": float(seg.max().item()),
                "mean_prob": float(seg.mean().item()),
            }
        )
    return spans


def _contiguous_duration_spans_from_probs(
    probs: torch.Tensor,
    threshold: float,
    frame_hop_sec: float,
    label: str,
) -> list[dict]:
    pred = probs >= threshold
    spans: list[dict] = []
    start = None
    for i, flag in enumerate(pred.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            seg = probs[start:end]
            spans.append(
                {
                    "label": label,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": float(start * frame_hop_sec),
                    "end_sec": float(end * frame_hop_sec),
                    "max_prob": float(seg.max().item()),
                    "mean_prob": float(seg.mean().item()),
                }
            )
            start = None
    if start is not None:
        end = len(pred)
        seg = probs[start:end]
        spans.append(
            {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "start_sec": float(start * frame_hop_sec),
                "end_sec": float(end * frame_hop_sec),
                "max_prob": float(seg.max().item()),
                "mean_prob": float(seg.mean().item()),
            }
        )
    return spans


def _decode_localized_transition_evidence(
    logits: torch.Tensor,
    length: int,
    labels: tuple[str, ...],
    thresholds: dict[str, float] | None = None,
    frame_hop_sec: float = 0.01,
) -> dict:
    probs = torch.sigmoid(logits[:length])
    clip_probs = probs.max(dim=0).values
    clip_probabilities: dict[str, float] = {}
    predicted_labels: list[str] = []
    predicted_spans: dict[str, list[dict]] = {}

    for idx, label in enumerate(labels):
        threshold = float((thresholds or {}).get(label, 0.5))
        clip_probability = float(clip_probs[idx].item())
        label_spans = _contiguous_transition_spans_from_probs(probs[:, idx], threshold, frame_hop_sec, label)
        clip_probabilities[label] = clip_probability
        predicted_spans[label] = label_spans
        if clip_probability >= threshold:
            predicted_labels.append(label)

    return {
        "clip_probabilities": clip_probabilities,
        "predicted_labels": predicted_labels,
        "predicted_spans": predicted_spans,
        "thresholds": {label: float((thresholds or {}).get(label, 0.5)) for label in labels},
    }


def _decode_localized_duration_evidence(
    logits: torch.Tensor,
    length: int,
    labels: tuple[str, ...],
    thresholds: dict[str, float] | None = None,
    frame_hop_sec: float = 0.01,
) -> dict:
    probs = torch.sigmoid(logits[:length])
    clip_probs = probs.max(dim=0).values
    clip_probabilities: dict[str, float] = {}
    predicted_labels: list[str] = []
    predicted_spans: dict[str, list[dict]] = {}

    for idx, label in enumerate(labels):
        threshold = float((thresholds or {}).get(label, 0.5))
        clip_probability = float(clip_probs[idx].item())
        label_spans = _contiguous_duration_spans_from_probs(probs[:, idx], threshold, frame_hop_sec, label)
        clip_probabilities[label] = clip_probability
        predicted_spans[label] = label_spans
        if clip_probability >= threshold:
            predicted_labels.append(label)

    return {
        "clip_probabilities": clip_probabilities,
        "predicted_labels": predicted_labels,
        "predicted_spans": predicted_spans,
        "thresholds": {label: float((thresholds or {}).get(label, 0.5)) for label in labels},
    }


def _maybe_override_duration_prediction(
    *,
    sequence_pred_name: str,
    localized_evidence: dict | None,
    canonical_chars: list[str] | None,
    position: int,
    override_threshold: float,
    override_chars: tuple[str, ...],
) -> tuple[str, bool, str | None]:
    if sequence_pred_name != "madd" or localized_evidence is None:
        return sequence_pred_name, False, None
    if not canonical_chars or position >= len(canonical_chars):
        return sequence_pred_name, False, None

    char = str(canonical_chars[position] or "")
    if char not in override_chars:
        return sequence_pred_name, False, None

    clip_probabilities = localized_evidence.get("clip_probabilities") or {}
    ghunnah_prob = float(clip_probabilities.get("ghunnah", 0.0))
    predicted_labels = localized_evidence.get("predicted_labels") or []
    if "ghunnah" not in predicted_labels:
        return sequence_pred_name, False, None
    if ghunnah_prob < override_threshold:
        return sequence_pred_name, False, None

    return "ghunnah", True, f"localized ghunnah override on {char} at p={ghunnah_prob:.2f}"


def _maybe_calibrate_duration_prediction(
    *,
    sequence_pred_name: str,
    sequence_confidence: float | None,
    localized_evidence: dict | None,
    canonical_chars: list[str] | None,
    position: int,
    calibrator: nn.Module | None,
    char_vocab: dict[str, int] | None,
    device: str,
) -> tuple[str, bool, str | None, dict[str, float] | None]:
    if calibrator is None or localized_evidence is None or canonical_chars is None:
        return sequence_pred_name, False, None, None

    numeric_features = build_duration_fusion_numeric_features(
        sequence_predicted_rule=sequence_pred_name,
        sequence_confidence=sequence_confidence,
        localized_clip_probabilities=localized_evidence.get("clip_probabilities"),
        localized_predicted_labels=localized_evidence.get("predicted_labels"),
    )
    prev_id, curr_id, next_id = encode_duration_context_chars(canonical_chars, position, char_vocab)
    with torch.no_grad():
        logits = calibrator(
            torch.tensor([numeric_features], dtype=torch.float32, device=device),
            torch.tensor([prev_id], dtype=torch.long, device=device),
            torch.tensor([curr_id], dtype=torch.long, device=device),
            torch.tensor([next_id], dtype=torch.long, device=device),
        )
        probs = logits.softmax(dim=-1)[0].cpu()
    predicted_idx = int(probs.argmax().item())
    predicted_rule = DURATION_FUSION_LABELS[predicted_idx]
    probabilities = {label: float(probs[idx].item()) for idx, label in enumerate(DURATION_FUSION_LABELS)}
    calibrated = predicted_rule != sequence_pred_name
    reason = None
    if calibrated:
        reason = f"learned duration fusion selected {predicted_rule}"
    return predicted_rule, calibrated, reason, probabilities


@dataclass
class TajweedInferencePipeline:
    duration_module: DurationRuleModule
    transition_module: Optional[TransitionRuleModule] = None
    duration_fusion_calibrator: Optional[nn.Module] = None
    duration_fusion_char_vocab: Optional[dict[str, int]] = None
    localized_duration_module: Optional[nn.Module] = None
    localized_transition_module: Optional[nn.Module] = None
    burst_module: Optional[QalqalahCNN] = None
    transition_thresholds: Optional[dict[str, float]] = None
    localized_duration_thresholds: Optional[dict[str, float]] = None
    localized_transition_thresholds: Optional[dict[str, float]] = None
    localized_duration_labels: tuple[str, ...] = ("ghunnah", "madd")
    localized_transition_labels: tuple[str, ...] = ("idgham", "ikhfa")
    localized_duration_frame_hop_sec: float = 0.01
    localized_transition_frame_hop_sec: float = 0.01
    duration_localizer_override_threshold: float = 0.98
    duration_localizer_override_chars: tuple[str, ...] = ("ن",)
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.duration_module.to(self.device).eval()
        if self.transition_module is not None:
            self.transition_module.to(self.device).eval()
        if self.duration_fusion_calibrator is not None:
            self.duration_fusion_calibrator.to(self.device).eval()
        if self.localized_duration_module is not None:
            self.localized_duration_module.to(self.device).eval()
        if self.localized_transition_module is not None:
            self.localized_transition_module.to(self.device).eval()
        if self.burst_module is not None:
            self.burst_module.to(self.device).eval()

    @torch.no_grad()
    def run_duration_only(self, x: torch.Tensor, input_length: int, canonical_phonemes: list[int], canonical_rules: list[int], word: str = "sample") -> dict:
        batch_x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([input_length], dtype=torch.long, device=self.device)
        log_probs, rule_logits = self.duration_module(batch_x, lengths)
        pred_phonemes, pred_rules, pred_rule_confidences = _decode_duration_rule_confidences(
            log_probs[0].cpu(),
            rule_logits[0].cpu(),
            input_length,
        )

        judgments = []
        for idx, expected in enumerate(canonical_rules):
            pred = pred_rules[idx] if idx < len(pred_rules) else 0
            confidence = pred_rule_confidences[idx] if idx < len(pred_rule_confidences) else None
            judgments.append({
                "position": idx,
                "rule": id_to_rule.get(expected, "none"),
                "predicted_rule": id_to_rule.get(pred, "none"),
                "is_correct": expected == pred,
                "detail": "" if expected == pred else f"expected {id_to_rule.get(expected, 'none')} but got {id_to_rule.get(pred, 'none')}",
                "confidence": confidence,
            })

        report = aggregate_diagnosis(
            word=word,
            canonical_phonemes=canonical_phonemes,
            predicted_phonemes=pred_phonemes,
            canonical_rules=canonical_rules,
            module_judgments=judgments,
        )
        return {"report": report.to_dict(), "feedback": render_feedback(report)}

    @torch.no_grad()
    def run_modular(
        self,
        *,
        canonical_phonemes: list[int],
        canonical_rules: list[int],
        canonical_chars: list[str] | None = None,
        word: str = "sample",
        duration_x: torch.Tensor | None = None,
        localized_duration_x: torch.Tensor | None = None,
        transition_mfcc: torch.Tensor | None = None,
        transition_ssl: torch.Tensor | None = None,
        localized_transition_x: torch.Tensor | None = None,
        burst_x: torch.Tensor | None = None,
    ) -> dict:
        plan = build_routing_plan(canonical_rules)
        pred_phonemes = canonical_phonemes
        module_judgments: list[dict] = []

        if plan.use_duration and duration_x is not None:
            batch_x = duration_x.unsqueeze(0).to(self.device)
            lengths = torch.tensor([duration_x.size(0)], dtype=torch.long, device=self.device)
            log_probs, rule_logits = self.duration_module(batch_x, lengths)
            pred_phonemes, pred_rules, pred_rule_confidences = _decode_duration_rule_confidences(
                log_probs[0].cpu(),
                rule_logits[0].cpu(),
                duration_x.size(0),
            )
            projected_rules, projected_confidences = _project_duration_rules_to_positions(
                rule_logits[0].cpu(),
                duration_x.size(0),
                len(canonical_rules),
            )
            localized_duration_evidence = None
            if self.localized_duration_module is not None and localized_duration_x is not None:
                localized_x = localized_duration_x.unsqueeze(0).to(self.device)
                localized_lengths = torch.tensor([localized_duration_x.size(0)], dtype=torch.long, device=self.device)
                localized_logits = self.localized_duration_module(localized_x, localized_lengths)[0].cpu()
                localized_duration_evidence = _decode_localized_duration_evidence(
                    localized_logits,
                    localized_duration_x.size(0),
                    self.localized_duration_labels,
                    self.localized_duration_thresholds,
                    self.localized_duration_frame_hop_sec,
                )
            for idx, expected in enumerate(canonical_rules):
                expected_name = id_to_rule.get(expected, "none")
                if expected_name not in {"madd", "ghunnah", "none"}:
                    continue
                if not canonical_phonemes:
                    if expected_name == "none":
                        continue
                    pred = projected_rules[idx] if idx < len(projected_rules) else 0
                    confidence = projected_confidences[idx] if idx < len(projected_confidences) else None
                else:
                    pred = pred_rules[idx] if idx < len(pred_rules) else 0
                    confidence = pred_rule_confidences[idx] if idx < len(pred_rule_confidences) else None
                localized_prob = None
                localized_probabilities = None
                localized_span_count = None
                localized_top_span = None
                localized_threshold = None
                localized_predicted_labels = []
                if localized_duration_evidence is not None and expected_name in {"madd", "ghunnah"}:
                    localized_probabilities = dict(localized_duration_evidence["clip_probabilities"])
                    localized_prob = localized_duration_evidence["clip_probabilities"].get(expected_name)
                    localized_threshold = localized_duration_evidence["thresholds"].get(expected_name)
                    localized_predicted_labels = list(localized_duration_evidence["predicted_labels"])
                    label_spans = localized_duration_evidence["predicted_spans"].get(expected_name, [])
                    localized_span_count = len(label_spans)
                    if label_spans:
                        localized_top_span = max(label_spans, key=lambda span: span.get("max_prob", 0.0))
                sequence_pred_name = id_to_rule.get(pred, "none")
                calibrated_rule_name, calibrated_applied, calibrated_reason, calibrated_probabilities = _maybe_calibrate_duration_prediction(
                    sequence_pred_name=sequence_pred_name,
                    sequence_confidence=confidence,
                    localized_evidence=localized_duration_evidence,
                    canonical_chars=canonical_chars,
                    position=idx,
                    calibrator=self.duration_fusion_calibrator,
                    char_vocab=self.duration_fusion_char_vocab,
                    device=self.device,
                )
                if self.duration_fusion_calibrator is not None and localized_duration_evidence is not None:
                    predicted_rule_name = calibrated_rule_name
                    fusion_applied = calibrated_applied
                    fusion_reason = calibrated_reason
                    decision_source = "learned_duration_fusion"
                else:
                    predicted_rule_name, fusion_applied, fusion_reason = _maybe_override_duration_prediction(
                        sequence_pred_name=sequence_pred_name,
                        localized_evidence=localized_duration_evidence,
                        canonical_chars=canonical_chars,
                        position=idx,
                        override_threshold=self.duration_localizer_override_threshold,
                        override_chars=self.duration_localizer_override_chars,
                    )
                    decision_source = (
                        "sequence_overridden_by_localized_evidence"
                        if fusion_applied
                        else "sequence_with_localized_evidence"
                        if localized_duration_evidence is not None and expected_name in {"madd", "ghunnah"}
                        else "sequence_only"
                    )
                module_judgments.append(
                    {
                        "position": idx,
                        "source_module": "duration",
                        "rule": expected_name,
                        "predicted_rule": predicted_rule_name,
                        "sequence_predicted_rule": sequence_pred_name,
                        "is_correct": expected_name == predicted_rule_name,
                        "detail": "" if expected_name == predicted_rule_name else f"expected {expected_name} but got {predicted_rule_name}",
                        "confidence": confidence,
                        "decision_source": decision_source,
                        "fusion_applied": fusion_applied,
                        "fusion_reason": fusion_reason,
                        "fusion_probabilities": calibrated_probabilities,
                        "localized_clip_probabilities": localized_probabilities,
                        "localized_clip_probability": localized_prob,
                        "localized_threshold": localized_threshold,
                        "localized_predicted_span_count": localized_span_count,
                        "localized_top_span": localized_top_span,
                        "localized_predicted_labels": localized_predicted_labels,
                    }
                )

        if plan.use_transition and self.transition_module is not None and transition_mfcc is not None and transition_ssl is not None:
            mfcc = transition_mfcc.unsqueeze(0).to(self.device)
            ssl = transition_ssl.unsqueeze(0).to(self.device)
            lengths = torch.tensor([transition_mfcc.size(0)], dtype=torch.long, device=self.device)
            logits = self.transition_module(mfcc, ssl, lengths)
            pred_label, pred_confidence = _decode_transition_prediction(logits, self.transition_thresholds)
            pred_name = TRANSITION_RULES[pred_label] if 0 <= pred_label < len(TRANSITION_RULES) else "none"
            localized_evidence = None
            if self.localized_transition_module is not None and localized_transition_x is not None:
                localized_x = localized_transition_x.unsqueeze(0).to(self.device)
                localized_lengths = torch.tensor([localized_transition_x.size(0)], dtype=torch.long, device=self.device)
                localized_logits = self.localized_transition_module(localized_x, localized_lengths)[0].cpu()
                localized_evidence = _decode_localized_transition_evidence(
                    localized_logits,
                    localized_transition_x.size(0),
                    self.localized_transition_labels,
                    self.localized_transition_thresholds,
                    self.localized_transition_frame_hop_sec,
                )
            for idx, expected in enumerate(canonical_rules):
                expected_name = id_to_rule.get(expected, "none")
                if expected_name not in {"ikhfa", "idgham"}:
                    continue
                localized_prob = None
                localized_span_count = None
                localized_top_span = None
                localized_threshold = None
                localized_predicted_labels = []
                if localized_evidence is not None:
                    localized_prob = localized_evidence["clip_probabilities"].get(expected_name)
                    localized_threshold = localized_evidence["thresholds"].get(expected_name)
                    localized_predicted_labels = list(localized_evidence["predicted_labels"])
                    label_spans = localized_evidence["predicted_spans"].get(expected_name, [])
                    localized_span_count = len(label_spans)
                    if label_spans:
                        localized_top_span = max(label_spans, key=lambda span: span.get("max_prob", 0.0))
                module_judgments.append(
                    {
                        "position": idx,
                        "source_module": "transition",
                        "rule": expected_name,
                        "predicted_rule": pred_name,
                        "is_correct": expected_name == pred_name,
                        "detail": "" if expected_name == pred_name else f"expected {expected_name} but got {pred_name}",
                        "confidence": pred_confidence,
                        "decision_source": "whole_verse_with_localized_evidence" if localized_evidence is not None else "whole_verse_only",
                        "localized_clip_probability": localized_prob,
                        "localized_threshold": localized_threshold,
                        "localized_predicted_span_count": localized_span_count,
                        "localized_top_span": localized_top_span,
                        "localized_predicted_labels": localized_predicted_labels,
                    }
                )

        if plan.use_burst and self.burst_module is not None and burst_x is not None:
            logits = self.burst_module(burst_x.unsqueeze(0).to(self.device))
            probs = logits.softmax(dim=-1)[0]
            pred_positive = int(logits.argmax(dim=-1)[0].item()) == 1
            for idx, expected in enumerate(canonical_rules):
                expected_name = id_to_rule.get(expected, "none")
                if expected_name != "qalqalah":
                    continue
                pred_name = "qalqalah" if pred_positive else "none"
                module_judgments.append(
                    {
                        "position": idx,
                        "source_module": "burst",
                        "rule": "qalqalah",
                        "predicted_rule": pred_name,
                        "is_correct": pred_positive,
                        "detail": "" if pred_positive else "expected qalqalah burst but detector did not fire",
                        "confidence": float(probs[1 if pred_positive else 0].item()),
                    }
                )

        report = aggregate_diagnosis(
            word=word,
            canonical_phonemes=canonical_phonemes,
            predicted_phonemes=pred_phonemes,
            canonical_rules=canonical_rules,
            module_judgments=module_judgments,
            canonical_chars=canonical_chars,
        )
        return {
            "routing_plan": {
                "use_duration": plan.use_duration,
                "use_duration_localizer": plan.use_duration and self.localized_duration_module is not None and localized_duration_x is not None,
                "use_transition": plan.use_transition,
                "use_transition_localizer": plan.use_transition and self.localized_transition_module is not None and localized_transition_x is not None,
                "use_burst": plan.use_burst,
            },
            "module_judgments": module_judgments,
            "report": report.to_dict(),
            "feedback": render_feedback(report),
        }
