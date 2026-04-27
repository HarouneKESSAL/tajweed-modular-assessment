from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset

from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import DurationFusionCalibrator
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json
from tajweed_assessment.models.common.decoding import ctc_lexicon_decode, ctc_prefix_beam_search
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch as collate_chunked_content_batch,
    normalize_text_target as normalize_chunked_text_target,
    split_content_indices as split_chunked_content_indices,
)
from content.train_content import (
    ManifestContentDataset,
    collate_content_batch as collate_full_content_batch,
    normalize_text_target as normalize_full_text_target,
    split_content_indices as split_full_content_indices,
)


DEFAULT_DURATION_MANIFEST = "data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl"
DEFAULT_LOCALIZED_DURATION_MANIFEST = "data/alignment/duration_time_projection_strict.jsonl"
DEFAULT_TRANSITION_MANIFEST = "data/manifests/retasy_transition_subset.jsonl"
DEFAULT_BURST_MANIFEST = "data/manifests/retasy_burst_subset.jsonl"
DEFAULT_LOCALIZED_TRANSITION_MANIFEST = "data/alignment/transition_time_projection_strict.jsonl"
DEFAULT_CONTENT_MANIFEST = "data/manifests/retasy_quranjson_train.jsonl"
DEFAULT_CHUNKED_CONTENT_MANIFEST = "data/manifests/retasy_content_chunks.jsonl"


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


def preferred_duration_checkpoint() -> str:
    return "duration_module.pt"


def approved_duration_fusion_checkpoint() -> Path:
    return PROJECT_ROOT / "checkpoints" / "duration_fusion_calibrator_approved.pt"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LocalizedTransitionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)


class LocalizedDurationBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)


def load_duration_module(checkpoint_name: str = "duration_module.pt") -> DurationRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / checkpoint_name)
    model = DurationRuleModule(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_phonemes=model_cfg["num_phonemes"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_transition_module(checkpoint_name: str = "transition_module.pt") -> TransitionRuleModule | None:
    ckpt_path = PROJECT_ROOT / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        return None
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
    ckpt = load_checkpoint(ckpt_path)
    model = TransitionRuleModule(
        mfcc_dim=model_cfg["mfcc_dim"],
        ssl_dim=model_cfg["ssl_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_transition_thresholds() -> dict[str, float] | None:
    path = PROJECT_ROOT / "checkpoints" / "transition_thresholds.json"
    if not path.exists():
        return None
    data = load_json(path)
    raw = data.get("thresholds", data) if isinstance(data, dict) else {}
    return {str(k): float(v) for k, v in raw.items()}


def load_localized_duration_module() -> tuple[nn.Module | None, tuple[str, ...], dict[str, float] | None]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / "localized_duration_model.pt"
    if not ckpt_path.exists():
        return None, tuple(), None
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt.get("config", {})
    label_vocab = tuple(ckpt.get("label_vocab", ["ghunnah", "madd"]))
    model = LocalizedDurationBiLSTM(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    decoder_path = PROJECT_ROOT / "checkpoints" / "localized_duration_decoder.json"
    thresholds = None
    if decoder_path.exists():
        data = load_json(decoder_path)
        raw = data.get("decoder_config", data) if isinstance(data, dict) else {}
        thresholds = {str(label): float(raw.get(label, {}).get("threshold", 0.5)) for label in label_vocab}

    return model, label_vocab, thresholds


def load_duration_fusion_calibrator(checkpoint_name: str = "") -> tuple[nn.Module | None, dict[str, int] | None]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / checkpoint_name if checkpoint_name else approved_duration_fusion_checkpoint()
    if not ckpt_path.exists():
        return None, None
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt.get("config", {})
    char_vocab = {str(k): int(v) for k, v in ckpt.get("char_vocab", {}).items()}
    model = DurationFusionCalibrator(
        num_numeric_features=int(config.get("num_numeric_features", 8)),
        char_vocab_size=max(len(char_vocab), 2),
        char_embedding_dim=int(config.get("char_embedding_dim", 8)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, char_vocab


def load_localized_transition_module() -> tuple[nn.Module | None, tuple[str, ...], dict[str, float] | None]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / "localized_transition_model.pt"
    if not ckpt_path.exists():
        return None, tuple(), None
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt.get("config", {})
    label_vocab = tuple(ckpt.get("label_vocab", ["idgham", "ikhfa"]))
    model = LocalizedTransitionBiLSTM(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    decoder_path = PROJECT_ROOT / "checkpoints" / "localized_transition_decoder.json"
    thresholds = None
    if decoder_path.exists():
        data = load_json(decoder_path)
        raw = data.get("decoder_config", data) if isinstance(data, dict) else {}
        thresholds = {str(label): float(raw.get(label, {}).get("threshold", 0.5)) for label in label_vocab}

    return model, label_vocab, thresholds


def load_burst_module() -> QalqalahCNN | None:
    ckpt_path = PROJECT_ROOT / "checkpoints" / "burst_module.pt"
    if not ckpt_path.exists():
        return None
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_burst.yaml")
    ckpt = load_checkpoint(ckpt_path)
    model = QalqalahCNN(
        input_dim=model_cfg["input_dim"],
        channels=tuple(model_cfg["channels"]),
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_content_module(checkpoint_name: str) -> tuple[ContentVerificationModule | None, dict | None]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        return None, None
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    ckpt = load_checkpoint(ckpt_path)
    ckpt_model_cfg = ckpt.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    num_chars = len(ckpt.get("char_to_id", {})) + 1
    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=num_chars,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_chunked_content_decoder_config(config_path: str = "checkpoints/content_chunked_decoder.json") -> dict[str, float | bool | str | int]:
    path = PROJECT_ROOT / config_path
    if not path.exists():
        return {"blank_penalty": 0.0, "use_cleanup": False, "decoder": "greedy", "beam_width": 5, "lexicon_source": "full"}
    data = load_json(path)
    return {
        "blank_penalty": float(data.get("blank_penalty", 0.0)),
        "use_cleanup": bool(data.get("use_cleanup", False)),
        "decoder": str(data.get("decoder", "greedy")),
        "beam_width": int(data.get("beam_width", 5)),
        "lexicon_source": str(data.get("lexicon_source", "full")),
    }


def collapse_excess_repetitions(text: str, max_run: int = 2) -> str:
    if not text:
        return text
    out = [text[0]]
    run_char = text[0]
    run_len = 1
    for ch in text[1:]:
        if ch == run_char:
            run_len += 1
            if run_len <= max_run:
                out.append(ch)
        else:
            run_char = ch
            run_len = 1
            out.append(ch)
    return "".join(out)


def chunked_content_postprocess(text: str) -> str:
    return collapse_excess_repetitions(text, max_run=2)


def apply_chunked_content_blank_penalty(log_probs: torch.Tensor, blank_penalty: float) -> torch.Tensor:
    if blank_penalty == 0.0:
        return log_probs
    adjusted = log_probs.clone()
    adjusted[..., 0] -= blank_penalty
    return adjusted


def decode_chunked_content_sequences(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    *,
    decoder: str,
    beam_width: int,
    lexicon_targets: list[list[int]] | None = None,
) -> list[list[int]]:
    if decoder == "beam":
        return [
            ctc_prefix_beam_search(seq_log_probs[: int(length)], int(length), beam_width=beam_width, blank_id=0)
            for seq_log_probs, length in zip(log_probs, lengths)
        ]
    if decoder == "lexicon":
        return [
            ctc_lexicon_decode(seq_log_probs, int(length), lexicon_targets or [], blank_id=0)
            for seq_log_probs, length in zip(log_probs, lengths)
        ]
    return greedy_decode_from_log_probs(log_probs, lengths)


def build_localized_transition_index(rows: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("id") or row.get("sample_id") or "")
        if key:
            index[key] = row
    return index


def decode_ids(ids, id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(int(i), "") for i in ids)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    dist = levenshtein(gold, pred)
    return max(0.0, 1.0 - (dist / len(gold)))


def build_canonical_rules(row: dict) -> list[int]:
    labels = row.get("normalized_char_labels", [])
    canonical_rules = []
    for item in labels:
        rules = item.get("rules", [])
        label = normalize_rule_name(rules[0]) if rules else "none"
        canonical_rules.append(rule_to_id.get(label, rule_to_id["none"]))
    return canonical_rules


def build_canonical_phonemes(row: dict) -> list[int]:
    phonemes = row.get("canonical_phonemes") or []
    if not phonemes:
        return []
    from tajweed_assessment.data.labels import phoneme_to_id

    return [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]


def build_canonical_chars(row: dict) -> list[str]:
    labels = row.get("normalized_char_labels", [])
    if labels:
        return [str(item.get("char", "")) for item in labels]
    text = row.get("normalized_text") or row.get("text") or ""
    return list(str(text))


def safe_accuracy(correct: int, total: int) -> float | None:
    if total <= 0:
        return None
    return correct / total


def format_acc(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def evaluate_duration_manifest(pipeline: TajweedInferencePipeline, rows: list[dict], limit: int = 0) -> dict:
    rows = rows[:limit] if limit > 0 else rows
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)
    route_counts = Counter()
    rule_totals = Counter()
    rule_correct = Counter()
    hybrid = {
        "localized_available": 0,
        "localized_same_as_sequence": 0,
        "localized_supports_gold": 0,
        "localized_supports_sequence": 0,
        "localized_disagrees_with_sequence": 0,
        "gold_supported_by_class": Counter(),
        "gold_total_by_class": Counter(),
        "sequence_supported_by_class": Counter(),
        "sequence_total_by_class": Counter(),
    }

    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
        ssl = ssl_extractor.from_mfcc(mfcc)
        result = pipeline.run_modular(
            canonical_phonemes=build_canonical_phonemes(row),
            canonical_rules=build_canonical_rules(row),
            canonical_chars=build_canonical_chars(row),
            word=row.get("normalized_text") or row.get("text") or row.get("sample_id") or "sample",
            duration_x=mfcc,
            localized_duration_x=mfcc,
            transition_mfcc=mfcc,
            transition_ssl=ssl,
            burst_x=mfcc,
        )

        routing = result["routing_plan"]
        if routing["use_duration"]:
            route_counts["duration"] += 1
        if routing["use_transition"]:
            route_counts["transition"] += 1
        if routing["use_burst"]:
            route_counts["burst"] += 1

        for judgment in result.get("module_judgments", []):
            if judgment.get("source_module") != "duration":
                continue
            rule_name = str(judgment.get("rule"))
            rule_totals[rule_name] += 1
            if judgment.get("is_correct"):
                rule_correct[rule_name] += 1

            if judgment.get("decision_source") in {"sequence_with_localized_evidence", "sequence_overridden_by_localized_evidence", "learned_duration_fusion"}:
                hybrid["localized_available"] += 1
                localized_labels = judgment.get("localized_predicted_labels") or []
                localized_labels = [str(label) for label in localized_labels]
                predicted_rule = str(judgment.get("predicted_rule", "none"))
                sequence_rule = str(judgment.get("sequence_predicted_rule", predicted_rule))
                if sequence_rule in localized_labels:
                    hybrid["localized_same_as_sequence"] += 1
                    hybrid["localized_supports_sequence"] += 1
                    hybrid["sequence_supported_by_class"][sequence_rule] += 1
                else:
                    hybrid["localized_disagrees_with_sequence"] += 1
                hybrid["sequence_total_by_class"][sequence_rule] += 1

                if rule_name in localized_labels:
                    hybrid["localized_supports_gold"] += 1
                    hybrid["gold_supported_by_class"][rule_name] += 1
                hybrid["gold_total_by_class"][rule_name] += 1

        if idx % 25 == 0:
            print(f"[duration] Processed {idx}/{len(rows)} samples...")

    total = sum(rule_totals.values())
    correct = sum(rule_correct.values())
    summary = {
        "samples": len(rows),
        "route_counts": dict(route_counts),
        "total_positions": total,
        "correct_positions": correct,
        "accuracy": safe_accuracy(correct, total),
        "rule_summary": {
            rule: {
                "total": rule_totals[rule],
                "correct": rule_correct[rule],
                "accuracy": safe_accuracy(rule_correct[rule], rule_totals[rule]),
            }
            for rule in sorted(rule_totals)
        },
    }
    if hybrid["localized_available"] > 0:
        summary["hybrid_support"] = {
            "localized_available": hybrid["localized_available"],
            "localized_same_as_sequence": hybrid["localized_same_as_sequence"],
            "localized_same_rate": safe_accuracy(hybrid["localized_same_as_sequence"], hybrid["localized_available"]),
            "localized_supports_gold": hybrid["localized_supports_gold"],
            "localized_supports_gold_rate": safe_accuracy(hybrid["localized_supports_gold"], hybrid["localized_available"]),
            "localized_supports_sequence": hybrid["localized_supports_sequence"],
            "localized_supports_sequence_rate": safe_accuracy(hybrid["localized_supports_sequence"], hybrid["localized_available"]),
            "localized_disagrees_with_sequence": hybrid["localized_disagrees_with_sequence"],
            "gold_supported_by_class": {
                label: {
                    "supported": hybrid["gold_supported_by_class"][label],
                    "total": hybrid["gold_total_by_class"][label],
                    "rate": safe_accuracy(hybrid["gold_supported_by_class"][label], hybrid["gold_total_by_class"][label]),
                }
                for label in sorted(hybrid["gold_total_by_class"])
            },
            "sequence_supported_by_class": {
                label: {
                    "supported": hybrid["sequence_supported_by_class"][label],
                    "total": hybrid["sequence_total_by_class"][label],
                    "rate": safe_accuracy(hybrid["sequence_supported_by_class"][label], hybrid["sequence_total_by_class"][label]),
                }
                for label in sorted(hybrid["sequence_total_by_class"])
            },
        }
    return summary


def decode_localized_transition(
    logits: torch.Tensor,
    length: int,
    label_vocab: tuple[str, ...],
    thresholds: dict[str, float] | None,
) -> tuple[str, list[str], dict[str, float]]:
    probs = torch.sigmoid(logits[:length])
    clip_probs = probs.max(dim=0).values
    clip_probabilities = {label: float(clip_probs[idx].item()) for idx, label in enumerate(label_vocab)}
    predicted_labels = [
        label
        for idx, label in enumerate(label_vocab)
        if float(clip_probs[idx].item()) >= float((thresholds or {}).get(label, 0.5))
    ]
    if not predicted_labels:
        return "none", predicted_labels, clip_probabilities
    best_label = max(predicted_labels, key=lambda label: clip_probabilities[label])
    return best_label, predicted_labels, clip_probabilities


def evaluate_transition_manifest(
    model: TransitionRuleModule | None,
    rows: list[dict],
    limit: int = 0,
    *,
    localized_model: nn.Module | None = None,
    localized_label_vocab: tuple[str, ...] = (),
    localized_thresholds: dict[str, float] | None = None,
    localized_index: dict[str, dict] | None = None,
) -> dict:
    if model is None:
        return {"available": False}
    rows = rows[:limit] if limit > 0 else rows
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)
    confusion = torch.zeros(len(TRANSITION_RULES), len(TRANSITION_RULES), dtype=torch.long)
    thresholds = load_transition_thresholds()
    hybrid = {
        "localized_available": 0,
        "localized_same_as_whole_verse": 0,
        "localized_supports_gold": 0,
        "localized_supports_whole_verse": 0,
        "localized_disagrees_with_whole_verse": 0,
        "gold_supported_by_class": Counter(),
        "gold_total_by_class": Counter(),
        "whole_verse_supported_by_class": Counter(),
        "whole_verse_total_by_class": Counter(),
    }

    model = model.to("cpu").eval()
    if localized_model is not None:
        localized_model = localized_model.to("cpu").eval()
    for idx, row in enumerate(rows, start=1):
        mfcc_single = extract_mfcc_features(row["audio_path"])
        mfcc = mfcc_single.unsqueeze(0)
        ssl = ssl_extractor.from_mfcc(mfcc_single).unsqueeze(0)
        lengths = torch.tensor([mfcc.size(1)], dtype=torch.long)
        with torch.no_grad():
            logits = model(mfcc, ssl, lengths)
            probs = logits.softmax(dim=-1)[0]
        if thresholds:
            non_none_probs = probs[1:]
            pred = int(non_none_probs.argmax().item()) + 1
            pred_name = TRANSITION_RULES[pred]
            if float(probs[pred].item()) < float(thresholds.get(pred_name, 0.5)):
                pred = 0
        else:
            pred = int(logits.argmax(dim=-1)[0].item())
        gold_name = normalize_rule_name((row.get("canonical_rules") or ["none"])[0])
        gold_name = gold_name if gold_name in TRANSITION_RULES else "none"
        gold = TRANSITION_RULES.index(gold_name)
        confusion[gold, pred] += 1

        row_key = str(row.get("id") or row.get("sample_id") or "")
        localized_row = localized_index.get(row_key) if localized_index else None
        if localized_model is not None and localized_row is not None and localized_label_vocab:
            hybrid["localized_available"] += 1
            localized_input = mfcc_single.unsqueeze(0)
            localized_lengths = torch.tensor([mfcc_single.size(0)], dtype=torch.long)
            with torch.no_grad():
                localized_logits = localized_model(localized_input, localized_lengths)[0].cpu()
            localized_pred_name, localized_predicted_labels, _ = decode_localized_transition(
                localized_logits,
                mfcc_single.size(0),
                localized_label_vocab,
                localized_thresholds,
            )

            pred_name = TRANSITION_RULES[pred]
            if localized_pred_name == pred_name:
                hybrid["localized_same_as_whole_verse"] += 1
            else:
                hybrid["localized_disagrees_with_whole_verse"] += 1

            if gold_name in localized_predicted_labels:
                hybrid["localized_supports_gold"] += 1
                hybrid["gold_supported_by_class"][gold_name] += 1
            hybrid["gold_total_by_class"][gold_name] += 1

            if pred_name in localized_predicted_labels:
                hybrid["localized_supports_whole_verse"] += 1
                hybrid["whole_verse_supported_by_class"][pred_name] += 1
            hybrid["whole_verse_total_by_class"][pred_name] += 1

        if idx % 25 == 0:
            print(f"[transition] Processed {idx}/{len(rows)} samples...")

    per_class = {}
    for idx, label in enumerate(TRANSITION_RULES):
        support = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        per_class[label] = {
            "total": support,
            "correct": correct,
            "accuracy": safe_accuracy(correct, support),
        }

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    summary = {
        "available": True,
        "samples": len(rows),
        "accuracy": safe_accuracy(correct, total),
        "confusion_matrix": confusion.tolist(),
        "class_summary": per_class,
    }
    if hybrid["localized_available"] > 0:
        summary["hybrid_support"] = {
            "localized_available": hybrid["localized_available"],
            "localized_same_as_whole_verse": hybrid["localized_same_as_whole_verse"],
            "localized_same_rate": safe_accuracy(hybrid["localized_same_as_whole_verse"], hybrid["localized_available"]),
            "localized_supports_gold": hybrid["localized_supports_gold"],
            "localized_supports_gold_rate": safe_accuracy(hybrid["localized_supports_gold"], hybrid["localized_available"]),
            "localized_supports_whole_verse": hybrid["localized_supports_whole_verse"],
            "localized_supports_whole_verse_rate": safe_accuracy(hybrid["localized_supports_whole_verse"], hybrid["localized_available"]),
            "localized_disagrees_with_whole_verse": hybrid["localized_disagrees_with_whole_verse"],
            "gold_supported_by_class": {
                label: {
                    "supported": hybrid["gold_supported_by_class"][label],
                    "total": hybrid["gold_total_by_class"][label],
                    "rate": safe_accuracy(hybrid["gold_supported_by_class"][label], hybrid["gold_total_by_class"][label]),
                }
                for label in sorted(hybrid["gold_total_by_class"])
            },
            "whole_verse_supported_by_class": {
                label: {
                    "supported": hybrid["whole_verse_supported_by_class"][label],
                    "total": hybrid["whole_verse_total_by_class"][label],
                    "rate": safe_accuracy(hybrid["whole_verse_supported_by_class"][label], hybrid["whole_verse_total_by_class"][label]),
                }
                for label in sorted(hybrid["whole_verse_total_by_class"])
            },
        }
    return summary


def evaluate_burst_manifest(model: QalqalahCNN | None, rows: list[dict], limit: int = 0) -> dict:
    if model is None:
        return {"available": False}
    rows = rows[:limit] if limit > 0 else rows
    label_names = ["none", "qalqalah"]
    confusion = torch.zeros(2, 2, dtype=torch.long)

    model = model.to("cpu").eval()
    for idx, row in enumerate(rows, start=1):
        x = extract_mfcc_features(row["audio_path"]).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        pred = int(logits.argmax(dim=-1)[0].item())
        gold = int(row.get("burst_label", 1 if (row.get("canonical_rules") or ["none"])[0] == "qalqalah" else 0))
        confusion[gold, pred] += 1

        if idx % 25 == 0:
            print(f"[burst] Processed {idx}/{len(rows)} samples...")

    per_class = {}
    for idx, label in enumerate(label_names):
        support = int(confusion[idx].sum().item())
        correct = int(confusion[idx, idx].item())
        per_class[label] = {
            "total": support,
            "correct": correct,
            "accuracy": safe_accuracy(correct, support),
        }

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    return {
        "available": True,
        "samples": len(rows),
        "accuracy": safe_accuracy(correct, total),
        "confusion_matrix": confusion.tolist(),
        "class_summary": per_class,
    }


def evaluate_full_content_manifest(
    model: ContentVerificationModule | None,
    checkpoint: dict | None,
    manifest_path: str,
    *,
    split: str = "val",
    limit: int = 0,
    feature_cache_dir: str = "data/interim/content_ssl_cache",
) -> dict:
    if model is None or checkpoint is None:
        return {"available": False, "mode": "full_verse"}

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    from tajweed_assessment.data.speed import SpeedNormalizationConfig

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    dataset = ManifestContentDataset(
        PROJECT_ROOT / manifest_path,
        sample_rate=data_cfg["sample_rate"],
        n_mfcc=data_cfg["n_mfcc"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / feature_cache_dir,
    )

    train_idx, val_idx = split_full_content_indices(dataset.rows, val_fraction=0.2, seed=train_cfg["seed"])
    if split == "train":
        indices = train_idx
    elif split == "val":
        indices = val_idx
    else:
        indices = list(range(len(dataset)))
    if limit > 0:
        indices = indices[:limit]

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=min(train_cfg["batch_size"], max(1, len(indices) or 1)),
        shuffle=False,
        collate_fn=collate_full_content_batch,
    )

    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    model = model.to("cpu").eval()

    total = 0
    exact = 0
    char_acc_sum = 0.0
    edit_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"])
            decoded = greedy_decode_from_log_probs(log_probs.cpu(), batch["input_lengths"])
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_full_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                total += 1
                exact += int(pred == gold)
                char_acc_sum += char_accuracy(gold, pred)
                edit_sum += levenshtein(gold, pred)

    return {
        "available": True,
        "mode": "full_verse",
        "samples": total,
        "split": split,
        "exact_match": safe_accuracy(exact, total),
        "char_accuracy": (char_acc_sum / total) if total else None,
        "edit_distance": (edit_sum / total) if total else None,
    }


def evaluate_chunked_content_manifest(
    model: ContentVerificationModule | None,
    checkpoint: dict | None,
    manifest_path: str,
    *,
    split: str = "val",
    split_mode: str = "reciter",
    limit: int = 0,
    feature_cache_dir: str = "data/interim/content_chunk_ssl_cache",
    decoder_config_path: str = "checkpoints/content_chunked_decoder.json",
) -> dict:
    if model is None or checkpoint is None:
        return {"available": False, "mode": "chunked"}

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    from tajweed_assessment.data.speed import SpeedNormalizationConfig

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    dataset = ChunkedContentDataset(
        PROJECT_ROOT / manifest_path,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / feature_cache_dir,
    )

    full_rows = list(dataset.rows)
    full_char_to_id = dict(dataset.char_to_id)
    train_idx, val_idx = split_chunked_content_indices(
        dataset.rows,
        val_fraction=0.2,
        seed=train_cfg["seed"],
        split_mode=split_mode,
    )
    if split == "train":
        indices = train_idx
    elif split == "val":
        indices = val_idx
    else:
        indices = list(range(len(dataset)))
    if limit > 0:
        indices = indices[:limit]
    eval_rows = [full_rows[i] for i in indices]

    subset_dataset = ChunkedContentDataset(
        PROJECT_ROOT / manifest_path,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / feature_cache_dir,
        indices=indices,
        char_to_id=full_char_to_id,
    )
    subset = Subset(subset_dataset, list(range(len(subset_dataset))))
    loader = DataLoader(
        subset,
        batch_size=min(train_cfg["batch_size"], max(1, len(indices) or 1)),
        shuffle=False,
        collate_fn=collate_chunked_content_batch,
    )

    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}
    model = model.to("cpu").eval()
    decoder_cfg = load_chunked_content_decoder_config(decoder_config_path)
    blank_penalty = float(decoder_cfg.get("blank_penalty", 0.0))
    use_cleanup = bool(decoder_cfg.get("use_cleanup", False))
    decoder = str(decoder_cfg.get("decoder", "greedy"))
    beam_width = int(decoder_cfg.get("beam_width", 5))
    lexicon_source = str(decoder_cfg.get("lexicon_source", "full"))
    if lexicon_source == "train":
        lexicon_rows = [full_rows[i] for i in train_idx]
    elif lexicon_source == "eval":
        lexicon_rows = eval_rows
    else:
        lexicon_rows = full_rows
    lexicon_texts = sorted({normalize_chunked_text_target(row.get("normalized_text", "")) for row in lexicon_rows})
    lexicon_targets = [
        [full_char_to_id[ch] for ch in text]
        for text in lexicon_texts
        if text and all(ch in full_char_to_id for ch in text)
    ]
    lexicon_text_set = set(lexicon_texts)
    eval_texts = [normalize_chunked_text_target(row.get("normalized_text", "")) for row in eval_rows]
    eval_texts_in_lexicon = sum(1 for text in eval_texts if text in lexicon_text_set)

    total = 0
    exact = 0
    char_acc_sum = 0.0
    edit_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["x"], batch["input_lengths"])
            decoded = decode_chunked_content_sequences(
                apply_chunked_content_blank_penalty(log_probs.cpu(), blank_penalty),
                batch["input_lengths"],
                decoder=decoder,
                beam_width=beam_width,
                lexicon_targets=lexicon_targets,
            )
            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_chunked_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)
                if use_cleanup:
                    pred = chunked_content_postprocess(pred)
                total += 1
                exact += int(pred == gold)
                char_acc_sum += char_accuracy(gold, pred)
                edit_sum += levenshtein(gold, pred)

    return {
        "available": True,
        "mode": "chunked",
        "samples": total,
        "split": split,
        "split_mode": split_mode,
        "decoder": {
            "blank_penalty": blank_penalty,
            "use_cleanup": use_cleanup,
            "decoder": decoder,
            "beam_width": beam_width,
            "decoder_config": str(PROJECT_ROOT / decoder_config_path),
            "lexicon_source": lexicon_source,
            "lexicon_size": len(lexicon_targets),
            "eval_unique_text_count": len(set(eval_texts)),
            "eval_texts_in_lexicon": int(eval_texts_in_lexicon),
            "eval_text_coverage": eval_texts_in_lexicon / max(total, 1),
        },
        "exact_match": safe_accuracy(exact, total),
        "char_accuracy": (char_acc_sum / total) if total else None,
        "edit_distance": (edit_sum / total) if total else None,
    }


def print_transition_summary(summary: dict) -> None:
    if not summary.get("available", True):
        print("Transition summary:")
        print("- checkpoint not available")
        return
    print("Transition summary:")
    print(f"- samples={summary['samples']} acc={format_acc(summary['accuracy'])}")
    for label, stats in summary["class_summary"].items():
        print(
            f"- {label}: correct={stats['correct']} total={stats['total']} acc={format_acc(stats['accuracy'])}"
        )
    hybrid = summary.get("hybrid_support")
    if hybrid:
        print(f"- localized_available={hybrid['localized_available']}")
        print(f"- localized_same_as_whole_verse={hybrid['localized_same_as_whole_verse']} rate={format_acc(hybrid['localized_same_rate'])}")
        print(f"- localized_supports_gold={hybrid['localized_supports_gold']} rate={format_acc(hybrid['localized_supports_gold_rate'])}")
        print(f"- localized_supports_whole_verse={hybrid['localized_supports_whole_verse']} rate={format_acc(hybrid['localized_supports_whole_verse_rate'])}")
        print(f"- localized_disagrees_with_whole_verse={hybrid['localized_disagrees_with_whole_verse']}")
        for label, stats in hybrid["gold_supported_by_class"].items():
            print(f"- gold_support {label}: supported={stats['supported']} total={stats['total']} rate={format_acc(stats['rate'])}")
        for label, stats in hybrid["whole_verse_supported_by_class"].items():
            print(f"- whole_verse_support {label}: supported={stats['supported']} total={stats['total']} rate={format_acc(stats['rate'])}")


def print_burst_summary(summary: dict) -> None:
    if not summary.get("available", True):
        print("Burst summary:")
        print("- checkpoint not available")
        return
    print("Burst summary:")
    print(f"- samples={summary['samples']} acc={format_acc(summary['accuracy'])}")
    for label, stats in summary["class_summary"].items():
        print(
            f"- {label}: correct={stats['correct']} total={stats['total']} acc={format_acc(stats['accuracy'])}"
        )


def print_content_summary(summary: dict) -> None:
    if not summary.get("available", True):
        print(f"Content summary ({summary.get('mode', 'unknown')}):")
        print("- checkpoint not available")
        return
    print(f"Content summary ({summary.get('mode', 'unknown')}):")
    split_mode = summary.get("split_mode")
    split_text = f"{summary['split']}/{split_mode}" if split_mode else str(summary["split"])
    edit_text = f"{summary['edit_distance']:.3f}" if summary["edit_distance"] is not None else "n/a"
    print(
        f"- samples={summary['samples']} split={split_text} "
        f"exact_match={format_acc(summary['exact_match'])} "
        f"char_accuracy={format_acc(summary['char_accuracy'])} "
        f"edit_distance={edit_text}"
    )
    decoder = summary.get("decoder")
    if isinstance(decoder, dict):
        print(
            f"- decoder={decoder.get('decoder')} blank_penalty={decoder.get('blank_penalty')} "
            f"lexicon_source={decoder.get('lexicon_source')} coverage={format_acc(decoder.get('eval_text_coverage'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-manifest", default=DEFAULT_DURATION_MANIFEST)
    parser.add_argument("--localized-duration-manifest", default=DEFAULT_LOCALIZED_DURATION_MANIFEST)
    parser.add_argument("--transition-manifest", default=DEFAULT_TRANSITION_MANIFEST)
    parser.add_argument("--localized-transition-manifest", default=DEFAULT_LOCALIZED_TRANSITION_MANIFEST)
    parser.add_argument("--burst-manifest", default=DEFAULT_BURST_MANIFEST)
    parser.add_argument("--content-manifest", default=DEFAULT_CONTENT_MANIFEST)
    parser.add_argument("--chunked-content-manifest", default=DEFAULT_CHUNKED_CONTENT_MANIFEST)
    parser.add_argument("--duration-checkpoint", default=preferred_duration_checkpoint())
    parser.add_argument(
        "--duration-fusion-checkpoint",
        default="",
        help="Optional duration fusion calibrator checkpoint in checkpoints/. Leave empty to use the conservative duration baseline.",
    )
    parser.add_argument("--transition-checkpoint", default=preferred_transition_checkpoint())
    parser.add_argument("--chunked-content-checkpoint", default="content_chunked_module.pt")
    parser.add_argument("--duration-limit", type=int, default=0)
    parser.add_argument("--transition-limit", type=int, default=0)
    parser.add_argument("--burst-limit", type=int, default=0)
    parser.add_argument("--content-limit", type=int, default=0)
    parser.add_argument("--content-split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--content-split-mode", choices=["reciter", "text"], default="reciter")
    parser.add_argument("--content-decoder-config", default="checkpoints/content_chunked_decoder.json")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    duration_rows = load_jsonl(PROJECT_ROOT / args.duration_manifest)
    transition_rows = load_jsonl(PROJECT_ROOT / args.transition_manifest)
    burst_rows = load_jsonl(PROJECT_ROOT / args.burst_manifest)
    _ = load_jsonl(PROJECT_ROOT / args.localized_duration_manifest) if (PROJECT_ROOT / args.localized_duration_manifest).exists() else []
    localized_transition_rows = load_jsonl(PROJECT_ROOT / args.localized_transition_manifest) if (PROJECT_ROOT / args.localized_transition_manifest).exists() else []

    duration_model = load_duration_module(args.duration_checkpoint)
    localized_duration_model, localized_duration_labels, localized_duration_thresholds = load_localized_duration_module()
    duration_fusion_calibrator, duration_fusion_char_vocab = load_duration_fusion_calibrator(args.duration_fusion_checkpoint)
    transition_model = load_transition_module(args.transition_checkpoint)
    localized_transition_model, localized_transition_labels, localized_transition_thresholds = load_localized_transition_module()
    burst_model = load_burst_module()
    chunked_content_model, chunked_content_checkpoint = load_content_module(args.chunked_content_checkpoint)
    full_content_model, full_content_checkpoint = load_content_module("content_module.pt")
    pipeline = TajweedInferencePipeline(
        duration_module=duration_model,
        transition_module=transition_model,
        duration_fusion_calibrator=duration_fusion_calibrator,
        duration_fusion_char_vocab=duration_fusion_char_vocab,
        localized_duration_module=localized_duration_model,
        localized_transition_module=localized_transition_model,
        burst_module=burst_model,
        transition_thresholds=load_transition_thresholds(),
        localized_duration_thresholds=localized_duration_thresholds,
        localized_transition_thresholds=localized_transition_thresholds,
        localized_duration_labels=localized_duration_labels or ("ghunnah", "madd"),
        localized_transition_labels=localized_transition_labels or ("idgham", "ikhfa"),
        device="cpu",
    )

    duration_summary = evaluate_duration_manifest(pipeline, duration_rows, args.duration_limit)
    transition_summary = evaluate_transition_manifest(
        transition_model,
        transition_rows,
        args.transition_limit,
        localized_model=localized_transition_model,
        localized_label_vocab=localized_transition_labels,
        localized_thresholds=localized_transition_thresholds,
        localized_index=build_localized_transition_index(localized_transition_rows),
    )
    burst_summary = evaluate_burst_manifest(burst_model, burst_rows, args.burst_limit)
    content_summary = evaluate_chunked_content_manifest(
        chunked_content_model,
        chunked_content_checkpoint,
        args.chunked_content_manifest,
        split=args.content_split,
        split_mode=args.content_split_mode,
        limit=args.content_limit,
        decoder_config_path=args.content_decoder_config,
    )
    content_reference_summary = evaluate_full_content_manifest(
        full_content_model,
        full_content_checkpoint,
        args.content_manifest,
        split=args.content_split,
        limit=args.content_limit,
    )

    summary = {
        "duration": duration_summary,
        "transition": transition_summary,
        "burst": burst_summary,
        "content": content_summary,
        "content_reference_full_verse": content_reference_summary,
        "duration_checkpoint": str(PROJECT_ROOT / "checkpoints" / args.duration_checkpoint),
        "transition_checkpoint": str(PROJECT_ROOT / "checkpoints" / args.transition_checkpoint),
    }

    print("")
    print("Duration summary:")
    print(
        f"- samples={duration_summary['samples']} positions={duration_summary['total_positions']} "
        f"acc={format_acc(duration_summary['accuracy'])}"
    )
    print(f"- route_counts={duration_summary['route_counts']}")
    for rule, stats in duration_summary["rule_summary"].items():
        print(f"- {rule}: correct={stats['correct']} total={stats['total']} acc={format_acc(stats['accuracy'])}")
    duration_hybrid = duration_summary.get("hybrid_support")
    if duration_hybrid:
        print(f"- localized_available={duration_hybrid['localized_available']}")
        print(f"- localized_same_as_sequence={duration_hybrid['localized_same_as_sequence']} rate={format_acc(duration_hybrid['localized_same_rate'])}")
        print(f"- localized_supports_gold={duration_hybrid['localized_supports_gold']} rate={format_acc(duration_hybrid['localized_supports_gold_rate'])}")
        print(f"- localized_supports_sequence={duration_hybrid['localized_supports_sequence']} rate={format_acc(duration_hybrid['localized_supports_sequence_rate'])}")
        print(f"- localized_disagrees_with_sequence={duration_hybrid['localized_disagrees_with_sequence']}")
        for label, stats in duration_hybrid["gold_supported_by_class"].items():
            print(f"- gold_support {label}: supported={stats['supported']} total={stats['total']} rate={format_acc(stats['rate'])}")
        for label, stats in duration_hybrid["sequence_supported_by_class"].items():
            print(f"- sequence_support {label}: supported={stats['supported']} total={stats['total']} rate={format_acc(stats['rate'])}")

    print("")
    print_transition_summary(transition_summary)
    print("")
    print_burst_summary(burst_summary)
    print("")
    print_content_summary(content_summary)
    print("")
    print("Content reference:")
    print_content_summary(content_reference_summary)

    if args.output_json:
        save_json(summary, PROJECT_ROOT / args.output_json)
        print("")
        print(f"Saved suite JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

