from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from tajweed_assessment.scoring.weighted_score import load_error_weights
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tajweed_assessment.data.labels import normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.common.decoding import decode_with_majority_rules
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import DurationFusionCalibrator
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


def preferred_duration_checkpoint() -> str:
    return "duration_module.pt"


def approved_duration_fusion_checkpoint() -> Path:
    return PROJECT_ROOT / "checkpoints" / "duration_fusion_calibrator_approved.pt"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_text(value: str | None) -> str:
    if value is None:
        return ""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower() in {"cp1252", "charmap"}:
        return str(value).encode("ascii", "backslashreplace").decode("ascii")
    return str(value)


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def load_duration_module() -> DurationRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / preferred_duration_checkpoint())
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


def load_transition_module() -> TransitionRuleModule | None:
    ckpt_path = PROJECT_ROOT / "checkpoints" / preferred_transition_checkpoint()
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

    decoder_path = PROJECT_ROOT / "checkpoints" / "localized_duration_decoder.json"
    thresholds = None
    if decoder_path.exists():
        data = load_json(decoder_path)
        raw = data.get("decoder_config", data) if isinstance(data, dict) else {}
        thresholds = {str(label): float(raw.get(label, {}).get("threshold", 0.5)) for label in label_vocab}

    return model, label_vocab, thresholds


def load_duration_fusion_calibrator() -> tuple[nn.Module | None, dict[str, int] | None]:
    ckpt_path = approved_duration_fusion_checkpoint()
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


def build_canonical_rules(row: dict) -> list[int]:
    labels = row.get("normalized_char_labels", [])
    if labels:
        canonical_rules = []
        for item in labels:
            rules = item.get("rules", [])
            label = normalize_rule_name(rules[0]) if rules else "none"
            canonical_rules.append(rule_to_id.get(label, rule_to_id["none"]))
        return canonical_rules

    fallback_rules = row.get("canonical_rules") or row.get("transition_rules") or []
    if fallback_rules:
        return [rule_to_id.get(normalize_rule_name(rule), rule_to_id["none"]) for rule in fallback_rules]
    return []


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--show-matches", action="store_true")
    parser.add_argument(
        "--error-weights",
        default=None,
        help="Optional path to weighted Tajweed error scoring YAML.",
    )
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(f"sample-index out of range: 0 <= idx < {len(rows)}")

    row = rows[args.sample_index]
    audio_path = row["audio_path"]
    error_weight_config = None
    if args.error_weights:
        weights_path = Path(args.error_weights)
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        error_weight_config = load_error_weights(weights_path)
        
    duration_model = load_duration_module()
    localized_duration_model, localized_duration_labels, localized_duration_thresholds = load_localized_duration_module()
    duration_fusion_calibrator, duration_fusion_char_vocab = load_duration_fusion_calibrator()
    transition_model = load_transition_module()
    localized_transition_model, localized_transition_labels, localized_transition_thresholds = load_localized_transition_module()
    burst_model = load_burst_module()
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
        error_weight_config=error_weight_config,
    )

    mfcc = extract_mfcc_features(audio_path)
    ssl = DummySSLFeatureExtractor(output_dim=64).from_mfcc(mfcc)

    canonical_rules = build_canonical_rules(row)
    canonical_phonemes = build_canonical_phonemes(row)
    canonical_chars = build_canonical_chars(row)

    # Until the real content module is fused into the report path, keep the
    # predicted content track only for display/debugging. If the manifest does
    # not carry canonical phonemes, run in rule-only mode and skip content
    # alignment entirely.
    with torch.no_grad():
        log_probs, _ = duration_model(mfcc.unsqueeze(0), torch.tensor([mfcc.size(0)], dtype=torch.long))
        decoded_phonemes, _ = decode_with_majority_rules(
            log_probs[0].cpu(),
            torch.zeros(log_probs.size(1), len(rule_to_id), dtype=torch.float32),
            mfcc.size(0),
        )

    result = pipeline.run_modular(
        canonical_phonemes=canonical_phonemes,
        canonical_rules=canonical_rules,
        canonical_chars=canonical_chars,
        word=row.get("normalized_text") or row.get("text") or row.get("original_text") or row.get("id") or row.get("sample_id") or "sample",
        duration_x=mfcc,
        localized_duration_x=mfcc,
        transition_mfcc=mfcc,
        transition_ssl=ssl,
        localized_transition_x=mfcc,
        burst_x=mfcc,
    )

    print(f"Sample ID    : {safe_text(row.get('id') or row.get('sample_id'))}")
    print(f"Surah        : {safe_text(row.get('surah_name'))}")
    print(f"Verse key    : {safe_text(row.get('quranjson_verse_key'))}")
    print(f"Text         : {safe_text(row.get('normalized_text') or row.get('text'))}")
    print("Routing plan :")
    print_json(result["routing_plan"])
    print("")
    print("Diagnosis report:")
    print_json(result["report"])
    print("")
    print("Feedback:")
    for line in result["feedback"]:
        print(safe_text(f"- {line}"))
    if result.get("weighted_score") is not None:
        print("")
        print("Weighted score:")
        print_json(result["weighted_score"])
    if args.show_matches:
        matched = [j for j in result.get("module_judgments", []) if j.get("is_correct")]
        if matched:
            print("")
            print("Matched findings:")
            for finding in matched:
                pos = int(finding["position"])
                char = canonical_chars[pos] if pos < len(canonical_chars) else None
                char_text = f' ("{safe_text(char)}")' if char else ""
                confidence = finding.get("confidence")
                confidence_text = f" confidence={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
                localized_prob = finding.get("localized_clip_probability")
                localized_span_count = finding.get("localized_predicted_span_count")
                localized_text = ""
                if finding.get("source_module") in {"duration", "transition"}:
                    parts = []
                    if isinstance(localized_prob, (int, float)):
                        parts.append(f"localized_prob={localized_prob:.2f}")
                    if isinstance(localized_span_count, int):
                        parts.append(f"localized_spans={localized_span_count}")
                    if parts:
                        localized_text = " " + " ".join(parts)
                line = f"- position {pos}{char_text}: expected {finding['rule']}, predicted {finding['predicted_rule']}{confidence_text}{localized_text}"
                print(safe_text(line))
        else:
            print("")
            print("Matched findings:")
            print("- None")


if __name__ == "__main__":
    main()

