from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tajweed_assessment.data.labels import id_to_rule, normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import DurationFusionCalibrator
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, save_json


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


def preferred_duration_checkpoint() -> str:
    return "duration_module.pt"


def approved_duration_fusion_checkpoint() -> Path:
    return PROJECT_ROOT / "checkpoints" / "duration_fusion_calibrator_approved.pt"


def safe_text(value: str | None) -> str:
    if value is None:
        return ""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower() in {"cp1252", "charmap"}:
        return str(value).encode("ascii", "backslashreplace").decode("ascii")
    return str(value)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    model.eval()

    decoder_path = PROJECT_ROOT / "checkpoints" / "localized_duration_decoder.json"
    thresholds = None
    if decoder_path.exists():
        with decoder_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument(
        "--duration-fusion-checkpoint",
        default="",
        help="Optional duration fusion calibrator checkpoint in checkpoints/. Leave empty to use the conservative duration baseline.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of rows to evaluate. 0 means all.")
    parser.add_argument("--show-examples", type=int, default=10, help="How many error examples to print.")
    parser.add_argument("--output-json", default="", help="Optional path to save the summary JSON.")
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    duration_model = load_duration_module()
    duration_fusion_calibrator, duration_fusion_char_vocab = load_duration_fusion_calibrator(args.duration_fusion_checkpoint)
    transition_model = load_transition_module()
    burst_model = load_burst_module()
    localized_duration_model, localized_duration_labels, localized_duration_thresholds = load_localized_duration_module()
    pipeline = TajweedInferencePipeline(
        duration_module=duration_model,
        duration_fusion_calibrator=duration_fusion_calibrator,
        duration_fusion_char_vocab=duration_fusion_char_vocab,
        localized_duration_module=localized_duration_model,
        localized_duration_thresholds=localized_duration_thresholds,
        localized_duration_labels=localized_duration_labels or ("ghunnah", "madd"),
        transition_module=transition_model,
        burst_module=burst_model,
        device="cpu",
    )
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)

    route_counts = Counter()
    module_totals = Counter()
    module_correct = Counter()
    rule_totals = Counter()
    rule_correct = Counter()
    samples_with_errors = 0
    clean_samples = 0
    examples: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
        ssl = ssl_extractor.from_mfcc(mfcc)
        canonical_rules = build_canonical_rules(row)
        canonical_phonemes = build_canonical_phonemes(row)
        canonical_chars = build_canonical_chars(row)

        result = pipeline.run_modular(
            canonical_phonemes=canonical_phonemes,
            canonical_rules=canonical_rules,
            canonical_chars=canonical_chars,
            word=row.get("normalized_text") or row.get("original_text") or row.get("id") or "sample",
            duration_x=mfcc,
            localized_duration_x=mfcc,
            transition_mfcc=mfcc,
            transition_ssl=ssl,
            burst_x=mfcc,
        )

        routing_plan = result["routing_plan"]
        if routing_plan["use_duration"]:
            route_counts["duration"] += 1
        if routing_plan["use_transition"]:
            route_counts["transition"] += 1
        if routing_plan["use_burst"]:
            route_counts["burst"] += 1

        judgments = result.get("module_judgments", [])
        row_errors = [j for j in judgments if not j.get("is_correct")]
        if row_errors:
            samples_with_errors += 1
        else:
            clean_samples += 1

        for judgment in judgments:
            source = judgment.get("source_module", "unknown")
            rule = judgment.get("rule", "unknown")
            module_totals[source] += 1
            rule_totals[rule] += 1
            if judgment.get("is_correct"):
                module_correct[source] += 1
                rule_correct[rule] += 1

        for judgment in row_errors:
            pos = int(judgment["position"])
            char = canonical_chars[pos] if pos < len(canonical_chars) else None
            examples.append(
                {
                    "sample_id": row.get("id"),
                    "surah_name": row.get("surah_name"),
                    "verse_key": row.get("quranjson_verse_key"),
                    "text": row.get("normalized_text"),
                    "source_module": judgment.get("source_module"),
                    "position": pos,
                    "char": char,
                    "expected_rule": judgment.get("rule"),
                    "predicted_rule": judgment.get("predicted_rule"),
                    "confidence": judgment.get("confidence"),
                    "detail": judgment.get("detail"),
                }
            )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    module_summary = {}
    for module_name in sorted(module_totals):
        module_summary[module_name] = {
            "total": module_totals[module_name],
            "correct": module_correct[module_name],
            "accuracy": safe_accuracy(module_correct[module_name], module_totals[module_name]),
        }

    rule_summary = {}
    for rule_name in sorted(rule_totals):
        rule_summary[rule_name] = {
            "total": rule_totals[rule_name],
            "correct": rule_correct[rule_name],
            "accuracy": safe_accuracy(rule_correct[rule_name], rule_totals[rule_name]),
        }

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "samples_evaluated": len(rows),
        "clean_samples": clean_samples,
        "samples_with_errors": samples_with_errors,
        "route_counts": dict(route_counts),
        "module_summary": module_summary,
        "rule_summary": rule_summary,
        "error_examples": examples[: args.show_examples],
    }

    print(f"Samples evaluated : {summary['samples_evaluated']}")
    print(f"Clean samples     : {clean_samples}")
    print(f"Samples w/ errors : {samples_with_errors}")
    print(f"Route counts      : {dict(route_counts)}")
    print("")
    print("Module summary:")
    for module_name, stats in module_summary.items():
        acc = stats["accuracy"]
        acc_text = f"{acc:.3f}" if acc is not None else "n/a"
        print(f"- {module_name}: correct={stats['correct']} total={stats['total']} acc={acc_text}")
    print("")
    print("Rule summary:")
    for rule_name, stats in rule_summary.items():
        acc = stats["accuracy"]
        acc_text = f"{acc:.3f}" if acc is not None else "n/a"
        print(f"- {rule_name}: correct={stats['correct']} total={stats['total']} acc={acc_text}")

    if args.show_examples > 0:
        print("")
        print(f"Error examples (top {min(args.show_examples, len(examples))}):")
        if not examples:
            print("- None")
        else:
            for example in examples[: args.show_examples]:
                char_text = f' ("{safe_text(example["char"])}")' if example["char"] else ""
                confidence = example["confidence"]
                confidence_text = f" confidence={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
                print(
                    f"- {safe_text(example['sample_id'])} {example['source_module']} position {example['position']}{char_text}: "
                    f"expected {example['expected_rule']}, predicted {example['predicted_rule']}.{confidence_text}"
                )
                if example["detail"]:
                    print(f"  detail: {safe_text(example['detail'])}")

    if args.output_json:
        save_json(summary, PROJECT_ROOT / args.output_json)
        print("")
        print(f"Saved summary JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

