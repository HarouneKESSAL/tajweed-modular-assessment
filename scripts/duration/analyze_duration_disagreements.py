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

from tajweed_assessment.data.labels import normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import DurationFusionCalibrator
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json


FOCUS_RULES = ("ghunnah", "madd")


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


def approved_duration_fusion_checkpoint() -> Path:
    return PROJECT_ROOT / "checkpoints" / "duration_fusion_calibrator_approved.pt"


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


def decode_localized_duration_clip(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--checkpoint-name", default="duration_module.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show-examples", type=int, default=15)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    localized_duration_model, localized_duration_labels, localized_duration_thresholds = load_localized_duration_module()
    duration_fusion_calibrator, duration_fusion_char_vocab = load_duration_fusion_calibrator()
    if localized_duration_model is None:
        raise FileNotFoundError("localized duration checkpoint not found at checkpoints/localized_duration_model.pt")

    pipeline = TajweedInferencePipeline(
        duration_module=load_duration_module(args.checkpoint_name),
        duration_fusion_calibrator=duration_fusion_calibrator,
        duration_fusion_char_vocab=duration_fusion_char_vocab,
        localized_duration_module=localized_duration_model,
        localized_duration_thresholds=localized_duration_thresholds,
        localized_duration_labels=localized_duration_labels or ("ghunnah", "madd"),
        device="cpu",
    )

    total_focus = 0
    disagreements = 0
    localized_supports_gold = 0
    localized_supports_sequence = 0
    sequence_correct_localized_wrong = 0
    sequence_wrong_localized_correct = 0
    pair_counts = Counter()
    gold_pair_counts = Counter()
    examples: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
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
        )

        with torch.no_grad():
            localized_logits = localized_duration_model(
                mfcc.unsqueeze(0),
                torch.tensor([mfcc.size(0)], dtype=torch.long),
            )[0].cpu()
        localized_top_label, localized_predicted_labels, localized_clip_probabilities = decode_localized_duration_clip(
            localized_logits,
            mfcc.size(0),
            localized_duration_labels or ("ghunnah", "madd"),
            localized_duration_thresholds,
        )

        for judgment in result.get("module_judgments", []):
            if judgment.get("source_module") != "duration":
                continue
            expected = str(judgment.get("rule"))
            sequence_pred = str(judgment.get("sequence_predicted_rule") or judgment.get("predicted_rule"))
            if expected not in FOCUS_RULES:
                continue

            total_focus += 1
            if expected in localized_predicted_labels:
                localized_supports_gold += 1
            if sequence_pred in localized_predicted_labels:
                localized_supports_sequence += 1

            if sequence_pred != localized_top_label:
                disagreements += 1
                pair_counts[f"{sequence_pred}->{localized_top_label}"] += 1
                gold_pair_counts[f"{expected}|seq={sequence_pred}|loc={localized_top_label}"] += 1
                if judgment.get("is_correct") and localized_top_label != expected:
                    sequence_correct_localized_wrong += 1
                if (not judgment.get("is_correct")) and localized_top_label == expected:
                    sequence_wrong_localized_correct += 1

                pos = int(judgment["position"])
                char = canonical_chars[pos] if pos < len(canonical_chars) else None
                examples.append(
                    {
                        "sample_id": row.get("id"),
                        "surah_name": row.get("surah_name"),
                        "verse_key": row.get("quranjson_verse_key"),
                        "text": row.get("normalized_text"),
                        "position": pos,
                        "char": char,
                        "expected_rule": expected,
                        "sequence_predicted_rule": sequence_pred,
                        "localized_top_rule": localized_top_label,
                        "localized_predicted_labels": list(localized_predicted_labels),
                        "sequence_confidence": judgment.get("confidence"),
                        "localized_clip_probabilities": localized_clip_probabilities,
                        "detail": judgment.get("detail"),
                    }
                )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    examples.sort(
        key=lambda item: (
            item["sequence_confidence"] is not None,
            float(item["sequence_confidence"] or 0.0),
            max(item["localized_clip_probabilities"].values()) if item["localized_clip_probabilities"] else 0.0,
        ),
        reverse=True,
    )

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "checkpoint": str(PROJECT_ROOT / "checkpoints" / args.checkpoint_name),
        "samples_analyzed": len(rows),
        "focus_rules": list(FOCUS_RULES),
        "total_focus_positions": total_focus,
        "disagreements": disagreements,
        "disagreement_rate": (disagreements / total_focus) if total_focus else None,
        "localized_supports_gold": localized_supports_gold,
        "localized_supports_gold_rate": (localized_supports_gold / total_focus) if total_focus else None,
        "localized_supports_sequence": localized_supports_sequence,
        "localized_supports_sequence_rate": (localized_supports_sequence / total_focus) if total_focus else None,
        "sequence_correct_localized_wrong": sequence_correct_localized_wrong,
        "sequence_wrong_localized_correct": sequence_wrong_localized_correct,
        "disagreement_pairs": dict(pair_counts),
        "gold_conditioned_disagreements": dict(gold_pair_counts),
        "top_disagreements": examples[: args.show_examples],
    }

    print(f"Samples analyzed                : {summary['samples_analyzed']}")
    print(f"Focus positions                 : {total_focus}")
    print(f"Disagreements                   : {disagreements}")
    if summary["disagreement_rate"] is not None:
        print(f"Disagreement rate               : {summary['disagreement_rate']:.3f}")
    print(f"Localized supports gold         : {localized_supports_gold}")
    if summary["localized_supports_gold_rate"] is not None:
        print(f"Localized supports gold rate    : {summary['localized_supports_gold_rate']:.3f}")
    print(f"Localized supports sequence     : {localized_supports_sequence}")
    if summary["localized_supports_sequence_rate"] is not None:
        print(f"Localized supports sequence rate: {summary['localized_supports_sequence_rate']:.3f}")
    print(f"Sequence correct / localizer wrong : {sequence_correct_localized_wrong}")
    print(f"Sequence wrong / localizer correct : {sequence_wrong_localized_correct}")

    print("")
    print("Disagreement pairs (sequence -> localizer):")
    if not pair_counts:
        print("- None")
    else:
        for pair, count in pair_counts.most_common():
            print(f"- {pair}: {count}")

    print("")
    print("Gold-conditioned disagreements:")
    if not gold_pair_counts:
        print("- None")
    else:
        for pair, count in gold_pair_counts.most_common():
            print(f"- {pair}: {count}")

    print("")
    print(f"Top disagreements (top {min(args.show_examples, len(examples))}):")
    if not examples:
        print("- None")
    else:
        for example in examples[: args.show_examples]:
            char_text = f' ("{safe_text(example["char"])}")' if example["char"] else ""
            sequence_confidence = example["sequence_confidence"]
            seq_conf_text = f" sequence_conf={sequence_confidence:.2f}" if isinstance(sequence_confidence, (int, float)) else ""
            local_probs = example["localized_clip_probabilities"]
            local_prob_text = " ".join(
                f"{label}={prob:.2f}" for label, prob in sorted(local_probs.items())
            )
            print(
                f"- {safe_text(example['sample_id'])} position {example['position']}{char_text}: "
                f"gold={example['expected_rule']} sequence={example['sequence_predicted_rule']} "
                f"localizer={example['localized_top_rule']}.{seq_conf_text} {local_prob_text}"
            )
            if example["detail"]:
                print(f"  detail: {example['detail']}")
            if example["text"]:
                print(f"  text: {safe_text(example['text'])}")

    if args.output_json:
        save_json(summary, PROJECT_ROOT / args.output_json)
        print("")
        print(f"Saved disagreement JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

