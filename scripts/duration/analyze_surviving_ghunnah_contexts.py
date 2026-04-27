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

from tajweed_assessment.data.labels import normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import DurationFusionCalibrator
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json, save_json


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


def char_window(chars: list[str], position: int, radius: int = 2) -> str:
    left = max(0, position - radius)
    right = min(len(chars), position + radius + 1)
    window = chars[left:right]
    center = position - left
    parts: list[str] = []
    for idx, ch in enumerate(window):
        if idx == center:
            parts.append(f"[{ch}]")
        else:
            parts.append(ch)
    return "".join(parts)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--checkpoint-name", default="duration_module.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show-top", type=int, default=15)
    parser.add_argument("--output-json", default="data/analysis/surviving_ghunnah_contexts.json")
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

    verse_counter = Counter()
    verse_position_counter = Counter()
    window_counter = Counter()
    char_counter = Counter()
    surah_verse_counter = Counter()
    examples: list[dict] = []
    confidence_by_verse: dict[str, list[float]] = defaultdict(list)
    localizer_prob_by_verse: dict[str, list[float]] = defaultdict(list)
    total_surviving = 0

    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
        canonical_rules = build_canonical_rules(row)
        canonical_phonemes = build_canonical_phonemes(row)
        canonical_chars = build_canonical_chars(row)
        text = str(row.get("normalized_text") or row.get("text") or "")
        surah_name = str(row.get("surah_name") or "")
        verse_key = str(row.get("quranjson_verse_key") or "")

        result = pipeline.run_modular(
            canonical_phonemes=canonical_phonemes,
            canonical_rules=canonical_rules,
            canonical_chars=canonical_chars,
            word=text or row.get("id") or "sample",
            duration_x=mfcc,
            localized_duration_x=mfcc,
        )

        for judgment in result.get("module_judgments", []):
            if judgment.get("source_module") != "duration":
                continue
            if judgment.get("rule") != "ghunnah":
                continue
            if judgment.get("predicted_rule") != "madd":
                continue

            pos = int(judgment["position"])
            char = canonical_chars[pos] if pos < len(canonical_chars) else None
            window = char_window(canonical_chars, pos, radius=2)
            verse_position_key = f"{text} @ {pos}"
            surah_verse_key = f"{surah_name} / {verse_key}"
            confidence = judgment.get("confidence")
            localized_prob = judgment.get("localized_clip_probability")

            total_surviving += 1
            verse_counter[text] += 1
            verse_position_counter[verse_position_key] += 1
            window_counter[window] += 1
            char_counter[str(char or "")] += 1
            surah_verse_counter[surah_verse_key] += 1
            if isinstance(confidence, (int, float)):
                confidence_by_verse[text].append(float(confidence))
            if isinstance(localized_prob, (int, float)):
                localizer_prob_by_verse[text].append(float(localized_prob))

            examples.append(
                {
                    "sample_id": row.get("id"),
                    "surah_name": surah_name,
                    "verse_key": verse_key,
                    "text": text,
                    "position": pos,
                    "char": char,
                    "window": window,
                    "sequence_confidence": confidence,
                    "localized_ghunnah_probability": localized_prob,
                    "localized_predicted_labels": judgment.get("localized_predicted_labels") or [],
                    "detail": judgment.get("detail"),
                }
            )

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    examples.sort(
        key=lambda item: (
            item["sequence_confidence"] is not None,
            float(item["sequence_confidence"] or 0.0),
            float(item["localized_ghunnah_probability"] or 0.0),
        ),
        reverse=True,
    )

    top_verses = []
    for text, count in verse_counter.most_common():
        top_verses.append(
            {
                "text": text,
                "count": count,
                "mean_sequence_confidence": mean(confidence_by_verse[text]),
                "mean_localized_ghunnah_probability": mean(localizer_prob_by_verse[text]),
            }
        )

    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "checkpoint": str(PROJECT_ROOT / "checkpoints" / args.checkpoint_name),
        "total_surviving_ghunnah_as_madd": total_surviving,
        "top_verses": top_verses,
        "top_verse_positions": [
            {"key": key, "count": count}
            for key, count in verse_position_counter.most_common(args.show_top)
        ],
        "top_context_windows": [
            {"window": window, "count": count}
            for window, count in window_counter.most_common(args.show_top)
        ],
        "top_chars": [
            {"char": char, "count": count}
            for char, count in char_counter.most_common(args.show_top)
        ],
        "top_surah_verses": [
            {"surah_verse": key, "count": count}
            for key, count in surah_verse_counter.most_common(args.show_top)
        ],
        "top_examples": examples[: args.show_top],
    }

    print(f"Surviving ghunnah->madd misses : {total_surviving}")
    print("")
    print("Top verses:")
    if not top_verses:
        print("- None")
    else:
        for item in top_verses[: args.show_top]:
            seq_text = f"{item['mean_sequence_confidence']:.2f}" if item["mean_sequence_confidence"] is not None else "n/a"
            loc_text = f"{item['mean_localized_ghunnah_probability']:.2f}" if item["mean_localized_ghunnah_probability"] is not None else "n/a"
            print(f"- {safe_text(item['text'])}: count={item['count']} mean_seq_conf={seq_text} mean_loc_ghunnah={loc_text}")

    print("")
    print("Top verse positions:")
    for item in summary["top_verse_positions"]:
        print(f"- {safe_text(item['key'])}: {item['count']}")

    print("")
    print("Top context windows:")
    for item in summary["top_context_windows"]:
        print(f"- {safe_text(item['window'])}: {item['count']}")

    print("")
    print("Top examples:")
    for item in summary["top_examples"]:
        seq_text = f"{item['sequence_confidence']:.2f}" if isinstance(item["sequence_confidence"], (int, float)) else "n/a"
        loc_text = f"{item['localized_ghunnah_probability']:.2f}" if isinstance(item["localized_ghunnah_probability"], (int, float)) else "n/a"
        print(
            f"- {safe_text(item['sample_id'])} {safe_text(item['text'])} pos={item['position']} "
            f'char="{safe_text(item["char"])}" window={safe_text(item["window"])} '
            f"seq_conf={seq_text} loc_ghunnah={loc_text}"
        )

    save_json(summary, PROJECT_ROOT / args.output_json)
    print("")
    print(f"Saved context JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

