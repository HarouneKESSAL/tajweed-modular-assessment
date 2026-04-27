from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

from tajweed_assessment.data.labels import normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.burst.qalqalah_cnn import QalqalahCNN
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, save_json


FOCUS_RULES = ("madd", "ghunnah")
DEFAULT_DURATION_MANIFEST = "data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl"
DEFAULT_OUTPUT_JSON = "data/analysis/duration_hardcases.json"


def preferred_transition_checkpoint() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "transition_module_hardcase.pt"
    if preferred.exists():
        return "transition_module_hardcase.pt"
    return "transition_module.pt"


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


def classify_duration_hardcase(expected: str, predicted: str) -> str:
    if expected == "ghunnah" and predicted == "madd":
        return "ghunnah_as_madd"
    if expected == "ghunnah" and predicted == "none":
        return "missed_ghunnah"
    if expected == "madd" and predicted == "ghunnah":
        return "madd_as_ghunnah"
    if expected == "madd" and predicted == "none":
        return "missed_madd"
    if expected == predicted and expected in FOCUS_RULES:
        return f"correct_{expected}"
    return "other"


def recommend_weight(category: str, confidence: float | None) -> float:
    base = 1.0
    if category in {"ghunnah_as_madd", "missed_ghunnah"}:
        base = 4.5
    elif category in {"madd_as_ghunnah", "missed_madd"}:
        base = 2.5
    elif category == "correct_ghunnah":
        base = 1.5
    elif category in {"correct_madd", "other"}:
        base = 1.0
    if isinstance(confidence, (int, float)) and confidence >= 0.9 and category not in {"correct_ghunnah", "correct_madd"}:
        base += 0.5
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_DURATION_MANIFEST)
    parser.add_argument("--checkpoint-name", default="duration_module.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--show-top", type=int, default=15)
    args = parser.parse_args()

    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]

    pipeline = TajweedInferencePipeline(
        duration_module=load_duration_module(args.checkpoint_name),
        transition_module=load_transition_module(),
        burst_module=load_burst_module(),
        device="cpu",
    )
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)

    category_counts = Counter()
    sample_scores: dict[str, float] = {}
    hardcases: list[dict] = []
    total_focus = 0
    correct_focus = 0

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
            transition_mfcc=mfcc,
            transition_ssl=ssl,
            burst_x=mfcc,
        )

        sample_id = str(row.get("id") or row.get("sample_id") or row.get("audio_path") or f"sample_{idx}")
        sample_best_weight = sample_scores.get(sample_id, 1.0)

        for judgment in result.get("module_judgments", []):
            if judgment.get("source_module") != "duration":
                continue
            expected = judgment.get("rule")
            predicted = judgment.get("predicted_rule")
            if expected not in FOCUS_RULES:
                continue
            total_focus += 1
            if judgment.get("is_correct"):
                correct_focus += 1

            category = classify_duration_hardcase(expected, predicted)
            category_counts[category] += 1
            weight = recommend_weight(category, judgment.get("confidence"))
            sample_best_weight = max(sample_best_weight, weight)

            if not judgment.get("is_correct") or category == "correct_ghunnah":
                pos = int(judgment["position"])
                char = canonical_chars[pos] if pos < len(canonical_chars) else None
                hardcases.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": row.get("audio_path"),
                        "text": row.get("normalized_text"),
                        "position": pos,
                        "char": char,
                        "expected_rule": expected,
                        "predicted_rule": predicted,
                        "confidence": judgment.get("confidence"),
                        "category": category,
                        "weight": weight,
                    }
                )

        sample_scores[sample_id] = sample_best_weight

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)} samples...")

    hardcases.sort(key=lambda item: (item["weight"], item["confidence"] or 0.0), reverse=True)
    summary = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "checkpoint": str(PROJECT_ROOT / "checkpoints" / args.checkpoint_name),
        "samples_analyzed": len(rows),
        "focus_positions": total_focus,
        "focus_accuracy": (correct_focus / total_focus) if total_focus else None,
        "category_counts": dict(category_counts),
        "weights": sample_scores,
        "hardcases": hardcases,
    }

    print(f"Samples analyzed : {len(rows)}")
    print(f"Focus positions  : {total_focus}")
    print(f"Focus accuracy   : {summary['focus_accuracy']:.3f}" if summary["focus_accuracy"] is not None else "Focus accuracy   : n/a")
    print("Hard-case categories:")
    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")
    print("")
    print(f"Top weighted rows (top {min(args.show_top, len(hardcases))}):")
    if not hardcases:
        print("- None")
    else:
        for item in hardcases[: args.show_top]:
            char_text = f' ("{safe_text(item["char"])}")' if item["char"] else ""
            confidence_text = f" confidence={item['confidence']:.2f}" if isinstance(item["confidence"], (int, float)) else ""
            print(
                f"- {safe_text(item['sample_id'])} position {item['position']}{char_text}: {item['category']} "
                f"expected={item['expected_rule']} predicted={item['predicted_rule']}{confidence_text} weight={item['weight']:.2f}"
            )

    save_json(summary, PROJECT_ROOT / args.output_json)
    print("")
    print(f"Saved hardcase JSON to {PROJECT_ROOT / args.output_json}")


if __name__ == "__main__":
    main()

