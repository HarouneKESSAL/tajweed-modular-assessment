from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءةىؤئ")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def existing_or_glob(path_value: str, pattern: str) -> Path | None:
    if path_value:
        path = resolve_path(path_value)
        if path.exists():
            return path

    matches = sorted((PROJECT_ROOT / "data/manifests").glob(pattern))
    return matches[0] if matches else None


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "normalized_text", "source_text", "aya_text_norm", "aya_text"]:
        value = row.get(key)
        if value:
            return normalize_text(str(value))
    return ""


def get_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("parent_id") or fallback)


def get_audio_path(row: dict[str, Any]) -> str:
    return str(row.get("audio_path") or "")


def collect_rule_strings(row: dict[str, Any]) -> list[str]:
    rules: list[str] = []

    for key in [
        "canonical_rules",
        "transition_rules",
        "transition_multilabel_rules",
        "rules",
    ]:
        value = row.get(key)
        if isinstance(value, list):
            rules.extend(str(item).lower() for item in value)
        elif value:
            rules.append(str(value).lower())

    spans = row.get("rule_spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict) and span.get("rule"):
                rules.append(str(span["rule"]).lower())

    return rules


def has_duration_target(row: dict[str, Any]) -> bool:
    rules = collect_rule_strings(row)
    return any(("madd" in rule or "ghunnah" in rule) for rule in rules)


def has_transition_target(row: dict[str, Any]) -> bool:
    multihot = row.get("transition_multihot")
    if isinstance(multihot, list) and any(float(x) >= 0.5 for x in multihot):
        return True

    rules = collect_rule_strings(row)
    return any(("ikhfa" in rule or "idgham" in rule) for rule in rules)


def has_burst_target(row: dict[str, Any]) -> bool:
    rules = collect_rule_strings(row)
    return any("qalqalah" in rule for rule in rules)


def audio_features(audio_path: str) -> dict[str, float]:
    if not audio_path:
        return {
            "audio_duration_sec": 0.0,
            "audio_log_duration": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_zcr": 0.0,
        }

    path = resolve_path(audio_path)
    try:
        audio, sample_rate = sf.read(str(path), always_2d=True)
        if audio.shape[1] > 1:
            values = audio.mean(axis=1)
        else:
            values = audio[:, 0]

        n = len(values)
        if n == 0:
            raise ValueError("empty audio")

        duration = float(n) / float(sample_rate)
        rms = float((values ** 2).mean() ** 0.5)
        peak = float(abs(values).max())
        zcr = float(((values[:-1] * values[1:]) < 0).mean()) if n > 1 else 0.0

        return {
            "audio_duration_sec": duration,
            "audio_log_duration": math.log1p(duration),
            "audio_rms": rms,
            "audio_peak": peak,
            "audio_zcr": zcr,
        }
    except Exception:
        return {
            "audio_duration_sec": 0.0,
            "audio_log_duration": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_zcr": 0.0,
        }


def text_features(text: str) -> dict[str, float]:
    text = normalize_text(text)
    compact = text.replace(" ", "")
    total_chars = max(1, len(compact))

    feats: dict[str, float] = {
        "text_char_count": float(len(compact)),
        "text_word_count": float(len(text.split())),
        "text_log_char_count": math.log1p(len(compact)),
        "text_log_word_count": math.log1p(len(text.split())),
    }

    for letter in ARABIC_LETTERS:
        feats[f"char_freq_{letter}"] = compact.count(letter) / total_chars

    # Very rough rule-trigger features. These are text-derived features, not labels.
    ikhfa_triggers = set("تثجدذزسشصضطظفقك")
    idgham_triggers = set("يرملون")
    qalqalah_letters = set("قطبجد")

    feats["contains_noon"] = 1.0 if "ن" in compact else 0.0
    feats["contains_meem"] = 1.0 if "م" in compact else 0.0
    feats["ikhfa_trigger_ratio"] = sum(compact.count(ch) for ch in ikhfa_triggers) / total_chars
    feats["idgham_trigger_ratio"] = sum(compact.count(ch) for ch in idgham_triggers) / total_chars
    feats["qalqalah_letter_ratio"] = sum(compact.count(ch) for ch in qalqalah_letters) / total_chars

    return feats


def vectorize_features(feature_map: dict[str, float], feature_names: list[str]) -> list[float]:
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--transition-manifest", default="data/manifests/retasy_transition_multilabel_gold_only.jsonl")
    parser.add_argument("--burst-manifest", default="")
    parser.add_argument("--output-jsonl", default="data/manifests/learned_routing_dataset_v1.jsonl")
    parser.add_argument("--output-feature-config", default="configs/learned_routing_features_v1.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    duration_path = existing_or_glob(args.duration_manifest, "*duration*.jsonl")
    transition_path = existing_or_glob(args.transition_manifest, "*transition_multilabel_gold_only*.jsonl")
    burst_path = existing_or_glob(args.burst_manifest, "*burst*.jsonl")

    print("Routing dataset sources")
    print("-----------------------")
    print(f"duration  : {duration_path}")
    print(f"transition: {transition_path}")
    print(f"burst     : {burst_path}")

    samples: dict[str, dict[str, Any]] = {}

    def ensure_sample(row: dict[str, Any], source: str, index: int) -> dict[str, Any]:
        row_id = get_id(row, f"{source}_{index:06d}")
        audio_path = get_audio_path(row)
        text = get_text(row)

        key = row_id
        if key not in samples:
            samples[key] = {
                "id": row_id,
                "audio_path": audio_path,
                "text": text,
                "sources": [],
                "targets": {
                    "use_duration": 0,
                    "use_transition": 0,
                    "use_burst": 0,
                },
            }

        if source not in samples[key]["sources"]:
            samples[key]["sources"].append(source)

        if not samples[key].get("audio_path") and audio_path:
            samples[key]["audio_path"] = audio_path
        if not samples[key].get("text") and text:
            samples[key]["text"] = text

        return samples[key]

    for i, row in enumerate(load_jsonl(duration_path)):
        sample = ensure_sample(row, "duration", i)

        # The duration manifest is already a duration-routing corpus.
        # Some rows may not expose madd/ghunnah labels in the generic fields
        # checked by has_duration_target(), so source membership is the safest
        # v1 routing target.
        sample["targets"]["use_duration"] = 1

    for i, row in enumerate(load_jsonl(transition_path)):
        sample = ensure_sample(row, "transition", i)
        sample["targets"]["use_transition"] = int(has_transition_target(row))

    for i, row in enumerate(load_jsonl(burst_path)):
        sample = ensure_sample(row, "burst", i)
        sample["targets"]["use_burst"] = int(has_burst_target(row))

    feature_names = (
        ["audio_duration_sec", "audio_log_duration", "audio_rms", "audio_peak", "audio_zcr"]
        + ["text_char_count", "text_word_count", "text_log_char_count", "text_log_word_count"]
        + [f"char_freq_{letter}" for letter in ARABIC_LETTERS]
        + [
            "contains_noon",
            "contains_meem",
            "ikhfa_trigger_ratio",
            "idgham_trigger_ratio",
            "qalqalah_letter_ratio",
        ]
    )

    rows = []
    rng = random.Random(args.seed)
    ids = list(samples.keys())
    rng.shuffle(ids)

    val_count = int(round(len(ids) * args.val_fraction))
    val_ids = set(ids[:val_count])

    for sample_id in ids:
        sample = samples[sample_id]
        fmap = {}
        fmap.update(audio_features(sample.get("audio_path", "")))
        fmap.update(text_features(sample.get("text", "")))

        targets = sample["targets"]
        row = {
            "id": sample["id"],
            "audio_path": sample.get("audio_path", ""),
            "text": sample.get("text", ""),
            "sources": sample["sources"],
            "feature_names": feature_names,
            "features": vectorize_features(fmap, feature_names),
            "target_names": ["use_duration", "use_transition", "use_burst"],
            "targets": [
                int(targets["use_duration"]),
                int(targets["use_transition"]),
                int(targets["use_burst"]),
            ],
            "split": "val" if sample_id in val_ids else "train",
        }
        rows.append(row)

    output_jsonl = resolve_path(args.output_jsonl)
    output_cfg = resolve_path(args.output_feature_config)

    write_jsonl(output_jsonl, rows)
    output_cfg.parent.mkdir(parents=True, exist_ok=True)
    output_cfg.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "target_names": ["use_duration", "use_transition", "use_burst"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    target_counts = Counter()
    split_counts = Counter(row["split"] for row in rows)
    for row in rows:
        for name, value in zip(row["target_names"], row["targets"]):
            if value:
                target_counts[name] += 1

    print("Built learned routing dataset")
    print("-----------------------------")
    print(f"rows         : {len(rows)}")
    print(f"split_counts : {dict(split_counts)}")
    print(f"target_counts: {dict(target_counts)}")
    print(f"features     : {len(feature_names)}")
    print(f"output       : {output_jsonl}")
    print(f"feature cfg  : {output_cfg}")


if __name__ == "__main__":
    main()
