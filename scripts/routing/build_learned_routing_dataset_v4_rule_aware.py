from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "routing"))

from build_learned_routing_dataset import (
    ARABIC_LETTERS,
    audio_features,
    existing_or_glob,
    get_audio_path,
    get_id,
    get_text,
    has_transition_target,
    load_jsonl,
    resolve_path,
    text_features,
    vectorize_features,
    write_jsonl,
)


IKHFA_TRIGGER_LETTERS = set("تثجدذزسشصضطظفقك")
IDGHAM_TRIGGER_LETTERS = set("يرملون")
QALQALAH_LETTERS = set("قطبجد")
MADD_LETTERS = set("اوي")
GHUNNAH_LETTERS = set("نم")


def normalize_arabic(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def compact_text(text: str) -> str:
    return normalize_arabic(text).replace(" ", "")


def count_noon_followed_by(text: str, trigger_letters: set[str]) -> int:
    compact = compact_text(text)
    count = 0
    for idx, ch in enumerate(compact[:-1]):
        if ch == "ن" and compact[idx + 1] in trigger_letters:
            count += 1
    return count


def count_word_final_qalqalah(text: str) -> int:
    words = normalize_arabic(text).split()
    return sum(1 for word in words if word and word[-1] in QALQALAH_LETTERS)


def count_any_qalqalah(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in QALQALAH_LETTERS)


def count_madd_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in MADD_LETTERS)


def count_ghunnah_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in GHUNNAH_LETTERS)


def rule_aware_text_features(text: str) -> dict[str, float]:
    text = normalize_arabic(text)
    compact = text.replace(" ", "")
    total_chars = max(1, len(compact))
    words = text.split()
    total_words = max(1, len(words))

    ikhfa_count = count_noon_followed_by(text, IKHFA_TRIGGER_LETTERS)
    idgham_count = count_noon_followed_by(text, IDGHAM_TRIGGER_LETTERS)
    transition_count = ikhfa_count + idgham_count

    qalqalah_any_count = count_any_qalqalah(text)
    qalqalah_final_count = count_word_final_qalqalah(text)

    madd_count = count_madd_letters(text)
    ghunnah_count = count_ghunnah_letters(text)

    return {
        "rule_ikhfa_candidate_count": float(ikhfa_count),
        "rule_idgham_candidate_count": float(idgham_count),
        "rule_transition_candidate_count": float(transition_count),
        "rule_has_ikhfa_candidate": 1.0 if ikhfa_count > 0 else 0.0,
        "rule_has_idgham_candidate": 1.0 if idgham_count > 0 else 0.0,
        "rule_has_transition_candidate": 1.0 if transition_count > 0 else 0.0,

        "rule_qalqalah_any_count": float(qalqalah_any_count),
        "rule_qalqalah_final_count": float(qalqalah_final_count),
        "rule_has_qalqalah_any": 1.0 if qalqalah_any_count > 0 else 0.0,
        "rule_has_qalqalah_final": 1.0 if qalqalah_final_count > 0 else 0.0,
        "rule_qalqalah_any_ratio": float(qalqalah_any_count) / total_chars,
        "rule_qalqalah_final_ratio": float(qalqalah_final_count) / total_words,

        "rule_madd_letter_count": float(madd_count),
        "rule_madd_letter_ratio": float(madd_count) / total_chars,
        "rule_has_madd_proxy": 1.0 if madd_count > 0 else 0.0,

        "rule_ghunnah_letter_count": float(ghunnah_count),
        "rule_ghunnah_letter_ratio": float(ghunnah_count) / total_chars,
        "rule_has_ghunnah_proxy": 1.0 if ghunnah_count > 0 else 0.0,

        "rule_transition_and_qalqalah": 1.0 if transition_count > 0 and qalqalah_any_count > 0 else 0.0,
        "rule_duration_and_transition": 1.0 if madd_count > 0 and transition_count > 0 else 0.0,
        "rule_duration_and_burst": 1.0 if madd_count > 0 and qalqalah_any_count > 0 else 0.0,
        "rule_all_three_proxy": 1.0 if madd_count > 0 and transition_count > 0 and qalqalah_any_count > 0 else 0.0,
    }


RULE_FEATURE_NAMES = [
    "rule_ikhfa_candidate_count",
    "rule_idgham_candidate_count",
    "rule_transition_candidate_count",
    "rule_has_ikhfa_candidate",
    "rule_has_idgham_candidate",
    "rule_has_transition_candidate",
    "rule_qalqalah_any_count",
    "rule_qalqalah_final_count",
    "rule_has_qalqalah_any",
    "rule_has_qalqalah_final",
    "rule_qalqalah_any_ratio",
    "rule_qalqalah_final_ratio",
    "rule_madd_letter_count",
    "rule_madd_letter_ratio",
    "rule_has_madd_proxy",
    "rule_ghunnah_letter_count",
    "rule_ghunnah_letter_ratio",
    "rule_has_ghunnah_proxy",
    "rule_transition_and_qalqalah",
    "rule_duration_and_transition",
    "rule_duration_and_burst",
    "rule_all_three_proxy",
]


def group_key_for_sample(sample: dict[str, Any], group_by: str) -> str:
    text = normalize_arabic(sample.get("text", ""))
    reciter_id = str(sample.get("reciter_id") or "")
    verse_key = str(sample.get("quranjson_verse_key") or "")
    surah_name = str(sample.get("surah_name") or "")
    sample_id = str(sample.get("id") or "")

    if group_by == "text":
        return f"text::{text or sample_id}"

    if group_by == "verse_key":
        key = f"{surah_name}::{verse_key}"
        return f"verse::{key if key.strip(':') else text or sample_id}"

    if group_by == "reciter_text":
        key = f"{reciter_id}::{text}"
        return f"reciter_text::{key if key.strip(':') else sample_id}"

    if group_by == "sample_id":
        return f"id::{sample_id}"

    raise ValueError(f"Unsupported group_by: {group_by}")


def has_burst_target_v4(row: dict[str, Any]) -> bool:
    if "burst_label" in row:
        return bool(int(row.get("burst_label") or 0))

    rules: list[str] = []

    for key in ["canonical_rules", "rules"]:
        value = row.get(key)
        if isinstance(value, list):
            rules.extend(str(item).lower() for item in value)

    spans = row.get("rule_spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict) and span.get("rule"):
                rules.append(str(span["rule"]).lower())

    return any("qalqalah" in rule for rule in rules)


def ensure_sample(
    samples: dict[str, dict[str, Any]],
    row: dict[str, Any],
    source: str,
    index: int,
) -> dict[str, Any]:
    row_id = get_id(row, f"{source}_{index:06d}")
    audio_path = get_audio_path(row)
    text = get_text(row)

    if row_id not in samples:
        samples[row_id] = {
            "id": row_id,
            "audio_path": audio_path,
            "text": text,
            "reciter_id": row.get("reciter_id", ""),
            "surah_name": row.get("surah_name", ""),
            "quranjson_verse_key": row.get("quranjson_verse_key", ""),
            "sources": [],
            "targets": {
                "use_duration": 0,
                "use_transition": 0,
                "use_burst": 0,
            },
        }

    sample = samples[row_id]

    if source not in sample["sources"]:
        sample["sources"].append(source)

    if not sample.get("audio_path") and audio_path:
        sample["audio_path"] = audio_path
    if not sample.get("text") and text:
        sample["text"] = text

    for key in ["reciter_id", "surah_name", "quranjson_verse_key"]:
        if not sample.get(key) and row.get(key):
            sample[key] = row.get(key)

    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--transition-manifest", default="data/manifests/retasy_transition_multilabel_gold_only.jsonl")
    parser.add_argument("--burst-manifest", default="data/manifests/retasy_burst_subset.jsonl")
    parser.add_argument("--output-jsonl", default="data/manifests/learned_routing_dataset_v4_rule_aware_group_text.jsonl")
    parser.add_argument("--output-feature-config", default="configs/learned_routing_features_v4_rule_aware_group_text.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--group-by", default="text", choices=["text", "verse_key", "reciter_text", "sample_id"])
    args = parser.parse_args()

    duration_path = existing_or_glob(args.duration_manifest, "*duration*.jsonl")
    transition_path = existing_or_glob(args.transition_manifest, "*transition_multilabel_gold_only*.jsonl")
    burst_path = existing_or_glob(args.burst_manifest, "*burst*.jsonl")

    print("Rule-aware grouped routing dataset sources")
    print("------------------------------------------")
    print(f"duration  : {duration_path}")
    print(f"transition: {transition_path}")
    print(f"burst     : {burst_path}")
    print(f"group_by  : {args.group_by}")

    samples: dict[str, dict[str, Any]] = {}

    for i, row in enumerate(load_jsonl(duration_path)):
        sample = ensure_sample(samples, row, "duration", i)
        sample["targets"]["use_duration"] = 1

    for i, row in enumerate(load_jsonl(transition_path)):
        sample = ensure_sample(samples, row, "transition", i)
        sample["targets"]["use_transition"] = int(has_transition_target(row))

    for i, row in enumerate(load_jsonl(burst_path)):
        sample = ensure_sample(samples, row, "burst", i)
        sample["targets"]["use_burst"] = int(has_burst_target_v4(row))

    base_feature_names = (
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

    feature_names = list(base_feature_names) + list(RULE_FEATURE_NAMES)

    grouped: dict[str, list[str]] = {}
    for sample_id, sample in samples.items():
        key = group_key_for_sample(sample, args.group_by)
        grouped.setdefault(key, []).append(sample_id)

    rng = random.Random(args.seed)
    group_keys = list(grouped.keys())
    rng.shuffle(group_keys)

    target_val_samples = int(round(len(samples) * args.val_fraction))
    val_groups = set()
    val_sample_count = 0

    for key in group_keys:
        if val_sample_count >= target_val_samples:
            break
        val_groups.add(key)
        val_sample_count += len(grouped[key])

    rows: list[dict[str, Any]] = []

    for sample_id, sample in samples.items():
        fmap: dict[str, float] = {}
        fmap.update(audio_features(sample.get("audio_path", "")))
        fmap.update(text_features(sample.get("text", "")))
        fmap.update(rule_aware_text_features(sample.get("text", "")))

        targets = sample["targets"]
        group_key = group_key_for_sample(sample, args.group_by)

        row = {
            "id": sample["id"],
            "audio_path": sample.get("audio_path", ""),
            "text": sample.get("text", ""),
            "reciter_id": sample.get("reciter_id", ""),
            "surah_name": sample.get("surah_name", ""),
            "quranjson_verse_key": sample.get("quranjson_verse_key", ""),
            "sources": sample["sources"],
            "label_source": "trusted_retasy_routing",
            "sample_weight": 1.0,
            "group_by": args.group_by,
            "group_key": group_key,
            "feature_names": feature_names,
            "features": vectorize_features(fmap, feature_names),
            "target_names": ["use_duration", "use_transition", "use_burst"],
            "targets": [
                int(targets["use_duration"]),
                int(targets["use_transition"]),
                int(targets["use_burst"]),
            ],
            "split": "val" if group_key in val_groups else "train",
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
                "group_by": args.group_by,
                "rule_aware_features": RULE_FEATURE_NAMES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    target_counts = Counter()
    split_counts = Counter(row["split"] for row in rows)
    group_split_counts = Counter()

    for key in grouped:
        group_split_counts["val" if key in val_groups else "train"] += 1

    for row in rows:
        for name, value in zip(row["target_names"], row["targets"]):
            if value:
                target_counts[name] += 1

    train_groups = {row["group_key"] for row in rows if row["split"] == "train"}
    val_groups_actual = {row["group_key"] for row in rows if row["split"] == "val"}
    overlap = train_groups & val_groups_actual

    print("Built rule-aware grouped learned routing dataset")
    print("------------------------------------------------")
    print(f"rows              : {len(rows)}")
    print(f"groups            : {len(grouped)}")
    print(f"split_counts      : {dict(split_counts)}")
    print(f"group_split_counts: {dict(group_split_counts)}")
    print(f"target_counts     : {dict(target_counts)}")
    print(f"features          : {len(feature_names)}")
    print(f"rule_features     : {len(RULE_FEATURE_NAMES)}")
    print(f"group_overlap     : {len(overlap)}")
    print(f"output            : {output_jsonl}")
    print(f"feature cfg       : {output_cfg}")

    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} overlapping groups")


if __name__ == "__main__":
    main()
