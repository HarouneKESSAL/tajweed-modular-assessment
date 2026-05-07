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

from build_learned_routing_dataset_v4_rule_aware import (
    normalize_arabic,
    rule_aware_text_features,
)
from build_learned_routing_dataset import (
    ARABIC_LETTERS,
    audio_features,
    text_features,
    vectorize_features,
    resolve_path,
    write_jsonl,
)


BASE_FEATURE_NAMES = (
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

FEATURE_NAMES = list(BASE_FEATURE_NAMES) + list(RULE_FEATURE_NAMES)
TARGET_NAMES = ["use_duration", "use_transition", "use_burst"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_key(text: str) -> str:
    text = normalize_arabic(text)
    return f"text::{text}"


def feature_vector(audio_path: str, text: str) -> list[float]:
    fmap: dict[str, float] = {}
    fmap.update(audio_features(audio_path))
    fmap.update(text_features(text))
    fmap.update(rule_aware_text_features(text))
    return vectorize_features(fmap, FEATURE_NAMES)


def convert_retasy_row(row: dict[str, Any]) -> dict[str, Any]:
    text = row.get("text", "")
    audio_path = row.get("audio_path", "")

    return {
        "id": row.get("id"),
        "audio_path": audio_path,
        "text": text,
        "reciter_id": row.get("reciter_id", ""),
        "surah_name": row.get("surah_name", ""),
        "quranjson_verse_key": row.get("quranjson_verse_key", ""),
        "ayah_key": row.get("ayah_key", ""),
        "sources": row.get("sources", ["retasy"]),
        "label_source": "trusted_retasy_routing",
        "sample_weight": 1.0,
        "group_by": "text",
        "group_key": group_key(text),
        "feature_names": FEATURE_NAMES,
        "features": feature_vector(audio_path, text),
        "target_names": TARGET_NAMES,
        "targets": row["targets"],
    }


def convert_hf_row(row: dict[str, Any], hf_weight: float) -> dict[str, Any]:
    text = row.get("text", "")
    audio_path = row.get("audio_path", "")

    return {
        "id": row.get("id") or row.get("sample_id"),
        "audio_path": audio_path,
        "text": text,
        "reciter_id": row.get("reciter_id", ""),
        "surah_name": row.get("surah_name", ""),
        "quranjson_verse_key": row.get("quranjson_verse_key", ""),
        "ayah_key": row.get("ayah_key", ""),
        "sources": ["hf_quran_md_ayahs"],
        "label_source": row.get("label_source", "weak_hf_quran_md_text_pattern"),
        "sample_weight": float(row.get("sample_weight", hf_weight)),
        "weak_rule_features": row.get("weak_rule_features", {}),
        "group_by": "text",
        "group_key": group_key(text),
        "feature_names": FEATURE_NAMES,
        "features": feature_vector(audio_path, text),
        "target_names": TARGET_NAMES,
        "targets": row["targets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retasy-dataset", default="data/manifests/learned_routing_dataset_v4_rule_aware_group_text.jsonl")
    parser.add_argument("--hf-manifest", default="data/manifests/hf_quran_md_ayah_routing_weak_all_ayahs_r1.jsonl")
    parser.add_argument("--output-jsonl", default="data/manifests/learned_routing_dataset_v5_retasy_hf_rule_aware_group_text.jsonl")
    parser.add_argument("--output-feature-config", default="configs/learned_routing_features_v5_retasy_hf_rule_aware_group_text.json")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--hf-weight", type=float, default=0.25)
    args = parser.parse_args()

    retasy_path = resolve_path(args.retasy_dataset)
    hf_path = resolve_path(args.hf_manifest)

    if not retasy_path.exists():
        raise FileNotFoundError(f"Missing Retasy routing dataset: {retasy_path}")
    if not hf_path.exists():
        raise FileNotFoundError(f"Missing HF routing manifest: {hf_path}")

    retasy_rows = [convert_retasy_row(row) for row in load_jsonl(retasy_path)]
    hf_rows = [convert_hf_row(row, args.hf_weight) for row in load_jsonl(hf_path)]

    rows = retasy_rows + hf_rows

    grouped: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(row["group_key"], []).append(idx)

    rng = random.Random(args.seed)
    keys = list(grouped.keys())
    rng.shuffle(keys)

    target_val = int(round(len(rows) * args.val_fraction))
    val_groups = set()
    val_count = 0

    for key in keys:
        if val_count >= target_val:
            break
        val_groups.add(key)
        val_count += len(grouped[key])

    for row in rows:
        row["split"] = "val" if row["group_key"] in val_groups else "train"

    train_groups = {row["group_key"] for row in rows if row["split"] == "train"}
    val_groups_actual = {row["group_key"] for row in rows if row["split"] == "val"}
    overlap = train_groups & val_groups_actual

    output_jsonl = resolve_path(args.output_jsonl)
    write_jsonl(output_jsonl, rows)

    output_cfg = resolve_path(args.output_feature_config)
    output_cfg.parent.mkdir(parents=True, exist_ok=True)
    output_cfg.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "target_names": TARGET_NAMES,
                "group_by": "text",
                "rule_aware_features": RULE_FEATURE_NAMES,
                "retasy_dataset": str(retasy_path),
                "hf_manifest": str(hf_path),
                "hf_weight": args.hf_weight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["label_source"] for row in rows)
    target_counts = Counter()
    combo_counts = Counter()
    group_counts = Counter(row["split"] for row in rows)

    for row in rows:
        active = []
        for name, value in zip(row["target_names"], row["targets"]):
            if value:
                target_counts[name] += 1
                active.append(name)
        combo_counts["+".join(active) if active else "none"] += 1

    print("Built v5 Retasy + HF rule-aware routing dataset")
    print("------------------------------------------------")
    print(f"rows          : {len(rows)}")
    print(f"retasy_rows   : {len(retasy_rows)}")
    print(f"hf_rows       : {len(hf_rows)}")
    print(f"groups        : {len(grouped)}")
    print(f"split_counts  : {dict(split_counts)}")
    print(f"source_counts : {dict(source_counts)}")
    print(f"target_counts : {dict(target_counts)}")
    print(f"combo_counts  : {dict(combo_counts)}")
    print(f"group_overlap : {len(overlap)}")
    print(f"features      : {len(FEATURE_NAMES)}")
    print(f"output        : {output_jsonl}")
    print(f"feature cfg   : {output_cfg}")

    if overlap:
        raise RuntimeError(f"group leakage detected: {len(overlap)} overlapping groups")


if __name__ == "__main__":
    main()
