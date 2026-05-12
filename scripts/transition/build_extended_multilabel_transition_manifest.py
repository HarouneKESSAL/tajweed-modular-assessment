from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.models.transition.multilabel_transition_module import (
    transition_multilabel_label_names,
    transition_rules_to_multihot,
)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def normalize_rules(row: dict[str, Any]) -> list[str]:
    raw_rules = row.get("transition_rules") or row.get("transition_multilabel_rules") or []
    if not isinstance(raw_rules, list):
        raw_rules = [raw_rules]

    found = set()
    for rule in raw_rules:
        value = str(rule).lower()
        if "ikhfa" in value:
            found.add("ikhfa")
        if "idgham" in value:
            found.add("idgham")

    ordered = []
    for label in transition_multilabel_label_names():
        if label in found:
            ordered.append(label)
    return ordered


def normalize_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    label_names = transition_multilabel_label_names()
    rules = normalize_rules(row)
    combo = "+".join(rules) if rules else "none"

    out = dict(row)
    out["id"] = row.get("id") or row.get("sample_id")
    out["transition_label_names"] = label_names
    out["transition_multilabel_rules"] = rules
    out["transition_multihot"] = transition_rules_to_multihot(rules, label_names=label_names)
    out["transition_combo"] = combo
    out["label_source"] = "gold_transition_subset"
    out["sample_weight"] = 1.0
    return out


def normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any] | None:
    audio_path = row.get("candidate_audio_path") or row.get("audio_path")
    if not audio_path:
        return None

    source_manifest = str(row.get("source_manifest", ""))

    # Prefer Retasy audio candidates because they match your existing audio domain.
    if "retasy_train.jsonl" not in source_manifest:
        return None

    text = row.get("candidate_text_norm") or row.get("text") or row.get("aya_text_norm") or row.get("aya_text")
    if not text:
        return None

    label_names = transition_multilabel_label_names()
    rules = ["ikhfa", "idgham"]

    out = {
        "id": row.get("id") or row.get("sample_id"),
        "sample_id": row.get("sample_id") or row.get("id"),
        "audio_path": audio_path,
        "feature_path": row.get("feature_path", ""),
        "text": text,
        "reciter_id": row.get("reciter_id", ""),
        "surah_name": row.get("surah_name", ""),
        "quranjson_verse_key": row.get("quranjson_verse_key", ""),
        "transition_rules": rules,
        "transition_label_names": label_names,
        "transition_multilabel_rules": rules,
        "transition_multihot": transition_rules_to_multihot(rules, label_names=label_names),
        "transition_combo": "ikhfa+idgham",
        "label_source": "weak_text_pattern",
        "sample_weight": 0.4,
        "source_manifest": source_manifest,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-manifest", default="data/manifests/retasy_transition_multilabel.jsonl")
    parser.add_argument("--candidate-manifest", default="data/analysis/transition_both_label_candidates.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_transition_multilabel_extended.jsonl")
    parser.add_argument("--max-weak-both", type=int, default=150)
    args = parser.parse_args()

    gold_rows_raw = load_jsonl(resolve_path(args.gold_manifest))
    candidate_rows_raw = load_jsonl(resolve_path(args.candidate_manifest))

    gold_rows = [normalize_gold_row(row) for row in gold_rows_raw]

    existing_ids = {str(row.get("id") or row.get("sample_id")) for row in gold_rows}

    weak_rows = []
    for row in candidate_rows_raw:
        out = normalize_candidate_row(row)
        if out is None:
            continue

        row_id = str(out.get("id") or out.get("sample_id"))
        if row_id in existing_ids:
            continue

        weak_rows.append(out)

    # Stable deterministic cap.
    weak_rows = weak_rows[: args.max_weak_both]

    rows = gold_rows + weak_rows

    counts = Counter(row["transition_combo"] for row in rows)
    source_counts = Counter(row["label_source"] for row in rows)

    write_jsonl(resolve_path(args.output), rows)

    print("Built extended multi-label transition manifest")
    print("----------------------------------------------")
    print(f"gold rows       : {len(gold_rows)}")
    print(f"weak both rows  : {len(weak_rows)}")
    print(f"total rows      : {len(rows)}")
    print(f"combo_counts    : {dict(counts)}")
    print(f"source_counts   : {dict(source_counts)}")
    print(f"output          : {resolve_path(args.output)}")


if __name__ == "__main__":
    main()
