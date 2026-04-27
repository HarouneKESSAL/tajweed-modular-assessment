from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import re

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------
# Duration-rule helpers
# ---------------------------------------------------------

# Keep this narrow on purpose for the first real subset.
# You can expand later if your quranjson repo uses more names.
DEFAULT_DURATION_RULE_VOCAB = [
    "ghunnah",
    "madd_2",
    "madd_4",
    "madd_6",
    "madd_246",
]

_DURATION_RULE_RE = re.compile(r"^madd(_.*)?$|^ghunnah$", re.IGNORECASE)


def is_duration_rule(rule_name: str) -> bool:
    if not rule_name:
        return False
    return bool(_DURATION_RULE_RE.match(rule_name.strip()))


def normalize_rule_name(rule_name: str) -> str:
    return rule_name.strip().lower()


def extract_duration_rules(rule_spans: Sequence[Dict[str, Any]]) -> List[str]:
    rules: List[str] = []
    for span in rule_spans:
        rule = normalize_rule_name(str(span.get("rule", "")))
        if is_duration_rule(rule):
            rules.append(rule)
    return sorted(set(rules))


def multi_hot(rule_names: Sequence[str], vocab: Sequence[str]) -> List[int]:
    rule_set = set(rule_names)
    return [1 if label in rule_set else 0 for label in vocab]


# ---------------------------------------------------------
# Data containers
# ---------------------------------------------------------

@dataclass
class DurationSample:
    id: str
    hf_index: int
    surah_name: Optional[str]
    hf_surah_number: Optional[int]
    quranjson_surah_number: Optional[int]
    quranjson_verse_key: Optional[str]
    quranjson_verse_index: Optional[int]

    aya_text: str
    aya_text_norm: str
    duration_ms: Optional[int]

    final_label: Optional[str]
    golden: Optional[bool]

    reciter_id: Optional[str]
    reciter_country: Optional[str]
    reciter_gender: Optional[str]
    reciter_age: Optional[str]
    reciter_qiraah: Optional[str]

    original_audio_path: Optional[str]
    audio_path: Optional[str]

    rule_spans: List[Dict[str, Any]]
    duration_rules: List[str]
    target: List[int]


# ---------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_duration_samples(
    rows: Iterable[Dict[str, Any]],
    *,
    rule_vocab: Optional[Sequence[str]] = None,
    require_unique_match: bool = True,
    require_nonempty_rules: bool = True,
    drop_not_related_quran: bool = True,
) -> List[DurationSample]:
    rows = list(rows)

    # Build a dynamic vocab if none was provided.
    if rule_vocab is None:
        discovered = set()
        for row in rows:
            for rule in extract_duration_rules(row.get("rule_spans", [])):
                discovered.add(rule)
        rule_vocab = sorted(discovered) if discovered else list(DEFAULT_DURATION_RULE_VOCAB)

    samples: List[DurationSample] = []

    for row in rows:
        if require_unique_match and row.get("match_status") != "matched_unique":
            continue

        if drop_not_related_quran and row.get("final_label") == "not_related_quran":
            continue

        duration_rules = extract_duration_rules(row.get("rule_spans", []))
        if require_nonempty_rules and not duration_rules:
            continue

        target = multi_hot(duration_rules, rule_vocab)

        samples.append(
            DurationSample(
                id=str(row["id"]),
                hf_index=int(row["hf_index"]),
                surah_name=row.get("surah_name"),
                hf_surah_number=row.get("hf_surah_number"),
                quranjson_surah_number=row.get("quranjson_surah_number"),
                quranjson_verse_key=row.get("quranjson_verse_key"),
                quranjson_verse_index=row.get("quranjson_verse_index"),
                aya_text=row.get("aya_text", ""),
                aya_text_norm=row.get("aya_text_norm", ""),
                duration_ms=row.get("duration_ms"),
                final_label=row.get("final_label"),
                golden=row.get("golden"),
                reciter_id=row.get("reciter_id"),
                reciter_country=row.get("reciter_country"),
                reciter_gender=row.get("reciter_gender"),
                reciter_age=row.get("reciter_age"),
                reciter_qiraah=row.get("reciter_qiraah"),
                original_audio_path=row.get("original_audio_path"),
                audio_path=row.get("audio_path"),
                rule_spans=list(row.get("rule_spans", [])),
                duration_rules=duration_rules,
                target=target,
            )
        )

    return samples


# ---------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------

class RealDurationDataset(Dataset):
    """
    First real-data dataset for the duration specialist.

    This dataset currently returns:
      - metadata
      - normalized target labels
      - raw rule spans

    It does NOT decode audio yet.
    That is intentional: get the real subset clean first, then wire audio in.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        rule_vocab: Optional[Sequence[str]] = None,
        require_unique_match: bool = True,
        require_nonempty_rules: bool = True,
        drop_not_related_quran: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = load_jsonl(self.manifest_path)
        self.samples = build_duration_samples(
            self.rows,
            rule_vocab=rule_vocab,
            require_unique_match=require_unique_match,
            require_nonempty_rules=require_nonempty_rules,
            drop_not_related_quran=drop_not_related_quran,
        )

        if rule_vocab is None:
            discovered = set()
            for s in self.samples:
                discovered.update(s.duration_rules)
            self.rule_vocab = sorted(discovered) if discovered else list(DEFAULT_DURATION_RULE_VOCAB)
        else:
            self.rule_vocab = list(rule_vocab)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "id": s.id,
            "hf_index": s.hf_index,
            "surah_name": s.surah_name,
            "hf_surah_number": s.hf_surah_number,
            "quranjson_surah_number": s.quranjson_surah_number,
            "quranjson_verse_key": s.quranjson_verse_key,
            "quranjson_verse_index": s.quranjson_verse_index,
            "aya_text": s.aya_text,
            "aya_text_norm": s.aya_text_norm,
            "duration_ms": s.duration_ms,
            "final_label": s.final_label,
            "golden": s.golden,
            "reciter_id": s.reciter_id,
            "reciter_country": s.reciter_country,
            "reciter_gender": s.reciter_gender,
            "reciter_age": s.reciter_age,
            "reciter_qiraah": s.reciter_qiraah,
            "original_audio_path": s.original_audio_path,
            "audio_path": s.audio_path,
            "rule_spans": s.rule_spans,
            "duration_rules": s.duration_rules,
            "target": torch.tensor(s.target, dtype=torch.float32),
        }

    def label_vocab(self) -> List[str]:
        return list(self.rule_vocab)

    def summary(self) -> Dict[str, Any]:
        label_counts = {label: 0 for label in self.rule_vocab}
        reciter_counts: Dict[str, int] = {}

        for s in self.samples:
            for label in s.duration_rules:
                if label in label_counts:
                    label_counts[label] += 1

            reciter_key = s.reciter_id or "Unknown"
            reciter_counts[reciter_key] = reciter_counts.get(reciter_key, 0) + 1

        return {
            "manifest_path": str(self.manifest_path),
            "num_samples": len(self.samples),
            "label_vocab": list(self.rule_vocab),
            "label_counts": label_counts,
            "num_unique_reciters": len(reciter_counts),
        }