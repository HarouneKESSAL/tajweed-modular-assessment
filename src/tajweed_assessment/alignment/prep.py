from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ARABIC_KEEP_SPACE_RE = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")

_DURATION_RULE_RE = re.compile(r"^madd(_.*)?$|^ghunnah$", re.IGNORECASE)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_rule_name(rule_name: str) -> str:
    return str(rule_name).strip().lower()


def is_duration_rule(rule_name: str) -> bool:
    return bool(_DURATION_RULE_RE.match(normalize_rule_name(rule_name)))


def coarse_duration_group(rule_name: str) -> Optional[str]:
    rule = normalize_rule_name(rule_name)
    if rule == "ghunnah":
        return "ghunnah"
    if rule.startswith("madd"):
        return "has_madd"
    return None


def normalize_char_for_alignment(ch: str) -> str:
    if ch == "ـ":
        return ""
    ch = _ARABIC_DIACRITICS_RE.sub("", ch)
    ch = (
        ch.replace("ٱ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    ch = _NON_ARABIC_KEEP_SPACE_RE.sub(" ", ch)
    return ch


def normalize_text_for_alignment(text: str) -> tuple[str, List[int]]:
    """
    Returns:
      - normalized text
      - mapping from normalized character index -> original character index

    Spaces are preserved but collapsed.
    """
    out_chars: List[str] = []
    char_map: List[int] = []

    for orig_idx, ch in enumerate(text):
        norm = normalize_char_for_alignment(ch)
        if not norm:
            continue

        for subch in norm:
            if subch.isspace():
                if out_chars and out_chars[-1] != " ":
                    out_chars.append(" ")
                    char_map.append(orig_idx)
            else:
                out_chars.append(subch)
                char_map.append(orig_idx)

    while out_chars and out_chars[0] == " ":
        out_chars.pop(0)
        char_map.pop(0)
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        char_map.pop()

    raw_text = "".join(out_chars)
    text_norm = _WHITESPACE_RE.sub(" ", raw_text).strip()

    if text_norm != raw_text:
        new_map: List[int] = []
        j = 0
        for ch in text_norm:
            while j < len(raw_text) and raw_text[j] != ch:
                j += 1
            if j < len(raw_text):
                new_map.append(char_map[j])
                j += 1
        char_map = new_map

    return text_norm, char_map


def build_original_char_rule_labels(text: str, rule_spans: List[Dict[str, Any]]) -> List[List[str]]:
    labels: List[List[str]] = [[] for _ in range(len(text))]

    for span in rule_spans:
        rule = normalize_rule_name(span.get("rule", ""))
        if not is_duration_rule(rule):
            continue

        start = int(span.get("start", 0))
        end = int(span.get("end", 0))

        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        for i in range(start, end):
            labels[i].append(rule)

    return [sorted(set(x)) for x in labels]


def _norm_positions_for_original_span(
    original_start: int,
    original_end: int,
    norm_to_orig: List[int],
    normalized_text: str,
    *,
    ignore_spaces: bool = True,
) -> List[int]:
    positions: List[int] = []

    for norm_idx, orig_idx in enumerate(norm_to_orig):
        if original_start <= orig_idx < original_end:
            if ignore_spaces and normalized_text[norm_idx] == " ":
                continue
            positions.append(norm_idx)

    return positions


def project_spans_to_normalized_text(
    rule_spans: List[Dict[str, Any]],
    norm_to_orig: List[int],
    normalized_text: str,
) -> List[Dict[str, Any]]:
    projected: List[Dict[str, Any]] = []

    for span in rule_spans:
        rule = normalize_rule_name(span.get("rule", ""))
        if not is_duration_rule(rule):
            continue

        start = int(span.get("start", 0))
        end = int(span.get("end", 0))

        norm_positions = _norm_positions_for_original_span(
            start,
            end,
            norm_to_orig,
            normalized_text,
            ignore_spaces=True,
        )

        if not norm_positions:
            continue

        norm_start = min(norm_positions)
        norm_end = max(norm_positions) + 1
        snippet = normalized_text[norm_start:norm_end]

        projected.append(
            {
                "rule": rule,
                "coarse_group": coarse_duration_group(rule),
                "original_start": start,
                "original_end": end,
                "norm_start": norm_start,
                "norm_end": norm_end,
                "text": snippet,
                "contains_space": " " in snippet,
            }
        )

    return projected


@dataclass
class AlignmentPrepRecord:
    id: str
    audio_path: str
    hf_index: int
    surah_name: Optional[str]
    quranjson_verse_key: Optional[str]
    quranjson_verse_index: Optional[int]

    original_text: str
    normalized_text: str
    normalized_char_to_original_index: List[int]

    duration_rule_spans_original: List[Dict[str, Any]]
    duration_rule_spans_normalized: List[Dict[str, Any]]

    gold_duration_labels: List[str]
    projected_duration_labels: List[str]
    projection_exact_label_match: bool

    normalized_char_labels: List[Dict[str, Any]]


def prepare_duration_alignment_record(row: Dict[str, Any]) -> AlignmentPrepRecord:
    original_text = row.get("aya_text", "")
    rule_spans = list(row.get("rule_spans", []))

    normalized_text, norm_to_orig = normalize_text_for_alignment(original_text)
    orig_char_labels = build_original_char_rule_labels(original_text, rule_spans)

    normalized_char_labels: List[Dict[str, Any]] = []
    for norm_idx, orig_idx in enumerate(norm_to_orig):
        ch = normalized_text[norm_idx]

        if ch == " ":
            labels: List[str] = []
        else:
            labels = orig_char_labels[orig_idx] if 0 <= orig_idx < len(orig_char_labels) else []

        coarse = sorted(set(filter(None, (coarse_duration_group(x) for x in labels))))
        normalized_char_labels.append(
            {
                "norm_index": norm_idx,
                "char": ch,
                "original_index": orig_idx,
                "original_char": original_text[orig_idx] if 0 <= orig_idx < len(original_text) else "",
                "rules": labels,
                "coarse_rules": coarse,
            }
        )

    projected_spans = project_spans_to_normalized_text(rule_spans, norm_to_orig, normalized_text)

    gold_duration_labels = sorted(
        set(
            normalize_rule_name(span.get("rule", ""))
            for span in rule_spans
            if is_duration_rule(span.get("rule", ""))
        )
    )

    projected_duration_labels = sorted(set(span["rule"] for span in projected_spans))
    projection_exact_label_match = set(projected_duration_labels) == set(gold_duration_labels)

    audio_path = row.get("audio_path")
    if not audio_path:
        raise RuntimeError(f"Row {row.get('id')} has no audio_path")

    return AlignmentPrepRecord(
        id=str(row["id"]),
        audio_path=str(audio_path),
        hf_index=int(row["hf_index"]),
        surah_name=row.get("surah_name"),
        quranjson_verse_key=row.get("quranjson_verse_key"),
        quranjson_verse_index=row.get("quranjson_verse_index"),
        original_text=original_text,
        normalized_text=normalized_text,
        normalized_char_to_original_index=norm_to_orig,
        duration_rule_spans_original=rule_spans,
        duration_rule_spans_normalized=projected_spans,
        gold_duration_labels=gold_duration_labels,
        projected_duration_labels=projected_duration_labels,
        projection_exact_label_match=projection_exact_label_match,
        normalized_char_labels=normalized_char_labels,
    )


def prepare_alignment_records(rows: List[Dict[str, Any]]) -> List[AlignmentPrepRecord]:
    out: List[AlignmentPrepRecord] = []
    for row in rows:
        try:
            out.append(prepare_duration_alignment_record(row))
        except Exception as e:
            print(f"Skipping {row.get('id', '<unknown>')}: {e}")
    return out