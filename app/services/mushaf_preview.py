from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FULL_QURAN_TAJWEED_JSONL = (
    PROJECT_ROOT / "data" / "manifests" / "quran_tajweed_reference_full.jsonl"
)

DEFAULT_COLOR = "#111827"

COARSE_RULE_COLORS = {
    "madd": "#dc2626",
    "ghunnah": "#16a34a",
    "ikhfa": "#d97706",
    "idgham": "#7c3aed",
    "qalqalah": "#2563eb",
}

RAW_RULE_COLORS = {
    "madd_2": "#dc2626",
    "madd_4": "#b91c1c",
    "madd_6": "#991b1b",
    "ghunnah": "#16a34a",
    "ikhfa": "#d97706",
    "idgham": "#7c3aed",
    "qalqalah": "#2563eb",
    "iqlab": "#0891b2",
    "lam_shamsiyyah": "#0284c7",
    "lam_qamariyyah": "#38bdf8",
    "hamzat_wasl": "#6b7280",
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


@lru_cache(maxsize=1)
def _load_rows() -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}

    if not FULL_QURAN_TAJWEED_JSONL.exists():
        return rows

    with FULL_QURAN_TAJWEED_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            surah = _safe_int(row.get("surah"))
            ayah = _safe_int(row.get("ayah"))

            if surah > 0 and ayah > 0:
                rows[(surah, ayah)] = row

    return rows


def _choose_label_rule(label: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Return (rule_name, color) for one normalized character label.
    Prefer modeled coarse rules, but preserve source colors when available.
    """
    details = label.get("rule_details") or []

    if not isinstance(details, list):
        details = []

    # Priority: rules currently scored by trained modules.
    priority = ["madd", "ghunnah", "ikhfa", "idgham", "qalqalah"]

    for wanted in priority:
        for detail in details:
            coarse = detail.get("coarse_rule")
            raw_rule = detail.get("rule")
            if coarse == wanted:
                color = (
                    COARSE_RULE_COLORS.get(str(coarse))
                    or RAW_RULE_COLORS.get(str(raw_rule))
                    or detail.get("color")
                    or DEFAULT_COLOR
                )
                return str(coarse), str(color)

    # Fallback: show any rule metadata color.
    for detail in details:
        raw_rule = detail.get("rule")
        coarse = detail.get("coarse_rule")
        color = (
            detail.get("color")
            or COARSE_RULE_COLORS.get(str(coarse))
            or RAW_RULE_COLORS.get(str(raw_rule))
        )

        if raw_rule and color:
            return str(raw_rule), str(color)

    return None, DEFAULT_COLOR


def _merge_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []

    merged: list[dict[str, Any]] = []

    for seg in segments:
        if (
            merged
            and merged[-1].get("rule") == seg.get("rule")
            and merged[-1].get("color") == seg.get("color")
        ):
            merged[-1]["text"] += seg.get("text", "")
        else:
            merged.append(dict(seg))

    return merged


def _build_segments_from_reference_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """
    The generated full reference stores compact character labels:
      normalized_char_labels[compact_index]

    But for display we want spaces from:
      normalized_text

    So we walk through normalized_text and map only non-space chars to labels.
    """
    display_text = (
        row.get("normalized_text")
        or row.get("original_text")
        or row.get("normalized_text_compact")
        or ""
    )

    labels = row.get("normalized_char_labels") or []
    if not isinstance(labels, list):
        labels = []

    segments: list[dict[str, Any]] = []
    compact_i = 0

    for ch in str(display_text):
        if ch.isspace():
            segments.append(
                {
                    "text": ch,
                    "rule": None,
                    "color": DEFAULT_COLOR,
                }
            )
            continue

        label = labels[compact_i] if compact_i < len(labels) else {}
        rule, color = _choose_label_rule(label)

        segments.append(
            {
                "text": ch,
                "rule": rule,
                "color": color or DEFAULT_COLOR,
            }
        )

        compact_i += 1

    return _merge_segments(segments)


def get_mushaf_preview(surah: int, ayah: int) -> dict[str, Any]:
    surah = int(surah)
    ayah = int(ayah)

    rows = _load_rows()
    row = rows.get((surah, ayah))

    if row is None:
        return {
            "available": False,
            "surah": surah,
            "ayah": ayah,
            "text": "",
            "segments": [],
            "reason": f"No full Qur'an Tajweed reference row found for {surah}:{ayah}.",
        }

    segments = _build_segments_from_reference_row(row)

    return {
        "available": True,
        "surah": surah,
        "ayah": ayah,
        "text": row.get("normalized_text") or row.get("original_text") or "",
        "segments": segments,
        "source_id": row.get("id") or row.get("sample_id"),
    }
