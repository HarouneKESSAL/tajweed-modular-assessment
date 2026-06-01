from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")
SOURCE_SURAH_DIR = ROOT / "external" / "quranjson-tajwid" / "source" / "surah"
COLORED_DIR = ROOT / "external" / "quranjson-tajwid" / "tajweed_colored"

OUT_JSONL = ROOT / "data" / "manifests" / "quran_tajweed_reference_full.jsonl"
OUT_SUMMARY_JSON = ROOT / "data" / "analysis" / "quran_tajweed_reference_full_summary.json"
OUT_MD = ROOT / "data" / "analysis" / "quran_tajweed_reference_full_summary.md"


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

QALQALAH_LETTERS = set("قطبجد")
NOON_MEEM_LETTERS = set("نم")


def normalize_char(ch: str) -> str:
    if ARABIC_DIACRITICS_RE.match(ch):
        return ""

    ch = ch.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    ch = ch.replace("ٱ", "ا")
    ch = ch.replace("ى", "ي")
    ch = ch.replace("ة", "ه")

    if ch.isspace():
        return ""

    return ch


def normalize_text_with_mapping(text: str) -> tuple[str, list[dict[str, Any]], dict[int, int]]:
    """
    Returns:
      normalized_text_without_spaces,
      normalized_char_labels,
      raw_index_to_norm_index

    raw positions are Python string indices.
    """
    chars: list[str] = []
    labels: list[dict[str, Any]] = []
    raw_to_norm: dict[int, int] = {}

    for raw_i, ch in enumerate(text):
        n = normalize_char(ch)
        if not n:
            continue

        norm_i = len(chars)
        chars.append(n)
        raw_to_norm[raw_i] = norm_i
        labels.append({
            "norm_index": norm_i,
            "char": n,
            "raw_indices": [raw_i],
            "rules": [],
            "coarse_rules": [],
            "rule_details": [],
        })

    return "".join(chars), labels, raw_to_norm


def spaced_normalized(text: str) -> str:
    # This keeps spaces for display/reference, but removes diacritics.
    out = []
    previous_space = False

    for ch in text:
        if ARABIC_DIACRITICS_RE.match(ch):
            continue

        ch = ch.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        ch = ch.replace("ٱ", "ا")
        ch = ch.replace("ى", "ي")
        ch = ch.replace("ة", "ه")

        if ch.isspace():
            if not previous_space:
                out.append(" ")
            previous_space = True
        else:
            out.append(ch)
            previous_space = False

    return "".join(out).strip()


def rule_to_coarse(rule: str) -> str | None:
    r = str(rule or "").lower()

    if r.startswith("madd") or "madd" in r:
        return "madd"

    if "ghunn" in r:
        return "ghunnah"

    if "ikhfa" in r:
        return "ikhfa"

    if "idgham" in r:
        return "idgham"

    if "qalqalah" in r or "qalqala" in r:
        return "qalqalah"

    # Iqlab exists in the color source, but our current transition module was
    # trained/evaluated for ikhfa/idgham/none, not iqlab. Keep it as metadata.
    return None


def module_for_coarse(coarse: str | None) -> str | None:
    if coarse in {"madd", "ghunnah"}:
        return "duration"
    if coarse in {"ikhfa", "idgham"}:
        return "transition"
    if coarse == "qalqalah":
        return "burst"
    return None


def get_source_verses(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))

    # Expected quranjson structure:
    # {
    #   "index": "112",
    #   "verse": {"verse_1": "...", ...}
    # }
    if isinstance(data, dict):
        verse_obj = data.get("verse")
        if isinstance(verse_obj, dict):
            return {str(k): str(v) for k, v in verse_obj.items()}

        # Some variants may store ayahs as a list.
        for key in ["verses", "ayahs", "ayat"]:
            if isinstance(data.get(key), list):
                out = {}
                for item in data[key]:
                    if not isinstance(item, dict):
                        continue
                    ayah_no = item.get("aya_no") or item.get("ayah") or item.get("number") or item.get("index")
                    text = item.get("text") or item.get("aya_text") or item.get("uthmani") or item.get("arabic")
                    if ayah_no is not None and text:
                        out[f"verse_{int(ayah_no)}"] = str(text)
                return out

    raise ValueError(f"Unsupported source surah structure: {path}")


def get_colored_verses(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    verse_obj = data.get("verse", {}) if isinstance(data, dict) else {}

    if not isinstance(verse_obj, dict):
        return {}

    out: dict[str, list[dict[str, Any]]] = {}
    for k, v in verse_obj.items():
        if isinstance(v, list):
            out[str(k)] = [x for x in v if isinstance(x, dict)]
    return out


def span_norm_indices(
    raw_start: int,
    raw_end: int,
    raw_to_norm: dict[int, int],
) -> list[int]:
    # Treat the JSON end offset as exclusive.
    return sorted({
        raw_to_norm[i]
        for i in range(raw_start, raw_end)
        if i in raw_to_norm
    })


def nearest_previous_norm_index(raw_start: int, raw_to_norm: dict[int, int]) -> int | None:
    candidates = [raw_i for raw_i in raw_to_norm if raw_i < raw_start]
    if not candidates:
        return None
    return raw_to_norm[max(candidates)]


def nearest_next_norm_index(raw_end: int, raw_to_norm: dict[int, int]) -> int | None:
    candidates = [raw_i for raw_i in raw_to_norm if raw_i >= raw_end]
    if not candidates:
        return None
    return raw_to_norm[min(candidates)]


def choose_rule_norm_indices(
    *,
    rule: str,
    coarse: str | None,
    raw_start: int,
    raw_end: int,
    raw_to_norm: dict[int, int],
    labels: list[dict[str, Any]],
) -> list[int]:
    """
    Convert colored-source spans to normalized target positions.

    The colored source marks visual spans, not always acoustic target letters.
    This function makes the target positions safer for our acoustic modules.
    """
    indices = span_norm_indices(raw_start, raw_end, raw_to_norm)

    # Diacritic-only spans, e.g. dagger alif, should attach to the previous
    # pronounced base letter rather than the following letter.
    if not indices:
        prev_i = nearest_previous_norm_index(raw_start, raw_to_norm)
        if prev_i is not None:
            indices = [prev_i]
        else:
            next_i = nearest_next_norm_index(raw_end, raw_to_norm)
            indices = [next_i] if next_i is not None else []

    if not indices:
        return []

    # Qalqalah can only be on قطبجد. Never allow Alif or other letters to become
    # burst/qalqalah targets.
    if coarse == "qalqalah" or "qalqalah" in str(rule).lower() or "qalqala" in str(rule).lower():
        valid = [
            i for i in indices
            if 0 <= i < len(labels) and labels[i].get("char") in QALQALAH_LETTERS
        ]

        if valid:
            return valid[:1]

        # Look very near the source span in case the source span starts at a mark.
        nearby_raw = range(max(0, raw_start - 2), raw_end + 3)
        nearby_norm = []
        for raw_i in nearby_raw:
            if raw_i in raw_to_norm:
                ni = raw_to_norm[raw_i]
                if 0 <= ni < len(labels) and labels[ni].get("char") in QALQALAH_LETTERS:
                    nearby_norm.append(ni)

        if nearby_norm:
            return sorted(set(nearby_norm))[:1]

        # Drop unsafe qalqalah spans instead of falsely scoring the wrong letter.
        return []

    # Transition rules are caused by a trigger letter/mark. For our current
    # module, use one main target position rather than coloring/scoring the whole
    # visual span.
    if coarse in {"ikhfa", "idgham"} or str(rule).lower() == "iqlab":
        for i in indices:
            if 0 <= i < len(labels) and labels[i].get("char") in NOON_MEEM_LETTERS:
                return [i]
        return [indices[0]]

    # Ghunnah should mainly target ن/م if present.
    if coarse == "ghunnah":
        for i in indices:
            if 0 <= i < len(labels) and labels[i].get("char") in NOON_MEEM_LETTERS:
                return [i]
        return [indices[0]]

    # For madd, keep the span when it maps to a base character. For diacritic-only
    # madd this was already attached to the previous base letter above.
    if coarse == "madd":
        return [indices[0]] if indices else []

    return sorted(set(indices))


def build_row(
    surah: int,
    ayah: int,
    original_text: str,
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_compact, labels, raw_to_norm = normalize_text_with_mapping(original_text)
    normalized_spaced = spaced_normalized(original_text)

    all_rule_spans = []
    duration_spans = []
    transition_spans = []
    burst_spans = []

    for rule_obj in rules:
        rule = str(rule_obj.get("rule") or "")
        coarse = rule_to_coarse(rule)
        module = module_for_coarse(coarse)

        try:
            raw_start = int(rule_obj.get("start"))
            raw_end = int(rule_obj.get("end"))
        except Exception:
            continue

        norm_indices = choose_rule_norm_indices(
            rule=rule,
            coarse=coarse,
            raw_start=raw_start,
            raw_end=raw_end,
            raw_to_norm=raw_to_norm,
            labels=labels,
        )
        if not norm_indices:
            continue

        start_norm = min(norm_indices)
        end_norm = max(norm_indices) + 1

        span = {
            "start": start_norm,
            "end": end_norm,
            "positions": norm_indices,
            "rule": rule,
            "coarse_rule": coarse,
            "module": module,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "color": rule_obj.get("color"),
            "color_name": rule_obj.get("color_name"),
            "rule_description": rule_obj.get("rule_description"),
            "text": normalized_compact[start_norm:end_norm],
        }

        all_rule_spans.append(span)

        for ni in norm_indices:
            label = labels[ni]
            if rule not in label["rules"]:
                label["rules"].append(rule)
            if coarse and coarse not in label["coarse_rules"]:
                label["coarse_rules"].append(coarse)
            label["rule_details"].append({
                "rule": rule,
                "coarse_rule": coarse,
                "module": module,
                "color": rule_obj.get("color"),
                "color_name": rule_obj.get("color_name"),
                "rule_description": rule_obj.get("rule_description"),
                "raw_start": raw_start,
                "raw_end": raw_end,
            })

        if module == "duration":
            duration_spans.append(span)
        elif module == "transition":
            transition_spans.append(span)
        elif module == "burst":
            burst_spans.append(span)

    # Keep all rules as metadata, but supported model spans separately.
    return {
        "id": f"quran_tajweed_{surah:03d}_{ayah:03d}",
        "sample_id": f"quran_tajweed_{surah:03d}_{ayah:03d}",
        "audio_path": "__USER_AUDIO_OVERRIDE__",
        "split": "reference",
        "source": "quranjson-tajwid/tajweed_colored",
        "surah": surah,
        "ayah": ayah,
        "surah_index": f"{surah:03d}",
        "verse_key": f"verse_{ayah}",
        "quranjson_verse_key": f"verse_{ayah}",
        "quranjson_verse_index": ayah,
        "original_text": original_text,
        "normalized_text": normalized_spaced,
        "normalized_text_compact": normalized_compact,
        "normalized_char_labels": labels,
        "tajweed_rule_spans_normalized": all_rule_spans,
        "rule_spans_normalized": all_rule_spans,
        "duration_rule_spans_normalized": duration_spans,
        "transition_rule_spans_normalized": transition_spans,
        "burst_rule_spans_normalized": burst_spans,
        "supported_rule_counts": {
            "duration": len(duration_spans),
            "transition": len(transition_spans),
            "burst": len(burst_spans),
        },
    }


def main() -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = {
        "surahs_seen": 0,
        "rows": 0,
        "missing_source": [],
        "missing_colored": [],
        "rule_counts": Counter(),
        "coarse_counts": Counter(),
        "module_counts": Counter(),
        "rows_with_supported_rules": 0,
        "rows_by_module": Counter(),
    }

    for surah in range(1, 115):
        source_path = SOURCE_SURAH_DIR / f"surah_{surah}.json"
        colored_path = COLORED_DIR / f"surah_{surah}.json"

        if not source_path.exists():
            summary["missing_source"].append(str(source_path))
            continue

        if not colored_path.exists():
            summary["missing_colored"].append(str(colored_path))
            colored_verses = {}
        else:
            colored_verses = get_colored_verses(colored_path)

        source_verses = get_source_verses(source_path)
        summary["surahs_seen"] += 1

        for verse_key, original_text in sorted(
            source_verses.items(),
            key=lambda kv: int(str(kv[0]).split("_")[-1]),
        ):
            try:
                ayah = int(str(verse_key).split("_")[-1])
            except Exception:
                continue

            # Use verse_1..verse_n. Ignore verse_0 because in colored files it
            # is usually the basmalah prefix for non-Fatiha surahs.
            rules = colored_verses.get(f"verse_{ayah}", [])

            row = build_row(
                surah=surah,
                ayah=ayah,
                original_text=original_text,
                rules=rules,
            )
            rows.append(row)

            if (
                row["duration_rule_spans_normalized"]
                or row["transition_rule_spans_normalized"]
                or row["burst_rule_spans_normalized"]
            ):
                summary["rows_with_supported_rules"] += 1

            if row["duration_rule_spans_normalized"]:
                summary["rows_by_module"]["duration"] += 1
            if row["transition_rule_spans_normalized"]:
                summary["rows_by_module"]["transition"] += 1
            if row["burst_rule_spans_normalized"]:
                summary["rows_by_module"]["burst"] += 1

            for span in row["tajweed_rule_spans_normalized"]:
                summary["rule_counts"][span["rule"]] += 1
                if span["coarse_rule"]:
                    summary["coarse_counts"][span["coarse_rule"]] += 1
                if span["module"]:
                    summary["module_counts"][span["module"]] += 1

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["rows"] = len(rows)
    summary_jsonable = {
        k: dict(v) if isinstance(v, Counter) else v
        for k, v in summary.items()
    }

    OUT_SUMMARY_JSON.write_text(
        json.dumps(summary_jsonable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Full Qur'an Tajweed reference manifest")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| rows | {summary['rows']} |")
    lines.append(f"| surahs_seen | {summary['surahs_seen']} |")
    lines.append(f"| rows_with_supported_rules | {summary['rows_with_supported_rules']} |")
    lines.append("")
    lines.append("## Module span counts")
    lines.append("")
    lines.append("| module | spans | rows_with_module |")
    lines.append("|---|---:|---:|")
    for module in ["duration", "transition", "burst"]:
        lines.append(
            f"| {module} | {summary['module_counts'].get(module, 0)} | "
            f"{summary['rows_by_module'].get(module, 0)} |"
        )

    lines.append("")
    lines.append("## Coarse rule counts")
    lines.append("")
    lines.append("| rule | count |")
    lines.append("|---|---:|")
    for rule, count in summary["coarse_counts"].most_common():
        lines.append(f"| {rule} | {count} |")

    lines.append("")
    lines.append("## Raw rule counts, top 40")
    lines.append("")
    lines.append("| rule | count |")
    lines.append("|---|---:|")
    for rule, count in summary["rule_counts"].most_common(40):
        lines.append(f"| {rule} | {count} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_JSONL)
    print(OUT_SUMMARY_JSON)
    print(OUT_MD)
    print(OUT_MD.read_text(encoding="utf-8")[:5000])


if __name__ == "__main__":
    main()
