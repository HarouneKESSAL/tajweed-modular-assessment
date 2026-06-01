from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_CANDIDATES = [
    PROJECT_ROOT / "configs" / "rule_manifest_json.json",
    PROJECT_ROOT / "data" / "manifests" / "rule_manifest_json.json",
    PROJECT_ROOT / "data" / "reference" / "rule_manifest_json.json",
]

MODELED_RULE_IDS = [
    "madd",
    "ghunnah",
    "ikhfa",
    "idgham_bighunnah",
    "idgham_bilaghunnah",
    "qalqalah",
]

COMMON_ALIASES = {
    "mad": "madd",
    "al_mad": "madd",
    "madd_2": "madd",
    "madd_4": "madd",
    "madd_6": "madd",
    "madd_246": "madd",
    "madd_munfasil": "madd",
    "madd_muttasil": "madd",
    "madd_lazim": "madd",
    "ghunna": "ghunnah",
    "gunnah": "ghunnah",
    "ikhfaa": "ikhfa",
    "ikhfa_haqiqi": "ikhfa",
    "ikhfa_shafawi": "ikhfa",
    "idgham": "idgham_bighunnah",
    "idgom": "idgham_bighunnah",
    "idghaam": "idgham_bighunnah",
    "idghaam_ghunnah": "idgham_bighunnah",
    "idghaam_shafawi": "idgham_bighunnah",
    "idghaam_no_ghunnah": "idgham_bilaghunnah",
    "qalqala": "qalqalah",
    "qalqalah_burst": "qalqalah",
}


@lru_cache(maxsize=1)
def load_rule_manifest() -> dict[str, Any]:
    for path in MANIFEST_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    return {}


def _text(value: Any, lang: str = "en") -> str:
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or value.get("ar") or "")
    return str(value or "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compact_index_to_display_index(text: str, compact_position: int) -> int:
    if compact_position < 0:
        return 0

    compact_i = 0
    for display_i, ch in enumerate(text):
        if ch.isspace():
            continue

        if compact_i == compact_position:
            return display_i

        compact_i += 1

    return max(0, len(text) - 1)


def extract_snippet(text: str, position: int, radius: int = 8) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    display_i = _compact_index_to_display_index(text, position)

    start = max(0, display_i - radius)
    end = min(len(text), display_i + radius + 1)

    while start > 0 and not text[start - 1].isspace():
        start -= 1

    while end < len(text) and not text[end].isspace():
        end += 1

    return text[start:end].strip()

def _display_index_to_compact_index(text: str, display_index: int) -> int:
    text = str(text or "")

    if display_index < 0:
        return -1

    compact_i = 0

    for i, ch in enumerate(text):
        if i >= display_index:
            return compact_i

        if not ch.isspace():
            compact_i += 1

    return compact_i


def _find_word_bounds_from_display_index(text: str, display_index: int) -> tuple[int, int]:
    text = str(text or "")

    if not text:
        return 0, 0

    display_index = max(0, min(display_index, len(text) - 1))

    if text[display_index].isspace():
        left = display_index - 1
        right = display_index + 1

        while left >= 0 or right < len(text):
            if left >= 0 and not text[left].isspace():
                display_index = left
                break

            if right < len(text) and not text[right].isspace():
                display_index = right
                break

            left -= 1
            right += 1

    start = display_index
    end = display_index + 1

    while start > 0 and not text[start - 1].isspace():
        start -= 1

    while end < len(text) and not text[end].isspace():
        end += 1

    return start, end


def _find_explicit_word_bounds(text: str, word: str) -> tuple[int, int]:
    text = str(text or "")
    word = str(word or "").strip()

    if not text or not word:
        return -1, -1

    idx = text.find(word)
    if idx >= 0:
        return idx, idx + len(word)

    return -1, -1


def extract_target_location(
    text: str,
    position: int,
    err: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert a technical compact character position into the learner-friendly
    word that contains the Tajweed issue.
    """
    text = str(text or "").strip()
    err = err or {}

    explicit_word = ""

    for key in [
        "target_word",
        "word",
        "expected_word",
        "reference_word",
        "matched_word",
        "location_word",
    ]:
        value = err.get(key)
        if value:
            explicit_word = str(value).strip()
            break

    display_i = _compact_index_to_display_index(text, position)
    target_letter = ""

    if 0 <= display_i < len(text) and not text[display_i].isspace():
        target_letter = text[display_i]

    if explicit_word:
        start, end = _find_explicit_word_bounds(text, explicit_word)

        return {
            "target_word": explicit_word,
            "word": explicit_word,
            "target_letter": target_letter,
            "word_start": start,
            "word_end": end,
            "compact_word_start": _display_index_to_compact_index(text, start) if start >= 0 else -1,
            "compact_word_end": _display_index_to_compact_index(text, end) if end >= 0 else -1,
            "from_error_payload": True,
        }

    if not text:
        return {
            "target_word": "",
            "word": "",
            "target_letter": "",
            "word_start": -1,
            "word_end": -1,
            "compact_word_start": -1,
            "compact_word_end": -1,
            "from_error_payload": False,
        }

    start, end = _find_word_bounds_from_display_index(text, display_i)
    word = text[start:end].strip()

    return {
        "target_word": word,
        "word": word,
        "target_letter": target_letter,
        "word_start": start,
        "word_end": end,
        "compact_word_start": _display_index_to_compact_index(text, start),
        "compact_word_end": _display_index_to_compact_index(text, end),
        "from_error_payload": False,
    }
    
    
def _next_non_space_char(text: str, compact_position: int) -> str:
    display_i = _compact_index_to_display_index(text, compact_position)

    for ch in text[display_i + 1:]:
        if not ch.isspace():
            return ch

    return ""


def normalize_rule_id(
    rule: str | None,
    *,
    reference_text: str = "",
    position: int = -1,
    seen: set[str] | None = None,
) -> str:
    manifest = load_rule_manifest()
    rules = manifest.get("rules", {})
    aliases = manifest.get("rule_aliases", {})

    raw = str(rule or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return ""

    if raw in rules:
        return raw

    if raw in COMMON_ALIASES:
        resolved = COMMON_ALIASES[raw]
        if resolved in rules:
            return resolved

    if seen is None:
        seen = set()

    if raw in seen:
        return raw

    seen.add(raw)

    alias = aliases.get(raw)
    if isinstance(alias, dict):
        if alias.get("resolve_to"):
            return normalize_rule_id(
                alias.get("resolve_to"),
                reference_text=reference_text,
                position=position,
                seen=seen,
            )

        if alias.get("resolution_type") == "contextual_alias":
            next_ch = _next_non_space_char(reference_text, position)
            for condition in alias.get("conditions", []):
                letters = condition.get("if_next_letter_in", [])
                if next_ch in letters:
                    return normalize_rule_id(
                        condition.get("resolve_to"),
                        reference_text=reference_text,
                        position=position,
                        seen=seen,
                    )

            return normalize_rule_id(
                alias.get("fallback_rule_id"),
                reference_text=reference_text,
                position=position,
                seen=seen,
            )

    return COMMON_ALIASES.get(raw, raw)


def _get_weighted_score(tajweed_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not tajweed_payload:
        return {}

    if isinstance(tajweed_payload.get("weighted_score"), dict):
        return tajweed_payload["weighted_score"]

    result = tajweed_payload.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("weighted_score"), dict):
            return result["weighted_score"]

        diagnosis = result.get("diagnosis")
        if isinstance(diagnosis, dict) and isinstance(diagnosis.get("weighted_score"), dict):
            return diagnosis["weighted_score"]

    return {}


def _rule_entry(rule_id: str) -> dict[str, Any]:
    manifest = load_rule_manifest()
    rules = manifest.get("rules", {})
    entry = rules.get(rule_id)
    return entry if isinstance(entry, dict) else {}


def _source_module(rule_id: str, err: dict[str, Any]) -> str:
    entry = _rule_entry(rule_id)
    return str(err.get("source_module") or entry.get("source_module") or "")


def infer_error_type(
    rule_id: str,
    err: dict[str, Any],
    *,
    reference_text: str,
) -> str:
    manifest = load_rule_manifest()
    module = _source_module(rule_id, err)

    detail = " ".join(
        str(err.get(k) or "")
        for k in ["detail", "message", "reason", "evidence"]
    ).lower()

    expected = rule_id
    predicted = normalize_rule_id(
        str(err.get("predicted") or err.get("predicted_rule") or err.get("pred") or ""),
        reference_text=reference_text,
        position=_safe_int(err.get("position")),
    )

    for rule in manifest.get("error_type_inference", {}).get(module, []):
        if not isinstance(rule, dict):
            continue

        if "if_expected_rule" in rule:
            exp = normalize_rule_id(
                rule.get("if_expected_rule"),
                reference_text=reference_text,
                position=_safe_int(err.get("position")),
            )
            if exp != expected:
                continue

        if "if_predicted_rule" in rule:
            pred = normalize_rule_id(
                rule.get("if_predicted_rule"),
                reference_text=reference_text,
                position=_safe_int(err.get("position")),
            )
            if pred != predicted:
                continue

        terms = rule.get("if_detail_contains_any")
        if terms:
            if not any(str(t).lower() in detail for t in terms):
                continue

        if rule.get("prefer_error_type"):
            return str(rule["prefer_error_type"])

    # Practical fallbacks for current module outputs.
    if rule_id == "madd":
        return "too_short" if "short" in detail or "insufficient" in detail else ""

    if rule_id == "ghunnah":
        if predicted == "madd":
            return "confused_with_madd"
        if predicted in {"", "none"}:
            return "missing_nasalization"
        return "too_short"

    if rule_id == "ikhfa":
        if predicted.startswith("idgham"):
            return "too_merged"
        if "clear" in detail or "izhar" in detail:
            return "too_clear"
        if "ghunnah" in detail or predicted in {"", "none"}:
            return "missing_ghunnah"
        return ""

    if rule_id.startswith("idgham"):
        if predicted in {"", "none"}:
            return "not_merged"
        if "without ghunnah" in detail or "missing ghunnah" in detail:
            return "missing_ghunnah"
        if "clear" in detail or "izhar" in detail:
            return "too_clear"
        return ""

    if rule_id == "qalqalah":
        if predicted in {"", "none"} or "did not fire" in detail or "no burst" in detail:
            return "missing_release"
        if "excessive" in detail or "too strong" in detail:
            return "excessive_release"
        if "vowel" in detail:
            return "added_vowel"
        return "missing_release"

    return ""


def _severity_defaults(rule_id: str) -> dict[str, Any]:
    manifest = load_rule_manifest()
    defaults = manifest.get("severity_policy", {}).get("rule_severity_defaults", {})

    if rule_id in defaults:
        return defaults[rule_id]

    if rule_id.startswith("idgham"):
        return defaults.get("idgham_bighunnah") or defaults.get("ikhfa") or {}

    return {}


def _confidence_adjustment(confidence: float) -> int:
    if confidence >= 0.9:
        return 5
    if confidence >= 0.75:
        return 3
    if confidence <= 0.45:
        return -8
    if confidence <= 0.6:
        return -4
    return 0


def severity_score(rule_id: str, err: dict[str, Any]) -> int:
    defaults = _severity_defaults(rule_id)
    base = int(defaults.get("base_score") or 50)
    confidence = _safe_float(err.get("confidence"), 0.0)
    score = base + _confidence_adjustment(confidence)
    return max(0, min(100, score))


def _pick_error_message(rule_id: str, error_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    entry = _rule_entry(rule_id)

    error_entry = {}
    if error_type:
        error_entry = entry.get("error_types", {}).get(error_type, {})

    if not error_entry:
        error_entry = entry.get("generic_error", {})

    return entry, error_entry


def build_supported_rules_payload() -> list[dict[str, Any]]:
    manifest = load_rule_manifest()
    rules = manifest.get("rules", {})

    if not rules:
        return [
            {"name": "madd", "module": "duration", "status": "modeled"},
            {"name": "ghunnah", "module": "duration", "status": "modeled"},
            {"name": "ikhfa", "module": "transition", "status": "modeled"},
            {"name": "idgham", "module": "transition", "status": "modeled"},
            {"name": "qalqalah", "module": "burst", "status": "modeled"},
        ]

    items = []
    for rule_id in MODELED_RULE_IDS:
        entry = rules.get(rule_id)
        if not isinstance(entry, dict):
            continue

        items.append(
            {
                "name": rule_id,
                "display_name": entry.get("display_name", {}),
                "module": entry.get("source_module"),
                "status": "modeled",
                "arabic_name": entry.get("arabic_name"),
                "english_name": entry.get("english_name"),
            }
        )

    return items


def build_readable_feedback(
    reference_text: str,
    tajweed_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    manifest = load_rule_manifest()
    weighted = _get_weighted_score(tajweed_payload)
    errors = weighted.get("errors") or []
    if not isinstance(errors, list):
        return []

    if len(errors) == 0:
        score = weighted.get("score")
        try:
            score_value = float(score)
        except Exception:
            score_value = None

        if score_value is None or score_value >= 95:
            return [
                {
                    "feedback_type": "positive",
                    "rule": "overall",
                    "rule_id": "overall",
                    "severity": "positive",
                    "severity_level": "positive",
                    "severity_score": 0,
                    "position": -1,
                    "location": "",
                    "message": "Excellent recitation. The ayah content was correct, and no Tajweed errors were detected.",
                    "message_ar": "تلاوة ممتازة. كان نص الآية صحيحًا، ولم يتم اكتشاف أخطاء تجويدية.",
                    "default_error_message": {"en": "", "ar": ""},
                    "corrective_message": {
                        "en": "Keep practising with the same clarity and steadiness.",
                        "ar": "استمر في التدريب بنفس الوضوح والثبات.",
                    },
                }
            ]

        return [
            {
                "feedback_type": "positive",
                "rule": "overall",
                "rule_id": "overall",
                "severity": "positive",
                "severity_level": "positive",
                "severity_score": 0,
                "position": -1,
                "location": "",
                "message": "Good recitation. No specific Tajweed errors were detected.",
                "message_ar": "تلاوة جيدة. لم يتم اكتشاف أخطاء تجويدية محددة.",
                "default_error_message": {"en": "", "ar": ""},
                "corrective_message": {
                    "en": "Continue practising to improve fluency and confidence.",
                    "ar": "واصل التدريب لتحسين السلاسة والثقة في الأداء.",
                },
            }
        ]


    max_items = (
        manifest.get("severity_policy", {})
        .get("presentation_policy", {})
        .get("max_feedback_items_default", 3)
    )

    items: list[dict[str, Any]] = []

    for err in errors:
        if not isinstance(err, dict):
            continue

        position = _safe_int(err.get("position"))
        raw_rule = err.get("rule") or err.get("expected") or err.get("expected_rule")
        rule_id = normalize_rule_id(raw_rule, reference_text=reference_text, position=position)

        entry = _rule_entry(rule_id)
        if not entry:
            continue

        error_type = infer_error_type(rule_id, err, reference_text=reference_text)
        entry, error_entry = _pick_error_message(rule_id, error_type)

        display_name = entry.get("display_name", {})
        rule_name_en = _text(display_name, "en") or entry.get("english_name") or rule_id
        rule_name_ar = _text(display_name, "ar") or entry.get("arabic_name") or rule_id

        default_en = _text(error_entry.get("default_error_message"), "en")
        corrective_en = _text(error_entry.get("corrective_message"), "en")
        default_ar = _text(error_entry.get("default_error_message"), "ar")
        corrective_ar = _text(error_entry.get("corrective_message"), "ar")

        snippet = extract_snippet(reference_text, position)
        target_location = extract_target_location(reference_text, position, err)
        target_word = str(target_location.get("target_word") or "").strip()
        target_letter = str(target_location.get("target_letter") or "").strip()

        confidence = _safe_float(err.get("confidence"), 0.0)
        score = severity_score(rule_id, err)
        sev_defaults = _severity_defaults(rule_id)

        if target_word:
            learner_title_en = f"{rule_name_en} in “{target_word}”"
            learner_title_ar = f"{rule_name_ar} في كلمة «{target_word}»"
            location_en = f"in the word “{target_word}”"
            location_ar = f"في كلمة «{target_word}»"
        elif snippet:
            learner_title_en = f"{rule_name_en} needs attention"
            learner_title_ar = f"{rule_name_ar} يحتاج إلى مراجعة"
            location_en = f"near “{snippet}”"
            location_ar = f"قرب «{snippet}»"
        else:
            learner_title_en = f"{rule_name_en} needs attention"
            learner_title_ar = f"{rule_name_ar} يحتاج إلى مراجعة"
            location_en = "in this ayah"
            location_ar = "في هذه الآية"

        message_en = f"{learner_title_en}. {default_en} {corrective_en}".strip()
        message_ar = f"{learner_title_ar}. {default_ar} {corrective_ar}".strip()

        items.append(
            {
                "feedback_type": "rule",
                "rule": rule_id,
                "rule_id": rule_id,
                "display_name": display_name,
                "rule_name_en": rule_name_en,
                "rule_name_ar": rule_name_ar,
                # Raw technical position is kept for debugging, but the frontend should
                # show target_word / learner_title instead of "position 41".
                "position": position,
                "location": target_word or snippet,
                "snippet": snippet,

                "target_word": target_word,
                "target_letter": target_letter,
                "word_start": target_location.get("word_start", -1),
                "word_end": target_location.get("word_end", -1),
                "compact_word_start": target_location.get("compact_word_start", -1),
                "compact_word_end": target_location.get("compact_word_end", -1),

                "location_en": location_en,
                "location_ar": location_ar,
                "learner_title": learner_title_en,
                "learner_title_ar": learner_title_ar,
                "learner_message": message_en,
                "learner_message_ar": message_ar,

                "confidence": confidence,
                "source_module": _source_module(rule_id, err),
                "predicted_rule": err.get("predicted") or err.get("predicted_rule") or err.get("pred"),
                "error_type": error_type or "generic",
                "classical_severity": sev_defaults.get("classical_severity", "lahn_khafi"),
                "severity": sev_defaults.get("severity_level", "medium"),
                "severity_level": sev_defaults.get("severity_level", "medium"),
                "severity_score": score,
                "feedback_priority": entry.get("feedback_priority"),
                "default_error_message": {
                    "en": default_en,
                    "ar": default_ar,
                },
                "corrective_message": {
                    "en": corrective_en,
                    "ar": corrective_ar,
                },
                "message": message_en,
                "message_ar": message_ar,
            }
        )

    items.sort(
        key=lambda x: (
            -int(x.get("severity_score") or 0),
            -float(x.get("confidence") or 0),
            int(x.get("position") or 0),
        )
    )

    return items[: int(max_items or 3)]


def build_tajweed_ui_payload(
    reference_text: str,
    tajweed_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest = load_rule_manifest()

    return {
        "manifest_version": manifest.get("manifest_version"),
        "supported_rules": build_supported_rules_payload(),
        "readable_feedback": build_readable_feedback(reference_text, tajweed_payload),
        "feedback_policy": {
            "content_priority": manifest.get("integration", {}).get("content_priority", True),
            "severity_policy_enabled": manifest.get("integration", {}).get("severity_policy_enabled", True),
        },
    }
