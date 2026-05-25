from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

try:
    from app.services.rule_feedback_manifest import load_rule_manifest
except Exception:
    def load_rule_manifest() -> dict[str, Any]:
        return {}


ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_arabic(text: str) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_with_norm(text: str) -> list[dict[str, str]]:
    words = re.findall(r"[\u0600-\u06FF]+|[A-Za-z0-9]+", str(text or ""))
    out = []

    for word in words:
        norm = normalize_arabic(word)
        if norm:
            out.append({"display": word, "norm": norm})

    return out


def _text(value: Any, lang: str = "en") -> str:
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or value.get("ar") or "")
    return str(value or "")


def _content_template(kind: str) -> dict[str, Any]:
    manifest = load_rule_manifest()
    templates = (
        manifest
        .get("content_feedback", {})
        .get("templates", {})
    )

    template = templates.get(kind)
    if isinstance(template, dict):
        return template

    fallback = {
        "deletion": {
            "default_error_message": {
                "en": "The expected word or sound was not clearly recited.",
                "ar": "لم يظهر اللفظ أو الصوت المتوقع بوضوح.",
            },
            "corrective_message": {
                "en": "First recite the missing Qur'anic unit correctly, then repeat the segment.",
                "ar": "ابدأ أولًا بتصحيح اللفظ القرآني الناقص، ثم أعد قراءة المقطع.",
            },
        },
        "insertion": {
            "default_error_message": {
                "en": "An extra word or sound seems to have been inserted.",
                "ar": "يبدو أنه أُضيف لفظ أو صوت زائد.",
            },
            "corrective_message": {
                "en": "Repeat the segment slowly and avoid adding sounds that are not present in the canonical text.",
                "ar": "أعد قراءة المقطع ببطء، وتجنّب إضافة أصوات غير موجودة في النص القرآني.",
            },
        },
        "substitution": {
            "default_error_message": {
                "en": "A different word or sound seems to have been recited.",
                "ar": "يبدو أنه تم نطق لفظ أو صوت مختلف.",
            },
            "corrective_message": {
                "en": "Focus first on reciting the correct Qur'anic unit. After it is correct, repeat it with Tajweed.",
                "ar": "ركّز أولًا على نطق اللفظ القرآني الصحيح، وبعد تصحيحه أعده مع تطبيق التجويد.",
            },
        },
        "low_alignment_confidence": {
            "default_error_message": {
                "en": "The system could not align this part with enough confidence.",
                "ar": "لم يتمكن النظام من مطابقة هذا الجزء بثقة كافية.",
            },
            "corrective_message": {
                "en": "Repeat this part more clearly and at a steady pace.",
                "ar": "أعد قراءة هذا الجزء بوضوح وبسرعة معتدلة.",
            },
        },
    }

    return fallback.get(kind, fallback["low_alignment_confidence"])


def _content_severity(kind: str) -> dict[str, Any]:
    manifest = load_rule_manifest()
    defaults = (
        manifest
        .get("severity_policy", {})
        .get("content_severity_defaults", {})
    )

    value = defaults.get(kind)
    if isinstance(value, dict):
        return value

    return {
        "classical_severity": "lahn_jali",
        "severity_level": "critical",
        "base_score": 100 if kind in {"deletion", "substitution"} else 95,
    }


def _join_display(tokens: list[dict[str, str]]) -> str:
    return " ".join(t["display"] for t in tokens).strip()


def _join_norm(tokens: list[dict[str, str]]) -> str:
    return " ".join(t["norm"] for t in tokens).strip()


def _build_item(
    *,
    kind: str,
    expected_tokens: list[dict[str, str]],
    recognized_tokens: list[dict[str, str]],
    gold_word_index: int,
    pred_word_index: int,
) -> dict[str, Any]:
    template = _content_template(kind)
    sev = _content_severity(kind)

    expected = _join_display(expected_tokens)
    recognized = _join_display(recognized_tokens)

    default_en = _text(template.get("default_error_message"), "en")
    corrective_en = _text(template.get("corrective_message"), "en")
    default_ar = _text(template.get("default_error_message"), "ar")
    corrective_ar = _text(template.get("corrective_message"), "ar")

    if kind == "deletion":
        title_en = "Missing expected text"
        title_ar = "نص متوقع ناقص"
        main_en = f"Expected “{expected}”, but it was not clearly recognized."
        main_ar = f"المتوقع «{expected}»، لكنه لم يظهر بوضوح في التعرّف."
    elif kind == "insertion":
        title_en = "Extra recognized text"
        title_ar = "نص زائد"
        main_en = f"The system recognized extra text: “{recognized}”."
        main_ar = f"تعرّف النظام على نص زائد: «{recognized}»."
    elif kind == "substitution":
        title_en = "Different text recognized"
        title_ar = "نص مختلف"
        main_en = f"Expected “{expected}”, but recognized “{recognized}”."
        main_ar = f"المتوقع «{expected}»، لكن المتعرّف عليه «{recognized}»."
    else:
        title_en = "Uncertain content alignment"
        title_ar = "مطابقة غير مؤكدة"
        main_en = "The system could not align this part confidently."
        main_ar = "لم يتمكن النظام من مطابقة هذا الجزء بثقة كافية."

    return {
        "feedback_type": "content",
        "error_type": kind,
        "title": {
            "en": title_en,
            "ar": title_ar,
        },
        "position": {
            "expected_word_index": gold_word_index,
            "recognized_word_index": pred_word_index,
        },
        "expected": expected,
        "recognized": recognized,
        "expected_norm": _join_norm(expected_tokens),
        "recognized_norm": _join_norm(recognized_tokens),
        "classical_severity": sev.get("classical_severity", "lahn_jali"),
        "severity_level": sev.get("severity_level", "critical"),
        "severity_score": sev.get("base_score", 100),
        "default_error_message": {
            "en": default_en,
            "ar": default_ar,
        },
        "corrective_message": {
            "en": corrective_en,
            "ar": corrective_ar,
        },
        "message": {
            "en": f"{main_en} {default_en} {corrective_en}".strip(),
            "ar": f"{main_ar} {default_ar} {corrective_ar}".strip(),
        },
    }


def build_content_feedback(
    *,
    gate: dict[str, Any] | None,
    reference: dict[str, Any] | None = None,
    mode: str = "guided",
    max_items: int = 5,
) -> dict[str, Any] | None:
    """
    Builds bilingual learner-facing content feedback.

    Returns None when content is accepted.
    """
    if not gate:
        return None

    if gate.get("accepted"):
        return None

    gold = str(gate.get("gold") or (reference or {}).get("text") or "")
    pred = str(gate.get("pred") or "")

    gold_tokens = tokenize_with_norm(gold)
    pred_tokens = tokenize_with_norm(pred)

    gold_norm = [t["norm"] for t in gold_tokens]
    pred_norm = [t["norm"] for t in pred_tokens]

    items: list[dict[str, Any]] = []

    matcher = SequenceMatcher(a=gold_norm, b=pred_norm, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        expected_part = gold_tokens[i1:i2]
        recognized_part = pred_tokens[j1:j2]

        if tag == "delete":
            kind = "deletion"
        elif tag == "insert":
            kind = "insertion"
        elif tag == "replace":
            kind = "substitution"
        else:
            kind = "low_alignment_confidence"

        items.append(
            _build_item(
                kind=kind,
                expected_tokens=expected_part,
                recognized_tokens=recognized_part,
                gold_word_index=i1,
                pred_word_index=j1,
            )
        )

    if not items and gold != pred:
        items.append(
            _build_item(
                kind="low_alignment_confidence",
                expected_tokens=gold_tokens,
                recognized_tokens=pred_tokens,
                gold_word_index=0,
                pred_word_index=0,
            )
        )

    items.sort(
        key=lambda x: (
            -int(x.get("severity_score") or 0),
            int(x.get("position", {}).get("expected_word_index") or 0),
        )
    )

    items = items[:max_items]

    char_accuracy = gate.get("char_accuracy")
    cer = gate.get("cer")
    edit_distance = gate.get("edit_distance")

    return {
        "available": True,
        "accepted": False,
        "mode": mode,
        "summary": {
            "en": "Content needs correction before Tajweed scoring.",
            "ar": "يجب تصحيح النص المقروء قبل تقييم أحكام التجويد.",
        },
        "policy": {
            "content_before_tajweed": True,
            "tajweed_skipped": True,
        },
        "expected": gold,
        "recognized": pred,
        "metrics": {
            "char_accuracy": char_accuracy,
            "cer": cer,
            "edit_distance": edit_distance,
        },
        "items": items,
    }
