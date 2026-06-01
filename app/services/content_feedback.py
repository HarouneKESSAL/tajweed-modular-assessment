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
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_with_norm(text: str) -> list[dict[str, Any]]:
    """
    Tokenize into display words and normalized comparison words.

    start/end are token offsets in the original display string. They are useful
    for UI highlighting/debugging, but the learner should mainly see
    word_to_correct / expected_word / recognized_word.
    """
    out: list[dict[str, Any]] = []

    for match in re.finditer(r"[\u0600-\u06FF]+|[A-Za-z0-9]+", str(text or "")):
        word = match.group(0)
        norm = normalize_arabic(word)

        if norm:
            out.append(
                {
                    "display": word,
                    "norm": norm,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    return out


def _text(value: Any, lang: str = "en") -> str:
    if isinstance(value, dict):
        return str(value.get(lang) or value.get("en") or value.get("ar") or "")
    return str(value or "")


def _content_template(kind: str) -> dict[str, Any]:
    manifest = load_rule_manifest()
    templates = manifest.get("content_feedback", {}).get("templates", {})

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
                "en": "First recite the missing Qur'anic word correctly, then repeat the segment.",
                "ar": "ابدأ أولًا بتصحيح الكلمة القرآنية الناقصة، ثم أعد قراءة المقطع.",
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
                "en": "Focus first on reciting the correct Qur'anic word. After it is correct, repeat it with Tajweed.",
                "ar": "ركّز أولًا على نطق الكلمة القرآنية الصحيحة، وبعد تصحيحها أعدها مع تطبيق التجويد.",
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
    defaults = manifest.get("severity_policy", {}).get("content_severity_defaults", {})

    value = defaults.get(kind)
    if isinstance(value, dict):
        return value

    return {
        "classical_severity": "lahn_jali",
        "severity_level": "critical",
        "base_score": 100 if kind in {"deletion", "substitution"} else 95,
    }


def _join_display(tokens: list[dict[str, Any]]) -> str:
    return " ".join(str(t.get("display") or "") for t in tokens).strip()


def _join_norm(tokens: list[dict[str, Any]]) -> str:
    return " ".join(str(t.get("norm") or "") for t in tokens).strip()


def _first_start(tokens: list[dict[str, Any]]) -> int:
    if not tokens:
        return -1
    try:
        return int(tokens[0].get("start", -1))
    except Exception:
        return -1


def _last_end(tokens: list[dict[str, Any]]) -> int:
    if not tokens:
        return -1
    try:
        return int(tokens[-1].get("end", -1))
    except Exception:
        return -1


def _item_title(kind: str) -> tuple[str, str]:
    if kind == "deletion":
        return "Missing word", "كلمة ناقصة"
    if kind == "insertion":
        return "Extra recognized word", "كلمة زائدة"
    if kind == "substitution":
        return "Different word recognized", "كلمة مختلفة"
    return "Uncertain content alignment", "مطابقة غير مؤكدة"


def _build_item(
    *,
    kind: str,
    expected_tokens: list[dict[str, Any]],
    recognized_tokens: list[dict[str, Any]],
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

    title_en, title_ar = _item_title(kind)

    # The exact learner-facing word.
    if kind == "deletion":
        word_to_correct = expected
        main_en = f'The word to correct is “{word_to_correct}”. It was expected but was not clearly recognized.'
        main_ar = f"الكلمة التي تحتاج إلى تصحيح هي «{word_to_correct}». كانت متوقعة لكنها لم تظهر بوضوح في التعرّف."
    elif kind == "insertion":
        word_to_correct = recognized
        main_en = f'The extra recognized word is “{word_to_correct}”. Check that you are reciting only the selected ayah.'
        main_ar = f"الكلمة الزائدة التي تم التعرف عليها هي «{word_to_correct}». تأكد من قراءة الآية المحددة فقط."
    elif kind == "substitution":
        word_to_correct = expected or recognized
        main_en = f'Word to correct: “{expected}”. The system recognized it as “{recognized}”.'
        main_ar = f"الكلمة التي تحتاج إلى تصحيح: «{expected}». تعرّف النظام عليها كـ «{recognized}»."
    else:
        word_to_correct = expected or recognized
        main_en = (
            f'This part needs clearer recitation: “{word_to_correct}”.'
            if word_to_correct
            else "The system could not align this part confidently."
        )
        main_ar = (
            f"هذا الجزء يحتاج إلى قراءة أوضح: «{word_to_correct}»."
            if word_to_correct
            else "لم يتمكن النظام من مطابقة هذا الجزء بثقة كافية."
        )

    message_en = f"{main_en} {default_en} {corrective_en}".strip()
    message_ar = f"{main_ar} {default_ar} {corrective_ar}".strip()

    expected_start = _first_start(expected_tokens)
    expected_end = _last_end(expected_tokens)
    recognized_start = _first_start(recognized_tokens)
    recognized_end = _last_end(recognized_tokens)

    return {
        "feedback_type": "content",
        "error_type": kind,
        "title": {
            "en": title_en,
            "ar": title_ar,
        },

        # Word-level fields for the frontend.
        "word_to_correct": word_to_correct,
        "target_word": word_to_correct,
        "expected_word": expected,
        "recognized_word": recognized,

        # Compatibility with your existing frontend.
        "expected": expected,
        "recognized": recognized,

        # Normalized debug fields.
        "expected_norm": _join_norm(expected_tokens),
        "recognized_norm": _join_norm(recognized_tokens),

        # Word indices are learner-friendly compared with character position,
        # but still mostly useful for debugging/highlighting.
        "position": {
            "expected_word_index": gold_word_index,
            "recognized_word_index": pred_word_index,
        },
        "expected_word_index": gold_word_index,
        "recognized_word_index": pred_word_index,
        "expected_word_start": expected_start,
        "expected_word_end": expected_end,
        "recognized_word_start": recognized_start,
        "recognized_word_end": recognized_end,

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

        # New direct learner-friendly message fields.
        "learner_title": title_en,
        "learner_title_ar": title_ar,
        "learner_message": message_en,
        "learner_message_ar": message_ar,

        # Existing bilingual message shape.
        "message": {
            "en": message_en,
            "ar": message_ar,
        },
    }


def _append_items_for_opcode(
    *,
    items: list[dict[str, Any]],
    tag: str,
    gold_tokens: list[dict[str, Any]],
    pred_tokens: list[dict[str, Any]],
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    max_items: int,
) -> None:
    """
    Convert a SequenceMatcher opcode into exact word-level feedback items.

    For replacements, this avoids returning a whole phrase when only one or two
    words are wrong. It pairs words one-by-one, then treats leftovers as
    deletion/insertion.
    """
    if len(items) >= max_items:
        return

    if tag == "delete":
        for offset, token in enumerate(gold_tokens[i1:i2]):
            if len(items) >= max_items:
                return
            items.append(
                _build_item(
                    kind="deletion",
                    expected_tokens=[token],
                    recognized_tokens=[],
                    gold_word_index=i1 + offset,
                    pred_word_index=j1,
                )
            )
        return

    if tag == "insert":
        for offset, token in enumerate(pred_tokens[j1:j2]):
            if len(items) >= max_items:
                return
            items.append(
                _build_item(
                    kind="insertion",
                    expected_tokens=[],
                    recognized_tokens=[token],
                    gold_word_index=i1,
                    pred_word_index=j1 + offset,
                )
            )
        return

    if tag == "replace":
        expected_part = gold_tokens[i1:i2]
        recognized_part = pred_tokens[j1:j2]
        pair_count = min(len(expected_part), len(recognized_part))

        for offset in range(pair_count):
            if len(items) >= max_items:
                return
            items.append(
                _build_item(
                    kind="substitution",
                    expected_tokens=[expected_part[offset]],
                    recognized_tokens=[recognized_part[offset]],
                    gold_word_index=i1 + offset,
                    pred_word_index=j1 + offset,
                )
            )

        # Remaining expected words are missing.
        for offset in range(pair_count, len(expected_part)):
            if len(items) >= max_items:
                return
            items.append(
                _build_item(
                    kind="deletion",
                    expected_tokens=[expected_part[offset]],
                    recognized_tokens=[],
                    gold_word_index=i1 + offset,
                    pred_word_index=j2,
                )
            )

        # Remaining recognized words are extra.
        for offset in range(pair_count, len(recognized_part)):
            if len(items) >= max_items:
                return
            items.append(
                _build_item(
                    kind="insertion",
                    expected_tokens=[],
                    recognized_tokens=[recognized_part[offset]],
                    gold_word_index=i2,
                    pred_word_index=j1 + offset,
                )
            )
        return

    if len(items) < max_items:
        items.append(
            _build_item(
                kind="low_alignment_confidence",
                expected_tokens=gold_tokens[i1:i2],
                recognized_tokens=pred_tokens[j1:j2],
                gold_word_index=i1,
                pred_word_index=j1,
            )
        )


def build_content_feedback(
    *,
    gate: dict[str, Any] | None,
    reference: dict[str, Any] | None = None,
    mode: str = "guided",
    max_items: int = 5,
) -> dict[str, Any] | None:
    """
    Build bilingual learner-facing content feedback.

    Returns None when content is accepted.

    The important behavior is word-level feedback:
      - no whole-ayah error as the main message
      - exact word_to_correct / expected_word / recognized_word fields
    """
    if not gate:
        return None

    if gate.get("accepted"):
        return None

    reference = reference or {}

    # Use content_text first because the content gate compares against ASR-friendly
    # Quran text, not the Tajweed/Mushaf text.
    gold = str(
        gate.get("gold")
        or reference.get("content_text")
        or reference.get("text")
        or ""
    )
    pred = str(gate.get("pred") or "")

    gold_tokens = tokenize_with_norm(gold)
    pred_tokens = tokenize_with_norm(pred)

    gold_norm = [str(t["norm"]) for t in gold_tokens]
    pred_norm = [str(t["norm"]) for t in pred_tokens]

    items: list[dict[str, Any]] = []

    matcher = SequenceMatcher(a=gold_norm, b=pred_norm, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        _append_items_for_opcode(
            items=items,
            tag=tag,
            gold_tokens=gold_tokens,
            pred_tokens=pred_tokens,
            i1=i1,
            i2=i2,
            j1=j1,
            j2=j2,
            max_items=max_items,
        )

        if len(items) >= max_items:
            break

    if not items and normalize_arabic(gold) != normalize_arabic(pred):
        items.append(
            _build_item(
                kind="low_alignment_confidence",
                expected_tokens=gold_tokens[:1],
                recognized_tokens=pred_tokens[:1],
                gold_word_index=0,
                pred_word_index=0,
            )
        )

    items.sort(
        key=lambda x: (
            -int(x.get("severity_score") or 0),
            int(x.get("expected_word_index") or 0),
            int(x.get("recognized_word_index") or 0),
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

        # Full texts are kept for debug/display if needed, but the frontend
        # should prioritize item.word_to_correct / item.expected_word.
        "expected": gold,
        "recognized": pred,
        "metrics": {
            "char_accuracy": char_accuracy,
            "cer": cer,
            "edit_distance": edit_distance,
        },
        "items": items,
    }
