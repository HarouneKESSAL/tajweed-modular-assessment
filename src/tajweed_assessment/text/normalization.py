import re
import unicodedata


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

TATWEEL = "\u0640"


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalize_hamza(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
    )


def normalize_alef_maqsura(text: str) -> str:
    return text.replace("ى", "ي")


def normalize_ta_marbuta(text: str) -> str:
    return text.replace("ة", "ه")


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def normalize_arabic_text(
    text: str,
    *,
    strip_marks: bool = True,
    normalize_hamza_forms: bool = True,
    normalize_ya: bool = True,
    normalize_ta: bool = False,
) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(TATWEEL, "")

    if strip_marks:
        text = strip_diacritics(text)

    if normalize_hamza_forms:
        text = normalize_hamza(text)

    if normalize_ya:
        text = normalize_alef_maqsura(text)

    if normalize_ta:
        text = normalize_ta_marbuta(text)

    return normalize_spaces(text)