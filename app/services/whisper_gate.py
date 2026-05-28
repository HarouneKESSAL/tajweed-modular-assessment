

from __future__ import annotations
from difflib import SequenceMatcher
import re
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor





PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "checkpoints" / "content_asr_whisper_medium_quran_v2_weighted"


# Fixed muqatta'at normalization only.
# This is acceptable because the disconnected-letter openings are a limited,
# known set in the Quran. General rasm/imla'i differences are handled by
# content_text in ayah_reference.py, not here.
MUQATTAAT_COMPACT_MAP: dict[str, str] = {
    # البقرة، آل عمران، العنكبوت، الروم، لقمان، السجدة
    "الم": "الم",
    "الفلامميم": "الم",
    "اليفلامميم": "الم",

    # الأعراف
    "المص": "المص",
    "الفلامميمصاد": "المص",

    # يونس، هود، يوسف، إبراهيم، الحجر
    "الر": "الر",
    "الفلامراء": "الر",
    "الفلامرا": "الر",

    # الرعد
    "المر": "المر",
    "الفلامميمراء": "المر",
    "الفلامميمرا": "المر",

    # مريم
    "كهيعص": "كهيعص",
    "كافهاءياءعينصاد": "كهيعص",
    "كافهاياءعينصاد": "كهيعص",

    # طه
    "طه": "طه",
    "طاهاء": "طه",
    "طاها": "طه",

    # الشعراء والقصص
    "طسم": "طسم",
    "طاسينميم": "طسم",
    "طاسيميم": "طسم",

    # النمل
    "طس": "طس",
    "طاسين": "طس",

    # يس
    "يس": "يس",
    "ياسين": "يس",
    "ياسن": "يس",

    # ص
    "ص": "ص",
    "صاد": "ص",

    # غافر، فصلت، الزخرف، الدخان، الجاثية، الأحقاف
    "حم": "حم",
    "حاميم": "حم",
    "حاءميم": "حم",

    # الشورى
    "عسق": "عسق",
    "عينسينقاف": "عسق",

    # ق
    "ق": "ق",
    "قاف": "ق",

    # القلم
    "ن": "ن",
    "نون": "ن",
}

MUQATTAAT_CANONICAL = {
    "الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس",
    "يس", "ص", "حم", "عسق", "ق", "ن",
}

LETTER_NAME_TO_SYMBOL = {
    "الف": "ا",
    "ألف": "ا",
    "اليف": "ا",
    "لام": "ل",
    "ميم": "م",
    "صاد": "ص",
    "راء": "ر",
    "را": "ر",
    "كاف": "ك",
    "هاء": "ه",
    "ها": "ه",
    "ياء": "ي",
    "يا": "ي",
    "عين": "ع",
    "طاء": "ط",
    "طا": "ط",
    "سين": "س",
    "حاء": "ح",
    "حا": "ح",
    "قاف": "ق",
    "نون": "ن",
}

def normalize_muqattaat_prediction_if_needed(gold_compact: str, pred_text: str) -> str:
    """
    If the expected text is a muqatta'at opening, normalize the ASR prediction
    from spoken letter names into the written disconnected-letter form.

    Example:
        gold: الم
        pred: الف لام ميم
        output: الم
    """
    if gold_compact not in MUQATTAAT_CANONICAL:
        return content_compare_compact(pred_text)

    pred_norm = normalize_text(pred_text)
    pred_compact = compact_text(pred_norm)

    if pred_compact in MUQATTAAT_COMPACT_MAP:
        return MUQATTAAT_COMPACT_MAP[pred_compact]

    tokens = pred_norm.split()
    mapped = ""

    for token in tokens:
        token_norm = normalize_text(token)
        mapped += LETTER_NAME_TO_SYMBOL.get(token_norm, token_norm)

    mapped_compact = compact_text(mapped)

    if mapped_compact in MUQATTAAT_CANONICAL:
        return mapped_compact

    return content_compare_compact(pred_text)

def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = text.replace("ـ", "")
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def content_compare_compact(text: str) -> str:
    """
    Compact form used by the content gate.

    General Quranic orthography differences should already be solved by using
    reference['content_text'] from ayah_reference.py.

    This function only handles muqatta'at, where the written form and spoken
    ASR form are structurally different, for example:
        الم  <->  الف لام ميم
    """
    compact = compact_text(text)
    return MUQATTAAT_COMPACT_MAP.get(compact, compact)


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]

        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))

        previous = current

    return previous[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0

    if not gold:
        return 0.0

    sm = SequenceMatcher(a=gold, b=pred)
    matches = sum(block.size for block in sm.get_matching_blocks())
    return matches / max(1, len(gold))


def load_pcm16_wav_16k_mono(audio_path: Path) -> np.ndarray:
    """
    Read the FFmpeg-converted WAV without torchaudio/torchcodec.

    Expected format:
      - WAV
      - 16 kHz
      - mono
      - PCM signed 16-bit
    """
    with wave.open(str(audio_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz WAV, got {sample_rate} Hz: {audio_path}")

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got sample_width={sample_width}: {audio_path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


@lru_cache(maxsize=1)
def load_whisper() -> tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = str(DEFAULT_MODEL_DIR)

    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ar",
        task="transcribe",
    )

    return processor, model, device


def transcribe_audio(audio_path: Path, max_new_tokens: int = 128) -> str:
    processor, model, device = load_whisper()
    audio = load_pcm16_wav_16k_mono(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        pred_ids = model.generate(
            input_features,
            max_new_tokens=max_new_tokens,
        )

    pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return normalize_text(pred)


def run_content_gate(
    audio_path: Path,
    gold_text: str,
    mode: str = "strict",
    pred_text: str | None = None,
) -> dict[str, Any]:
    # Human-readable forms for UI/debug.
    gold_norm = normalize_text(gold_text)
    pred_norm = normalize_text(pred_text) if pred_text is not None else transcribe_audio(audio_path)

    # Comparison forms for the content decision.
    gold_compact = content_compare_compact(gold_norm)
    pred_compact = normalize_muqattaat_prediction_if_needed(gold_compact, pred_norm)

    gold_raw_compact = compact_text(gold_norm)
    pred_raw_compact = compact_text(pred_norm)

    ed = levenshtein(gold_compact, pred_compact)
    cer = ed / max(1, len(gold_compact))
    acc = char_accuracy(gold_compact, pred_compact)

    exact = pred_compact == gold_compact

    if mode == "strict":
        accepted = exact
    elif mode == "cer":
        accepted = cer <= 0.03
    else:
        accepted = exact

    if exact:
        verdict = "accepted_exact"
    elif accepted:
        verdict = "accepted_cer"
    else:
        verdict = "rejected_content_mismatch"

    return {
        "accepted": bool(accepted),
        "verdict": verdict,
        "mode": mode,
        "exact": bool(exact),

        "gold": gold_norm,
        "pred": pred_norm,

        "gold_compact": gold_compact,
        "pred_compact": pred_compact,
        "gold_raw_compact": gold_raw_compact,
        "pred_raw_compact": pred_raw_compact,
        "content_normalization_applied": bool(
            gold_compact != gold_raw_compact or pred_compact != pred_raw_compact
        ),

        "char_accuracy": float(acc),
        "cer": float(cer),
        "edit_distance": int(ed),
        "gold_len": len(gold_compact),
        "pred_len": len(pred_compact),
    }