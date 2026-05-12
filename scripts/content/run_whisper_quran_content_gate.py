from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


MUQATTAAT = {
    "الم", "الر", "المر", "المص", "كهيعص", "طه", "طسم", "طس",
    "يس", "ص", "حم", "عسق", "ق", "ن"
}


# Conservative mapping: only used when gold is one of the disconnected-letter ayahs.
MUQATTAAT_SPOKEN_TO_SCRIPT = {
    # الم
    "الفلامميم": "الم",
    "الفلاميم": "الم",
    "الافلاميم": "الم",
    "الافلامميم": "الم",
    "الاخلامميم": "الم",
    "الاخلاميم": "الم",
    "الف لام ميم": "الم",
    "اليف لام ميم": "الم",
    "الفلام ميم": "الم",
    "الاخلام ميم": "الم",

    # يس
    "ياسين": "يس",
    "يا سين": "يس",
    "ياسن": "يس",

    # طسم
    "طسم": "طسم",
    "طاسيم": "طسم",
    "طاسينميم": "طسم",
    "طا سين ميم": "طسم",
    "باسيم": "طسم",
    "قاسيمميم": "طسم",
    "قاسيم ميم": "طسم",

    # عسق
    "عينسينقاف": "عسق",
    "عين سين قاف": "عسق",
    "عين سينقاف": "عسق",
    "عنسنقاف": "عسق",

    # حم
    "حاميم": "حم",
    "حا ميم": "حم",

    # طه
    "طاها": "طه",
    "طا ها": "طه",

    # ق / ن / ص
    "قاف": "ق",
    "نون": "ن",
    "صاد": "ص",
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("audio_id") or "")


def get_audio_path(row: dict[str, Any]) -> str:
    for key in ["audio_path", "audio", "path", "wav_path"]:
        if row.get(key):
            return str(row[key])
    raise KeyError(f"No audio path key found in row keys: {sorted(row.keys())}")


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "normalized_text", "target", "transcript"]:
        if row.get(key):
            return str(row[key])
    raise KeyError(f"No text key found in row keys: {sorted(row.keys())}")


def normalize_arabic(text: str) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")  # tatweel
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def normalize_muqattaat_prediction(gold_norm: str, pred_norm: str) -> str:
    gold_c = compact(gold_norm)
    pred = str(pred_norm or "").strip()
    pred_c = compact(pred)

    if gold_c not in MUQATTAAT:
        return pred_norm

    if pred in MUQATTAAT_SPOKEN_TO_SCRIPT:
        return MUQATTAAT_SPOKEN_TO_SCRIPT[pred]
    if pred_c in MUQATTAAT_SPOKEN_TO_SCRIPT:
        return MUQATTAAT_SPOKEN_TO_SCRIPT[pred_c]

    return pred_norm


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (a[i - 1] != b[j - 1]),
            )
            prev = old
    return dp[n]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0
    ed = levenshtein(gold, pred)
    return max(0.0, 1.0 - ed / max(1, len(gold)))


def load_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        return wav.float(), sr
    except Exception:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32")
        wav = torch.tensor(audio)
        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        return wav.float(), sr


def run_whisper_asr(
    *,
    model_dir: str,
    audio_path: str,
    device: str,
    max_new_tokens: int,
) -> str:
    model_path = project_path(model_dir)
    audio = project_path(audio_path)

    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path)).to(device)
    model.eval()

    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ar",
            task="transcribe",
        )
    except Exception:
        pass

    wav, sr = load_audio_16k(audio)
    inputs = processor(
        wav.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
    )
    features = inputs.input_features.to(device)

    with torch.no_grad():
        generated = model.generate(
            features,
            max_new_tokens=max_new_tokens,
        )

    pred = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return pred


def score_content(gold_text: str, pred_text: str, *, mode: str, min_char_accuracy: float, max_cer: float) -> dict[str, Any]:
    gold_norm = normalize_arabic(gold_text)
    pred_norm_raw = normalize_arabic(pred_text)
    pred_norm = normalize_muqattaat_prediction(gold_norm, pred_norm_raw)

    gold_c = compact(gold_norm)
    pred_c_raw = compact(pred_norm_raw)
    pred_c = compact(pred_norm)

    ed = levenshtein(gold_c, pred_c)
    cer = ed / max(1, len(gold_c))
    acc = char_accuracy(gold_c, pred_c)
    exact = pred_c == gold_c

    if exact:
        verdict = "accepted_exact"
        accepted = True
    elif mode == "near_exact" and acc >= min_char_accuracy and cer <= max_cer:
        verdict = "accepted_near_exact_review_recommended"
        accepted = True
    else:
        verdict = "rejected_content_mismatch"
        accepted = False

    return {
        "accepted": accepted,
        "verdict": verdict,
        "mode": mode,
        "gold_norm": gold_norm,
        "pred_norm_raw": pred_norm_raw,
        "pred_norm": pred_norm,
        "gold_compact": gold_c,
        "pred_compact_raw": pred_c_raw,
        "pred_compact": pred_c,
        "exact_compact": exact,
        "char_accuracy": acc,
        "cer": cer,
        "edit_distance": ed,
        "gold_len": len(gold_c),
        "pred_len": len(pred_c),
        "muqattaat_normalized": pred_c != pred_c_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--audio-path")
    parser.add_argument("--gold-text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mode", choices=["strict", "near_exact"], default="strict")
    parser.add_argument("--min-char-accuracy", type=float, default=0.98)
    parser.add_argument("--max-cer", type=float, default=0.02)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    row = None
    if args.manifest:
        rows = load_jsonl(project_path(args.manifest))
        if args.split != "all":
            rows = [r for r in rows if str(r.get("split", "train")) == args.split]
        row = rows[args.sample_index]
        audio_path = get_audio_path(row)
        gold_text = get_text(row)
        sample_id = get_row_id(row)
    else:
        if not args.audio_path or args.gold_text is None:
            raise SystemExit("Provide either --manifest or both --audio-path and --gold-text.")
        audio_path = args.audio_path
        gold_text = args.gold_text
        sample_id = ""

    pred_text = run_whisper_asr(
        model_dir=args.model_dir,
        audio_path=audio_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    score = score_content(
        gold_text,
        pred_text,
        mode=args.mode,
        min_char_accuracy=args.min_char_accuracy,
        max_cer=args.max_cer,
    )

    result = {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "model_dir": str(project_path(args.model_dir)),
        "gold_text": gold_text,
        "pred_text": pred_text,
        "content_gate": score,
    }

    print("Whisper Quran content gate")
    print("--------------------------")
    print(f"Sample ID : {sample_id}")
    print(f"Audio     : {audio_path}")
    print(f"Model     : {project_path(args.model_dir)}")
    print("")
    print(f"Gold      : {score['gold_norm']}")
    print(f"Pred raw  : {score['pred_norm_raw']}")
    if score["muqattaat_normalized"]:
        print(f"Pred norm : {score['pred_norm']}")
    print("")
    print(json.dumps(score, ensure_ascii=False, indent=2))

    if args.output_json:
        out = project_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
