from __future__ import annotations

import argparse
import csv
import gc
import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app.services.ayah_reference import get_ayah_reference, normalize_text, levenshtein
from app.services.whisper_gate import content_compare_compact


DEFAULT_MODELS = [
    "checkpoints/content_asr_whisper_medium_quran_v2_weighted",
    "tarteel-ai/whisper-base-ar-quran",
    "IJyad/whisper-large-v3-Tarteel",
]


def load_pcm16_wav_16k_mono(audio_path: Path) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16000 Hz WAV, got {sample_rate}: {audio_path}")

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got {sample_width}: {audio_path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


def audio_duration_sec(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def estimate_max_new_tokens(audio_path: Path) -> int:
    duration = audio_duration_sec(audio_path)
    return max(16, min(192, int(duration * 8) + 16))


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0

    # Simple character accuracy based on edit distance.
    ed = levenshtein(gold, pred)
    return max(0.0, 1.0 - (ed / max(1, len(gold))))


def transcribe(
    *,
    model_id: str,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    audio_path: Path,
) -> str:
    audio = load_pcm16_wav_16k_mono(audio_path)
    max_new_tokens = estimate_max_new_tokens(audio_path)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features.to(device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 5,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 4,
        "early_stopping": True,
    }

    if "attention_mask" in inputs:
        generate_kwargs["attention_mask"] = inputs.attention_mask.to(device)

    with torch.no_grad():
        pred_ids = model.generate(input_features, **generate_kwargs)

    pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return normalize_text(pred)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))

    return rows


def evaluate_model(model_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    print(f"\nLoading model: {model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()

    try:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="ar",
            task="transcribe",
        )
    except Exception:
        pass

    output_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        audio_path = Path(row["audio_path"])
        surah = int(row["surah"])
        ayah = int(row["ayah"])

        reference = get_ayah_reference(surah, ayah)
        gold_text = reference.get("content_text") or reference["text"]

        start = time.perf_counter()
        pred_text = transcribe(
            model_id=model_id,
            processor=processor,
            model=model,
            device=device,
            audio_path=audio_path,
        )
        elapsed = time.perf_counter() - start

        gold_compact = content_compare_compact(gold_text)
        pred_compact = content_compare_compact(pred_text)

        ed = levenshtein(gold_compact, pred_compact)
        cer = ed / max(1, len(gold_compact))
        acc = char_accuracy(gold_compact, pred_compact)
        exact = gold_compact == pred_compact

        accepted_segment = exact or cer <= 0.05 or acc >= 0.95

        print(
            f"[{idx}/{len(rows)}] {surah}:{ayah} "
            f"CER={cer:.3f} ACC={acc:.3f} exact={exact} "
            f"time={elapsed:.2f}s"
        )

        output_rows.append(
            {
                "model": model_id,
                "audio_path": str(audio_path),
                "surah": surah,
                "ayah": ayah,
                "gold_text": gold_text,
                "pred_text": pred_text,
                "gold_compact": gold_compact,
                "pred_compact": pred_compact,
                "cer": cer,
                "char_accuracy": acc,
                "edit_distance": ed,
                "exact": exact,
                "accepted_segment": accepted_segment,
                "duration_sec": audio_duration_sec(audio_path),
                "transcription_time_sec": elapsed,
            }
        )

    del model
    del processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return output_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    print("\n=== Summary ===")

    for model_id, model_rows in by_model.items():
        n = len(model_rows)
        avg_cer = sum(float(r["cer"]) for r in model_rows) / max(1, n)
        avg_acc = sum(float(r["char_accuracy"]) for r in model_rows) / max(1, n)
        exact_rate = sum(1 for r in model_rows if r["exact"]) / max(1, n)
        accept_rate = sum(1 for r in model_rows if r["accepted_segment"]) / max(1, n)
        avg_time = sum(float(r["transcription_time_sec"]) for r in model_rows) / max(1, n)

        print(f"\n{model_id}")
        print(f"  samples: {n}")
        print(f"  avg CER: {avg_cer:.4f}")
        print(f"  avg char accuracy: {avg_acc:.4f}")
        print(f"  exact rate: {exact_rate:.2%}")
        print(f"  segment accept rate: {accept_rate:.2%}")
        print(f"  avg transcription time: {avg_time:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL with audio_path, surah, ayah.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/content_asr_model_comparison.csv"),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
    )
    args = parser.parse_args()

    rows = read_manifest(args.manifest)

    all_results: list[dict[str, Any]] = []

    for model_id in args.models:
        all_results.extend(evaluate_model(model_id, rows))

    write_csv(args.output, all_results)
    print_summary(all_results)

    print(f"\nWrote results to: {args.output}")


if __name__ == "__main__":
    main()