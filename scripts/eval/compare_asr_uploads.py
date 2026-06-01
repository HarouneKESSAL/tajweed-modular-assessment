from __future__ import annotations

import argparse
import csv
import gc
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app.services.ayah_reference import normalize_text


DEFAULT_MODELS = [
    "checkpoints/content_asr_whisper_medium_quran_v2_weighted",
    "tarteel-ai/whisper-base-ar-quran",
    "IJyad/whisper-large-v3-Tarteel",
]


AUDIO_EXTENSIONS = {".wav"}


def find_audio_files(audio_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p
        for p in audio_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    # Skip generated tiny/empty files if any exist.
    return sorted(files)


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


def transcribe(
    *,
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


def evaluate_model(model_id: str, audio_files: list[Path]) -> list[dict]:
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

    rows: list[dict] = []

    for idx, audio_path in enumerate(audio_files, start=1):
        try:
            duration = audio_duration_sec(audio_path)

            start = time.perf_counter()
            pred_text = transcribe(
                processor=processor,
                model=model,
                device=device,
                audio_path=audio_path,
            )
            elapsed = time.perf_counter() - start

            print(
                f"[{idx}/{len(audio_files)}] {audio_path} "
                f"duration={duration:.2f}s time={elapsed:.2f}s"
            )

            rows.append(
                {
                    "model": model_id,
                    "audio_path": str(audio_path),
                    "duration_sec": duration,
                    "transcription_time_sec": elapsed,
                    "speed_ratio": elapsed / max(0.001, duration),
                    "pred_text": pred_text,
                    "pred_len": len(pred_text),
                    "status": "ok",
                    "error": "",
                }
            )

        except Exception as exc:
            print(f"[ERROR] {audio_path}: {exc}")

            rows.append(
                {
                    "model": model_id,
                    "audio_path": str(audio_path),
                    "duration_sec": "",
                    "transcription_time_sec": "",
                    "speed_ratio": "",
                    "pred_text": "",
                    "pred_len": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    del model
    del processor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}

    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    print("\n=== Summary ===")

    for model_id, model_rows in by_model.items():
        ok_rows = [r for r in model_rows if r["status"] == "ok"]

        if not ok_rows:
            print(f"\n{model_id}: no successful transcriptions")
            continue

        avg_time = sum(float(r["transcription_time_sec"]) for r in ok_rows) / len(ok_rows)
        avg_speed = sum(float(r["speed_ratio"]) for r in ok_rows) / len(ok_rows)
        avg_pred_len = sum(int(r["pred_len"]) for r in ok_rows) / len(ok_rows)

        print(f"\n{model_id}")
        print(f"  files: {len(model_rows)}")
        print(f"  successful: {len(ok_rows)}")
        print(f"  avg transcription time: {avg_time:.2f}s")
        print(f"  avg speed ratio: {avg_speed:.2f}x audio duration")
        print(f"  avg output length: {avg_pred_len:.1f} chars")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/uploads"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/asr_uploads_model_comparison.csv"),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only the first N files for quick testing. 0 means all files.",
    )
    args = parser.parse_args()

    audio_files = find_audio_files(
        args.audio_dir,
        recursive=not args.no_recursive,
    )

    if args.limit and args.limit > 0:
        audio_files = audio_files[: args.limit]

    print(f"Found {len(audio_files)} audio files in {args.audio_dir}")

    if not audio_files:
        return

    all_rows: list[dict] = []

    for model_id in args.models:
        all_rows.extend(evaluate_model(model_id, audio_files))

    write_csv(args.output, all_rows)
    print_summary(all_rows)

    print(f"\nWrote results to: {args.output}")


if __name__ == "__main__":
    main()