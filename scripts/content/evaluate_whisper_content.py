from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.evaluation.content_metrics import compute_content_metrics
from tajweed_assessment.text.normalization import normalize_arabic_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_audio_path(row: dict[str, Any]) -> Path:
    for key in ("audio_path", "path", "wav_path", "mp3_path"):
        value = row.get(key)
        if value:
            return resolve_path(str(value))
    raise KeyError(f"Could not find audio path in row keys: {sorted(row.keys())}")


def get_expected_text(row: dict[str, Any]) -> str:
    candidate_keys = (
        "expected_text",
        "normalized_text",
        "text",
        "chunk_text",
        "target_text",
        "transcript",
        "original_text",
    )

    for key in candidate_keys:
        value = row.get(key)
        if value:
            return str(value).strip()

    raise KeyError(f"Could not find expected text in row keys: {sorted(row.keys())}")


def get_sample_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id", "utt_id", "audio_id"):
        value = row.get(key)
        if value:
            return str(value)
    return f"sample_{index}"


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return None


def get_time_bounds(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Return chunk start/end in seconds if the manifest contains timing fields.

    If no timing fields exist, the full audio file will be used.
    """
    start_keys = (
        "start_time",
        "start_sec",
        "start_seconds",
        "chunk_start",
        "chunk_start_time",
        "chunk_start_sec",
        "begin_time",
        "begin_sec",
    )
    end_keys = (
        "end_time",
        "end_sec",
        "end_seconds",
        "chunk_end",
        "chunk_end_time",
        "chunk_end_sec",
        "finish_time",
        "finish_sec",
    )

    return _first_float(row, start_keys), _first_float(row, end_keys)


def get_sample_bounds(row: dict[str, Any]) -> tuple[int | None, int | None]:
    """
    Return chunk start/end as sample indices if the manifest contains sample fields.

    These are used only when time-in-seconds fields are not available.
    """
    start_keys = (
        "start_sample",
        "chunk_start_sample",
        "sample_start",
        "begin_sample",
    )
    end_keys = (
        "end_sample",
        "chunk_end_sample",
        "sample_end",
        "finish_sample",
    )

    return _first_int(row, start_keys), _first_int(row, end_keys)


def load_audio_with_soundfile(audio_path: Path) -> tuple[np.ndarray, int]:
    """
    Load audio using soundfile instead of torchaudio.load.

    This avoids the TorchCodec dependency path on newer torchaudio versions.
    """
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)

    if audio.ndim == 2:
        # soundfile shape is [time, channels], so average over channels.
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"Loaded empty audio file: {audio_path}")

    return audio, int(sample_rate)


def crop_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    start_time: float | None,
    end_time: float | None,
    start_sample: int | None,
    end_sample: int | None,
) -> np.ndarray:
    """
    Crop audio to the chunk if timing exists.

    Priority:
    1. start_time/end_time in seconds
    2. start_sample/end_sample
    3. full audio
    """
    total_samples = len(audio)

    if start_time is not None or end_time is not None:
        left = int((start_time or 0.0) * sample_rate)
        right = int(end_time * sample_rate) if end_time is not None else total_samples
    elif start_sample is not None or end_sample is not None:
        left = start_sample or 0
        right = end_sample if end_sample is not None else total_samples
    else:
        return audio

    left = max(0, min(left, total_samples))
    right = max(left, min(right, total_samples))

    cropped = audio[left:right]

    if cropped.size == 0:
        raise ValueError(
            f"Crop produced empty audio: left={left}, right={right}, "
            f"total_samples={total_samples}"
        )

    return cropped


def load_audio_16khz(
    audio_path: Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    start_sample: int | None = None,
    end_sample: int | None = None,
) -> torch.Tensor:
    """
    Load audio, optionally crop it to a chunk, convert to mono, and resample to 16 kHz.

    Whisper expects 16 kHz mono audio.
    """
    audio, sample_rate = load_audio_with_soundfile(audio_path)

    audio = crop_audio(
        audio,
        sample_rate,
        start_time=start_time,
        end_time=end_time,
        start_sample=start_sample,
        end_sample=end_sample,
    )

    waveform = torch.from_numpy(audio).float()

    if sample_rate != 16000:
        # Resample expects shape [..., time]. Use [1, time] then squeeze back.
        waveform = waveform.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )
        waveform = resampler(waveform).squeeze(0)

    return waveform.contiguous()


class WhisperContentEvaluator:
    def __init__(
        self,
        model_name: str,
        language: str,
        task: str,
        device: str,
        max_new_tokens: int,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "[warning] CUDA was requested, but this PyTorch install has no CUDA. "
                "Falling back to CPU."
            )
            device = "cpu"

        self.device = torch.device(device)
        self.model_name = model_name
        self.language = language
        self.task = task
        self.max_new_tokens = max_new_tokens

        self.processor = WhisperProcessor.from_pretrained(
            model_name,
            language=language,
            task=task,
        )

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Prefer modern generation controls instead of custom forced_decoder_ids.
        # This avoids the warning about deprecated custom forced_decoder_ids.
        self.model.generation_config.language = language
        self.model.generation_config.task = task
        self.model.generation_config.forced_decoder_ids = None

    @torch.no_grad()
    def transcribe(self, waveform_16khz: torch.Tensor) -> str:
        audio_array = waveform_16khz.detach().cpu().numpy()

        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs.input_features.to(self.device)

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generate_kwargs: dict[str, Any] = {
            "input_features": input_features,
            "language": self.language,
            "task": self.task,
            "max_new_tokens": self.max_new_tokens,
        }

        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        predicted_ids = self.model.generate(**generate_kwargs)

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return text.strip()


def evaluate_manifest(
    *,
    manifest_path: Path,
    output_json: Path,
    model_name: str,
    language: str,
    task: str,
    device: str,
    max_samples: int | None,
    start_index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    if start_index < 0 or start_index >= len(rows):
        raise IndexError(f"start-index out of range: 0 <= start-index < {len(rows)}")

    selected_rows = rows[start_index:]
    if max_samples is not None:
        selected_rows = selected_rows[:max_samples]

    evaluator = WhisperContentEvaluator(
        model_name=model_name,
        language=language,
        task=task,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    predictions: list[str] = []
    targets: list[str] = []
    items: list[dict[str, Any]] = []

    total = len(selected_rows)

    for local_idx, row in enumerate(selected_rows):
        global_idx = start_index + local_idx
        sample_id = get_sample_id(row, global_idx)
        audio_path = get_audio_path(row)
        expected_text = get_expected_text(row)

        start_time, end_time = get_time_bounds(row)
        start_sample, end_sample = get_sample_bounds(row)

        chunk_duration = None
        if start_time is not None and end_time is not None:
            chunk_duration = max(0.0, float(end_time) - float(start_time))

        print(f"[{local_idx + 1}/{total}] {sample_id} -> {audio_path}")
        if chunk_duration is not None:
            print(f"  chunk_sec: {chunk_duration:.3f}")
        elif start_sample is not None or end_sample is not None:
            print(f"  chunk_samples: start={start_sample} end={end_sample}")
        else:
            print("  chunk_sec: full audio, no timing fields found")

        try:
            waveform = load_audio_16khz(
                audio_path,
                start_time=start_time,
                end_time=end_time,
                start_sample=start_sample,
                end_sample=end_sample,
            )

            audio_duration_16khz = waveform.numel() / 16000.0
            print(f"  audio_to_whisper_sec: {audio_duration_16khz:.3f}")

            predicted_text = evaluator.transcribe(waveform)
            error = None

            print(f"  expected : {expected_text}")
            print(f"  predicted: {predicted_text}")
        except Exception as exc:
            predicted_text = ""
            error = repr(exc)
            print(f"  ERROR: {error}")

        predictions.append(predicted_text)
        targets.append(expected_text)

        expected_normalized = normalize_arabic_text(expected_text)
        predicted_normalized = normalize_arabic_text(predicted_text)

        item = {
            "index": global_idx,
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "start_time": start_time,
            "end_time": end_time,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "chunk_duration_sec": chunk_duration,
            "expected_text": expected_text,
            "predicted_text": predicted_text,
            "expected_normalized": expected_normalized,
            "predicted_normalized": predicted_normalized,
            "strict_match": predicted_text == expected_text,
            "normalized_match": predicted_normalized == expected_normalized,
        }

        if error is not None:
            item["error"] = error

        items.append(item)

    metrics = compute_content_metrics(predictions, targets)
    error_count = sum(1 for item in items if "error" in item)

    result = {
        "model_name": model_name,
        "language": language,
        "task": task,
        "device": str(evaluator.device),
        "manifest": str(manifest_path),
        "start_index": start_index,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "samples": metrics.samples,
        "error_count": error_count,
        "metrics": {
            "exact_match": metrics.exact_match,
            "normalized_exact_match": metrics.normalized_exact_match,
            "char_accuracy": metrics.char_accuracy,
            "normalized_char_accuracy": metrics.normalized_char_accuracy,
            "mean_edit_distance": metrics.mean_edit_distance,
            "normalized_mean_edit_distance": metrics.normalized_mean_edit_distance,
        },
        "items": items,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper zero-shot transcription on a content manifest."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSONL manifest containing audio_path and expected text.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to save evaluation JSON.",
    )
    parser.add_argument(
        "--model-name",
        default="openai/whisper-small",
        help="Hugging Face Whisper model name.",
    )
    parser.add_argument(
        "--language",
        default="arabic",
        help="Whisper language prompt.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper task.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of samples to evaluate. Use -1 for all samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in the manifest.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens Whisper may generate per chunk.",
    )

    args = parser.parse_args()

    max_samples = None if args.max_samples < 0 else args.max_samples

    result = evaluate_manifest(
        manifest_path=resolve_path(args.manifest),
        output_json=resolve_path(args.output_json),
        model_name=args.model_name,
        language=args.language,
        task=args.task,
        device=args.device,
        max_samples=max_samples,
        start_index=args.start_index,
        max_new_tokens=args.max_new_tokens,
    )

    print("")
    print("Whisper content evaluation summary")
    print("----------------------------------")
    print(f"model_name                   : {result['model_name']}")
    print(f"device                       : {result['device']}")
    print(f"samples                      : {result['samples']}")
    print(f"errors                       : {result['error_count']}")
    print(f"exact_match                  : {result['metrics']['exact_match']:.3f}")
    print(f"normalized_exact_match       : {result['metrics']['normalized_exact_match']:.3f}")
    print(f"char_accuracy                : {result['metrics']['char_accuracy']:.3f}")
    print(f"normalized_char_accuracy     : {result['metrics']['normalized_char_accuracy']:.3f}")
    print(f"mean_edit_distance           : {result['metrics']['mean_edit_distance']:.3f}")
    print(
        "normalized_mean_edit_distance: "
        f"{result['metrics']['normalized_mean_edit_distance']:.3f}"
    )
    print(f"saved                        : {resolve_path(args.output_json)}")


if __name__ == "__main__":
    main()