from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.evaluation.content_metrics import compute_content_metrics
from tajweed_assessment.models.content.whisper_ctc import (
    WhisperCTCConfig,
    WhisperCTCContentModel,
    ctc_greedy_decode,
)
from tajweed_assessment.text.normalization import normalize_arabic_text


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text(row: dict[str, Any]) -> str:
    for key in ("normalized_text", "expected_text", "text", "chunk_text", "target_text", "transcript"):
        value = row.get(key)
        if value:
            return normalize_arabic_text(str(value).strip())
    raise KeyError(f"No text found in row: {sorted(row.keys())}")


def get_audio_path(row: dict[str, Any]) -> Path:
    for key in ("audio_path", "path", "wav_path", "mp3_path"):
        value = row.get(key)
        if value:
            return resolve_path(str(value))
    raise KeyError(f"No audio path found in row: {sorted(row.keys())}")


def split_rows_text_holdout(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_text: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_text.setdefault(get_text(row), []).append(row)

    texts = list(by_text)
    rng = random.Random(seed)
    rng.shuffle(texts)

    val_count = max(1, int(round(len(texts) * val_fraction)))
    val_texts = set(texts[:val_count])

    train_rows = [row for row in rows if get_text(row) not in val_texts]
    val_rows = [row for row in rows if get_text(row) in val_texts]
    return train_rows, val_rows


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def crop_audio(row: dict[str, Any], audio: np.ndarray, sample_rate: int) -> np.ndarray:
    start = row.get("start_sec")
    end = row.get("end_sec")
    if start is None or end is None:
        return audio

    start_i = max(0, int(float(start) * sample_rate))
    end_i = min(len(audio), int(float(end) * sample_rate))

    if end_i <= start_i:
        return audio

    cropped = audio[start_i:end_i]
    return cropped if cropped.size else audio


def resample_to_16khz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio.astype(np.float32)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    return resampler(waveform).squeeze(0).numpy().astype(np.float32)


class EvalDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio, sr = load_audio(get_audio_path(row))
        audio = crop_audio(row, audio, sr)
        audio = resample_to_16khz(audio, sr)

        return {
            "id": row.get("id") or row.get("sample_id") or str(index),
            "audio": audio,
            "audio_len": int(len(audio)),
            "text": get_text(row),
        }


class EvalCollator:
    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        features = self.processor(
            [item["audio"] for item in items],
            sampling_rate=16000,
            return_tensors="pt",
        )
        return {
            "ids": [item["id"] for item in items],
            "texts": [item["text"] for item in items],
            "input_features": features.input_features,
            "audio_lengths": torch.tensor(
                [int(item["audio_len"]) for item in items],
                dtype=torch.long,
            ),
        }


def ids_to_text(ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(int(i), "") for i in ids)


def whisper_ctc_input_lengths(
    audio_lengths_16khz: torch.Tensor,
    max_time_steps: int,
) -> torch.Tensor:
    """
    Convert raw 16 kHz audio lengths to Whisper encoder time lengths.

    Whisper uses 10 ms mel frames, then the encoder downsamples by 2, so the
    encoder has about one frame per 320 waveform samples.
    """
    lengths = torch.ceil(audio_lengths_16khz.float() / 320.0).long()
    return torch.clamp(lengths, min=1, max=max_time_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "full"], default="val")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    ckpt = torch.load(resolve_path(args.checkpoint), map_location="cpu")
    model_name = str(ckpt["model_name"])
    char_to_id = {str(k): int(v) for k, v in ckpt["char_to_id"].items()}
    id_to_char = {int(k): str(v) for k, v in ckpt["id_to_char"].items()}

    rows = load_jsonl(resolve_path(args.manifest))
    train_rows, val_rows = split_rows_text_holdout(rows, val_fraction=0.2, seed=1337)

    if args.split == "train":
        selected_rows = train_rows
    elif args.split == "val":
        selected_rows = val_rows
    else:
        selected_rows = rows

    if args.max_samples > 0:
        selected_rows = selected_rows[: args.max_samples]

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    processor = WhisperProcessor.from_pretrained(model_name, language="arabic", task="transcribe")

    model = WhisperCTCContentModel(
        WhisperCTCConfig(
            model_name=model_name,
            num_labels=len(char_to_id) + 1,
            freeze_encoder=True,
            dropout=0.0,
        )
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        EvalDataset(selected_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=EvalCollator(processor),
    )

    predictions: list[str] = []
    targets: list[str] = []
    items: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            log_probs = model(batch["input_features"].to(device))
            input_lengths = whisper_ctc_input_lengths(
                batch["audio_lengths"].to(device),
                log_probs.size(1),
            )
            decoded = ctc_greedy_decode(log_probs, input_lengths=input_lengths.cpu())

            for sample_id, gold, pred_ids in zip(batch["ids"], batch["texts"], decoded):
                pred = ids_to_text(pred_ids, id_to_char)
                predictions.append(pred)
                targets.append(gold)
                items.append(
                    {
                        "sample_id": sample_id,
                        "expected": gold,
                        "predicted": pred,
                        "exact": pred == gold,
                    }
                )

    metrics = compute_content_metrics(predictions, targets)

    result = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "model_name": model_name,
        "split": args.split,
        "samples": metrics.samples,
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

    out_path = resolve_path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Whisper CTC evaluation")
    print("----------------------")
    print(f"checkpoint             : {result['checkpoint']}")
    print(f"model_name             : {model_name}")
    print(f"split                  : {args.split}")
    print(f"samples                : {metrics.samples}")
    print(f"exact_match            : {metrics.exact_match:.3f}")
    print(f"normalized_exact_match : {metrics.normalized_exact_match:.3f}")
    print(f"char_accuracy          : {metrics.char_accuracy:.3f}")
    print(f"normalized_char_accuracy: {metrics.normalized_char_accuracy:.3f}")
    print(f"saved                  : {out_path}")


if __name__ == "__main__":
    main()