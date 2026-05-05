from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.models.content.whisper_ctc import (
    WhisperCTCConfig,
    WhisperCTCContentModel,
    ctc_greedy_decode,
)
from tajweed_assessment.text.normalization import normalize_arabic_text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_text(row: dict[str, Any]) -> str:
    for key in ("normalized_text", "expected_text", "text", "chunk_text", "target_text", "transcript"):
        value = row.get(key)
        if value:
            return normalize_arabic_text(str(value).strip())
    raise KeyError(f"No text field found in row keys: {sorted(row.keys())}")


def get_audio_path(row: dict[str, Any]) -> Path:
    for key in ("audio_path", "path", "wav_path", "mp3_path"):
        value = row.get(key)
        if value:
            return resolve_path(str(value))
    raise KeyError(f"No audio path found in row keys: {sorted(row.keys())}")


def load_audio_float32(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return audio, int(sample_rate)


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
    if cropped.size == 0:
        return audio
    return cropped


def resample_to_16khz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio.astype(np.float32)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform).squeeze(0)
    return waveform.numpy().astype(np.float32)


def build_char_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    chars = sorted({ch for row in rows for ch in get_text(row)})
    return {ch: idx + 1 for idx, ch in enumerate(chars)}  # 0 is CTC blank


def split_rows_text_holdout(
    rows: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


class WhisperCTCDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], char_to_id: dict[str, int]) -> None:
        self.rows = rows
        self.char_to_id = char_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = get_audio_path(row)
        audio, sample_rate = load_audio_float32(audio_path)
        audio = crop_audio(row, audio, sample_rate)
        audio = resample_to_16khz(audio, sample_rate)

        text = get_text(row)
        target = [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

        return {
            "id": row.get("id") or row.get("sample_id") or str(index),
            "audio": audio,
            "audio_len": int(len(audio)),
            "text": text,
            "target": torch.tensor(target, dtype=torch.long),
        }


class WhisperCTCCollator:
    def __init__(self, processor: WhisperProcessor) -> None:
        self.processor = processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        audio_arrays = [item["audio"] for item in items]
        features = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        )

        targets = [item["target"] for item in items]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        flat_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)

        return {
            "ids": [item["id"] for item in items],
            "texts": [item["text"] for item in items],
            "input_features": features.input_features,
            "audio_lengths": torch.tensor(
                [int(item["audio_len"]) for item in items],
                dtype=torch.long,
            ),
            "targets": flat_targets,
            "target_lengths": target_lengths,
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


def evaluate(
    model: WhisperCTCContentModel,
    loader: DataLoader,
    id_to_char: dict[int, str],
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total = 0
    exact = 0
    char_acc_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            input_features = batch["input_features"].to(device)
            log_probs = model(input_features)
            input_lengths = whisper_ctc_input_lengths(
                batch["audio_lengths"].to(device),
                log_probs.size(1),
            )
            decoded_ids = ctc_greedy_decode(log_probs, input_lengths=input_lengths.cpu())

            for gold, pred_ids in zip(batch["texts"], decoded_ids):
                pred = ids_to_text(pred_ids, id_to_char)
                total += 1
                exact += int(pred == gold)

                if gold:
                    dist = levenshtein(gold, pred)
                    char_acc_sum += max(0.0, 1.0 - dist / len(gold))
                else:
                    char_acc_sum += 1.0 if not pred else 0.0

    return {
        "samples": float(total),
        "exact_match": exact / total if total else 0.0,
        "char_accuracy": char_acc_sum / total if total else 0.0,
    }


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + int(ca != cb),
                )
            )
        prev = curr

    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--model-config", default="configs/model_content_whisper_ctc.yaml")
    parser.add_argument("--train-config", default="configs/train_content_whisper_ctc.yaml")
    parser.add_argument("--output-checkpoint", default="checkpoints/content_whisper_ctc_tiny.pt")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    model_cfg = load_yaml(resolve_path(args.model_config))
    train_cfg = load_yaml(resolve_path(args.train_config))

    seed = int(train_cfg.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)

    rows = load_jsonl(resolve_path(args.manifest))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise ValueError("No rows selected for training.")

    train_rows, val_rows = split_rows_text_holdout(
        rows,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        seed=seed,
    )

    max_train = int(train_cfg.get("max_train_samples", 0))
    max_val = int(train_cfg.get("max_val_samples", 0))

    if max_train > 0:
        train_rows = train_rows[:max_train]
    if max_val > 0:
        val_rows = val_rows[:max_val]

    char_to_id = build_char_vocab(rows)
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}

    model_name = str(model_cfg.get("model_name", "openai/whisper-tiny"))
    processor = WhisperProcessor.from_pretrained(model_name, language="arabic", task="transcribe")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    epochs = args.epochs or int(train_cfg.get("epochs", 3))
    device_name = args.device or str(train_cfg.get("device", "cpu"))

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    train_loader = DataLoader(
        WhisperCTCDataset(train_rows, char_to_id),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=WhisperCTCCollator(processor),
    )
    val_loader = DataLoader(
        WhisperCTCDataset(val_rows, char_to_id),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=WhisperCTCCollator(processor),
    )

    model = WhisperCTCContentModel(
        WhisperCTCConfig(
            model_name=model_name,
            num_labels=len(char_to_id) + 1,
            freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            head_type=str(model_cfg.get("head_type", "linear")),
            head_hidden_dim=int(model_cfg.get("head_hidden_dim", 128)),
            head_num_layers=int(model_cfg.get("head_num_layers", 1)),
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print("Whisper CTC training")
    print("--------------------")
    print(f"model_name     : {model_name}")
    print(f"device         : {device}")
    print(f"freeze_encoder : {model_cfg.get('freeze_encoder', True)}")
    print(f"train_samples  : {len(train_rows)}")
    print(f"val_samples    : {len(val_rows)}")
    print(f"vocab_size     : {len(char_to_id) + 1}")
    print(f"epochs         : {epochs}")

    best_val_char = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []

        for step, batch in enumerate(train_loader, start=1):
            input_features = batch["input_features"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)

            log_probs = model(input_features)  # [B, T, C]
            batch_size_now, time_steps, _classes = log_probs.shape
            input_lengths = whisper_ctc_input_lengths(
                batch["audio_lengths"].to(device),
                time_steps,
            )

            loss = criterion(
                log_probs.transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))

            if step % 5 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={losses[-1]:.4f}")

        val_metrics = evaluate(model, val_loader, id_to_char, device)
        mean_loss = sum(losses) / max(1, len(losses))

        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} "
            f"val_exact={val_metrics['exact_match']:.3f} "
            f"val_char={val_metrics['char_accuracy']:.3f}"
        )

        if val_metrics["char_accuracy"] > best_val_char:
            best_val_char = val_metrics["char_accuracy"]
            out_path = resolve_path(args.output_checkpoint)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "char_to_id": char_to_id,
                    "id_to_char": {str(k): v for k, v in id_to_char.items()},
                    "config": {
                        "model": model_cfg,
                        "train": train_cfg,
                        "manifest": args.manifest,
                    },
                    "val_metrics": val_metrics,
                },
                out_path,
            )
            print(f"saved best checkpoint: {out_path}")

    print(f"best_val_char_accuracy={best_val_char:.3f}")


if __name__ == "__main__":
    main()