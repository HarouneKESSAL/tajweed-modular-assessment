from __future__ import annotations

import argparse
import inspect
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_arabic(text: str, compact: bool = False) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # Light normalization for ASR stability.
    text = text.replace("ٱ", "ا").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")

    text = re.sub(r"\s+", " ", text).strip()
    if compact:
        text = text.replace(" ", "")
    return text


def get_audio_path(row: dict[str, Any]) -> Path:
    for key in ["audio_path", "audio", "path", "wav_path", "audio_filepath", "file"]:
        value = row.get(key)
        if value:
            return resolve_path(value)
    raise KeyError(f"No audio path field found in row keys: {list(row.keys())}")


def get_text(row: dict[str, Any], compact_targets: bool) -> str:
    for key in [
        "normalized_text",
        "clean_text",
        "text",
        "transcript",
        "target",
        "ayah_text",
        "reference_text",
        "word",
    ]:
        value = row.get(key)
        if value:
            return normalize_arabic(value, compact=compact_targets)
    raise KeyError(f"No text field found in row keys: {list(row.keys())}")


def get_id(row: dict[str, Any], idx: int) -> str:
    for key in ["id", "sample_id", "audio_id", "utterance_id"]:
        value = row.get(key)
        if value:
            return str(value)
    return f"sample_{idx:06d}"


def filter_rows(rows: list[dict[str, Any]], split: str, limit: int = 0) -> list[dict[str, Any]]:
    if split == "all":
        selected = rows
    else:
        selected = [r for r in rows if str(r.get("split", "train")) == split]

    if not selected:
        raise RuntimeError(f"No rows found for split={split!r}. Available split values: {sorted(set(str(r.get('split', 'train')) for r in rows))}")

    if limit > 0:
        selected = selected[:limit]
    return selected


class QuranWhisperDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        processor: WhisperProcessor,
        compact_targets: bool = False,
        sample_rate: int = 16000,
    ) -> None:
        self.rows = rows
        self.processor = processor
        self.compact_targets = compact_targets
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        audio_path = get_audio_path(row)
        text = get_text(row, compact_targets=self.compact_targets)

        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="np",
        ).input_features[0]

        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": features,
            "labels": labels,
            "text": text,
            "id": get_id(row, idx),
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    kwargs = dict(
        output_dir=str(resolve_path(args.output_dir)),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    if args.eval_steps > 0:
        kwargs.update(
            do_eval=True,
            eval_steps=args.eval_steps,
        )
        try:
            return Seq2SeqTrainingArguments(**kwargs, evaluation_strategy="steps")
        except TypeError:
            return Seq2SeqTrainingArguments(**kwargs, eval_strategy="steps")

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--compact-targets", action="store_true")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--freeze-encoder", action="store_true")

    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    rows = load_jsonl(manifest_path)
    train_rows = filter_rows(rows, args.train_split, args.max_train_samples)
    eval_rows = filter_rows(rows, args.eval_split, args.max_eval_samples)

    print("Whisper Quran ASR training")
    print("--------------------------")
    print("manifest:", manifest_path)
    print("model:", args.model_name)
    print("train rows:", len(train_rows))
    print("eval rows:", len(eval_rows))
    print("compact_targets:", args.compact_targets)

    processor = WhisperProcessor.from_pretrained(args.model_name, language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
    model.config.suppress_tokens = []

    if args.freeze_encoder:
        model.freeze_encoder()
        print("encoder frozen")

    train_ds = QuranWhisperDataset(train_rows, processor, compact_targets=args.compact_targets)
    eval_ds = QuranWhisperDataset(eval_rows, processor, compact_targets=args.compact_targets)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = build_training_args(args)

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = processor.feature_extractor
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = processor.feature_extractor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    out_dir = resolve_path(args.output_dir)
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    meta = {
        "base_model": args.model_name,
        "manifest": str(manifest_path),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "compact_targets": args.compact_targets,
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_dir)


if __name__ == "__main__":
    main()
