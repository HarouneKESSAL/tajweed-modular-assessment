from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


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


def get_text(row: dict[str, Any]) -> str:
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
            return str(value)
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


def levenshtein_seq(a: list[Any], b: list[Any]) -> int:
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold:
        return 0.0
    ed = levenshtein_seq(list(gold), list(pred))
    return max(0.0, 1.0 - ed / max(1, len(gold)))


def score_one(gold_raw: str, pred_raw: str) -> dict[str, Any]:
    gold_norm = normalize_arabic(gold_raw, compact=False)
    pred_norm = normalize_arabic(pred_raw, compact=False)
    gold_compact = normalize_arabic(gold_raw, compact=True)
    pred_compact = normalize_arabic(pred_raw, compact=True)

    char_ed = levenshtein_seq(list(gold_compact), list(pred_compact))
    word_ed = levenshtein_seq(gold_norm.split(), pred_norm.split())

    return {
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "gold_compact": gold_compact,
        "pred_compact": pred_compact,
        "exact_norm": gold_norm == pred_norm,
        "exact_compact": gold_compact == pred_compact,
        "char_edit_distance": char_ed,
        "word_edit_distance": word_ed,
        "char_accuracy": char_accuracy(gold_compact, pred_compact),
        "gold_char_len": len(gold_compact),
        "pred_char_len": len(pred_compact),
        "gold_words": len(gold_norm.split()),
        "pred_words": len(pred_norm.split()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    manifest_path = resolve_path(args.manifest)
    rows = filter_rows(load_jsonl(manifest_path), args.split, args.limit)

    model_dir = resolve_path(args.model_dir)
    processor = WhisperProcessor.from_pretrained(str(model_dir), language="Arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir)).to(device).eval()

    model.generation_config.language = "arabic"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")

    print("Whisper Quran ASR evaluation")
    print("----------------------------")
    print("model:", model_dir)
    print("manifest:", manifest_path)
    print("split:", args.split)
    print("samples:", len(rows))
    print("device:", device)

    results = []

    for i in tqdm(range(0, len(rows), args.batch_size)):
        batch_rows = rows[i : i + args.batch_size]
        audios = []
        for row in batch_rows:
            audio, _sr = librosa.load(str(get_audio_path(row)), sr=16000, mono=True)
            audios.append(audio)

        features = processor.feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            generated = model.generate(
                features,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

        preds = processor.batch_decode(generated, skip_special_tokens=True)

        for local_idx, (row, pred_raw) in enumerate(zip(batch_rows, preds)):
            global_idx = i + local_idx
            gold_raw = get_text(row)
            scores = score_one(gold_raw, pred_raw)

            result = {
                "id": get_id(row, global_idx),
                "audio_path": str(get_audio_path(row)),
                "gold_raw": gold_raw,
                "pred_raw": pred_raw,
                **scores,
            }
            results.append(result)

    n = len(results)
    total_chars = sum(r["gold_char_len"] for r in results)
    total_char_ed = sum(r["char_edit_distance"] for r in results)
    total_words = sum(r["gold_words"] for r in results)
    total_word_ed = sum(r["word_edit_distance"] for r in results)

    summary = {
        "model_dir": str(model_dir),
        "manifest": str(manifest_path),
        "split": args.split,
        "samples": n,
        "exact_norm_rate": sum(1 for r in results if r["exact_norm"]) / max(1, n),
        "exact_compact_rate": sum(1 for r in results if r["exact_compact"]) / max(1, n),
        "avg_char_accuracy": sum(r["char_accuracy"] for r in results) / max(1, n),
        "cer": total_char_ed / max(1, total_chars),
        "wer": total_word_ed / max(1, total_words),
        "avg_gold_char_len": total_chars / max(1, n),
        "avg_pred_char_len": sum(r["pred_char_len"] for r in results) / max(1, n),
    }

    out_jsonl = resolve_path(args.output_jsonl)
    out_summary = resolve_path(args.output_summary_json)
    out_md = resolve_path(args.output_md)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    worst = sorted(results, key=lambda r: r["char_accuracy"])[:30]
    best = sorted(results, key=lambda r: r["char_accuracy"], reverse=True)[:20]

    lines = []
    lines.append("# Whisper Quran ASR evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for r in worst:
        lines.append(f"### {r['id']}")
        lines.append(f"- gold: `{r['gold_norm']}`")
        lines.append(f"- pred: `{r['pred_norm']}`")
        lines.append(f"- char_accuracy: {r['char_accuracy']:.3f}")
        lines.append(f"- CER contribution edit/gold_len: {r['char_edit_distance']}/{r['gold_char_len']}")
        lines.append("")

    lines.append("")
    lines.append("## Best examples")
    lines.append("")
    for r in best:
        lines.append(f"### {r['id']}")
        lines.append(f"- gold: `{r['gold_norm']}`")
        lines.append(f"- pred: `{r['pred_norm']}`")
        lines.append(f"- char_accuracy: {r['char_accuracy']:.3f}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("summary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved:", out_jsonl)
    print("saved:", out_summary)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
