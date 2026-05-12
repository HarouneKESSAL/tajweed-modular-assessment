from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
import wave
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule


ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"

BUCKETS = [
    ("001_020", 1, 20),
    ("021_040", 21, 40),
    ("041_060", 41, 60),
    ("061_090", 61, 90),
    ("091_140", 91, 140),
    ("141_plus", 141, 10000),
]


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_arabic_text(text: str, keep_spaces: bool = True) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[^ء-ي\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not keep_spaces:
        text = text.replace(" ", "")
    return text


def compact(text: str) -> str:
    return normalize_arabic_text(text, keep_spaces=False)


def get_raw_text(row: dict[str, Any]) -> str:
    return (
        row.get("text")
        or row.get("source_text")
        or row.get("normalized_text")
        or row.get("word")
        or ""
    )


def get_full_label(row: dict[str, Any]) -> str:
    return compact(row.get("normalized_text") or get_raw_text(row))


def bucket_name(text: str) -> str:
    n = len(compact(text))
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return "unknown"


def edit_distance(a: str, b: str) -> int:
    a = compact(a)
    b = compact(b)
    dp = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if ca == cb else 1),
            )
            prev = old

    return dp[-1]


def char_accuracy(gold: str, pred: str) -> float:
    gold = compact(gold)
    pred = compact(pred)
    if not gold:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - edit_distance(gold, pred) / len(gold))


def to_int_id_to_char(raw: dict[Any, str]) -> dict[int, str]:
    if isinstance(next(iter(raw.keys())), str):
        return {int(k): v for k, v in raw.items()}
    return raw


def audio_duration_sec(audio_path: str | Path) -> float:
    path = resolve_path(audio_path)

    # HF files are WAV. Use Python's built-in WAV reader for duration
    # so this diagnostic does not depend on torchaudio.info/load or TorchCodec/FFmpeg.
    with wave.open(str(path), "rb") as f:
        return float(f.getnframes()) / float(f.getframerate())

def split_words(raw_text: str, max_words_per_chunk: int) -> list[str]:
    normalized_with_spaces = normalize_arabic_text(raw_text, keep_spaces=True)
    words = [w for w in normalized_with_spaces.split(" ") if compact(w)]

    if not words:
        label = compact(raw_text)
        if not label:
            return []
        # Fallback: split by character length when spaces are missing.
        max_chars = 12
        return [label[i : i + max_chars] for i in range(0, len(label), max_chars)]

    chunks = []
    for i in range(0, len(words), max_words_per_chunk):
        chunks.append("".join(words[i : i + max_words_per_chunk]))

    return chunks


def build_window_manifest(
    rows: list[dict[str, Any]],
    split: str,
    limit: int,
    max_words_per_chunk: int,
    min_window_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [r for r in rows if r.get("split", "train") == split]
    if not selected:
        selected = rows
    if limit > 0:
        selected = selected[:limit]

    chunk_rows: list[dict[str, Any]] = []
    verse_meta: dict[str, Any] = {}

    for verse_index, row in enumerate(selected):
        raw_text = get_raw_text(row)
        full_label = get_full_label(row)
        if not full_label:
            continue

        audio_path = row.get("audio_path")
        if not audio_path:
            continue

        try:
            dur = float(row.get("audio_duration_sec") or 0.0)
        except (TypeError, ValueError):
            dur = 0.0

        if dur <= 0:
            try:
                dur = audio_duration_sec(audio_path)
            except Exception:
                continue

        verse_id = str(row.get("id") or row.get("sample_id") or f"verse_{verse_index:06d}")
        text_chunks = split_words(raw_text, max_words_per_chunk=max_words_per_chunk)

        if not text_chunks:
            continue

        lengths = [max(1, len(compact(c))) for c in text_chunks]
        total_len = sum(lengths)

        verse_meta[verse_id] = {
            "id": verse_id,
            "audio_path": audio_path,
            "gold": full_label,
            "raw_text": raw_text,
            "duration_sec": dur,
            "num_chunks": len(text_chunks),
            "chunk_ids": [],
        }

        cursor = 0.0
        for chunk_idx, (chunk_text, chunk_len) in enumerate(zip(text_chunks, lengths)):
            start = dur * (cursor / total_len)
            cursor += chunk_len
            end = dur * (cursor / total_len)

            if end - start < min_window_sec:
                center = (start + end) / 2.0
                start = max(0.0, center - min_window_sec / 2.0)
                end = min(dur, center + min_window_sec / 2.0)

            chunk_id = f"{verse_id}_win_{chunk_idx:02d}"
            verse_meta[verse_id]["chunk_ids"].append(chunk_id)

            chunk_rows.append(
                {
                    "id": chunk_id,
                    "parent_id": verse_id,
                    "chunk_index": chunk_idx,
                    "audio_path": audio_path,
                    "normalized_text": compact(chunk_text),
                    "text": compact(chunk_text),
                    "start_sec": start,
                    "end_sec": end,
                    "split": split,
                    "source_kind": "hf_ayah_window",
                }
            )

    return chunk_rows, verse_meta


def greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    id_to_char: dict[int, str],
    blank_penalty: float,
) -> list[str]:
    adjusted = logits.clone()
    adjusted[..., 0] -= blank_penalty

    ids = adjusted.argmax(dim=-1).detach().cpu()
    out: list[str] = []

    for seq, length in zip(ids, input_lengths.detach().cpu().tolist()):
        chars = []
        prev = 0
        for idx in seq[:length].tolist():
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(id_to_char.get(idx, ""))
            prev = idx
        out.append("".join(chars))

    return out


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    if total == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "char_accuracy": 0.0,
            "edit_distance": 0.0,
            "avg_gold_len": 0.0,
            "avg_pred_len": 0.0,
            "avg_chunks": 0.0,
        }

    return {
        "samples": total,
        "exact_match": sum(1 for x in items if compact(x["gold"]) == compact(x["pred"])) / total,
        "char_accuracy": sum(x["char_accuracy"] for x in items) / total,
        "edit_distance": sum(x["edit_distance"] for x in items) / total,
        "avg_gold_len": sum(x["gold_len"] for x in items) / total,
        "avg_pred_len": sum(x["pred_len"] for x in items) / total,
        "avg_chunks": sum(x.get("num_chunks", 0) for x in items) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-words-per-chunk", type=int, default=3)
    parser.add_argument("--min-window-sec", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blank-penalty", type=float, default=0.4)
    parser.add_argument("--work-manifest", default="data/interim/fullverse_as_chunks_windows.jsonl")
    parser.add_argument("--feature-cache-dir", default="data/interim/fullverse_as_chunks_ssl_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    rows = load_jsonl(resolve_path(args.manifest))
    window_rows, verse_meta = build_window_manifest(
        rows=rows,
        split=args.split,
        limit=args.limit,
        max_words_per_chunk=args.max_words_per_chunk,
        min_window_sec=args.min_window_sec,
    )

    work_manifest = resolve_path(args.work_manifest)
    write_jsonl(work_manifest, window_rows)

    ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
    char_to_id = ckpt["char_to_id"]
    id_to_char = to_int_id_to_char(ckpt["id_to_char"])
    hidden_dim = int(ckpt.get("config", {}).get("model", {}).get("hidden_dim", 96))

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = ChunkedContentDataset(
        work_manifest,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=resolve_path(args.feature_cache_dir),
        char_to_id=char_to_id,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_content_batch)

    chunk_predictions: dict[str, str] = {}
    chunk_gold: dict[str, str] = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            ids = batch["ids"]
            texts = batch["texts"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char, args.blank_penalty)

            for chunk_id, gold, pred in zip(ids, texts, preds):
                chunk_predictions[chunk_id] = compact(pred)
                chunk_gold[chunk_id] = compact(gold)

    verse_items = []
    by_bucket = defaultdict(list)

    for verse_id, meta in verse_meta.items():
        pred = "".join(chunk_predictions.get(cid, "") for cid in meta["chunk_ids"])
        gold = meta["gold"]

        item = {
            "id": verse_id,
            "gold": gold,
            "pred": pred,
            "raw_text": meta["raw_text"],
            "bucket": bucket_name(gold),
            "num_chunks": meta["num_chunks"],
            "duration_sec": meta["duration_sec"],
            "gold_len": len(compact(gold)),
            "pred_len": len(compact(pred)),
            "edit_distance": edit_distance(gold, pred),
            "char_accuracy": char_accuracy(gold, pred),
            "chunks": [
                {
                    "id": cid,
                    "gold": chunk_gold.get(cid, ""),
                    "pred": chunk_predictions.get(cid, ""),
                }
                for cid in meta["chunk_ids"]
            ],
        }

        verse_items.append(item)
        by_bucket[item["bucket"]].append(item)

    result = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "manifest": str(resolve_path(args.manifest)),
        "work_manifest": str(work_manifest),
        "split": args.split,
        "limit": args.limit,
        "max_words_per_chunk": args.max_words_per_chunk,
        "min_window_sec": args.min_window_sec,
        "blank_penalty": args.blank_penalty,
        "num_window_rows": len(window_rows),
        "overall": summarize(verse_items),
        "buckets": {name: summarize(items) for name, items in sorted(by_bucket.items())},
        "worst_examples": sorted(verse_items, key=lambda x: x["char_accuracy"])[:50],
        "best_examples": sorted(verse_items, key=lambda x: x["char_accuracy"], reverse=True)[:30],
    }

    out_json = resolve_path(args.output_json)
    out_md = resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Full-verse as chunks diagnostics")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- limit: {args.limit}")
    lines.append(f"- max_words_per_chunk: {args.max_words_per_chunk}")
    lines.append(f"- min_window_sec: {args.min_window_sec}")
    lines.append(f"- blank_penalty: {args.blank_penalty}")
    lines.append(f"- window_rows: {len(window_rows)}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    for k, v in result["overall"].items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| bucket | samples | exact | char_accuracy | edit_distance | avg_gold_len | avg_pred_len | avg_chunks |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in result["buckets"].items():
        lines.append(
            f"| {name} | {m['samples']} | {m['exact_match']:.3f} | {m['char_accuracy']:.3f} | {m['edit_distance']:.3f} | {m['avg_gold_len']:.1f} | {m['avg_pred_len']:.1f} | {m['avg_chunks']:.1f} |"
        )

    lines.append("")
    lines.append("## Worst examples")
    lines.append("")
    for e in result["worst_examples"][:25]:
        lines.append(f"### {e['id']}")
        lines.append(f"- bucket: {e['bucket']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['pred']}`")
        lines.append(f"- char_accuracy: {e['char_accuracy']:.3f}")
        lines.append(f"- lengths gold/pred: {e['gold_len']}/{e['pred_len']}")
        lines.append(f"- chunks: {e['num_chunks']}")
        for ch in e["chunks"][:8]:
            lines.append(f"  - `{ch['gold']}` → `{ch['pred']}`")
        lines.append("")

    lines.append("")
    lines.append("## Best examples")
    lines.append("")
    for e in result["best_examples"][:15]:
        lines.append(f"### {e['id']}")
        lines.append(f"- gold: `{e['gold']}`")
        lines.append(f"- pred: `{e['pred']}`")
        lines.append(f"- char_accuracy: {e['char_accuracy']:.3f}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print("Full-verse as chunks diagnostics")
    print("--------------------------------")
    print("window rows:", len(window_rows))
    print("overall:", result["overall"])
    print("buckets:", result["buckets"])
    print("saved:", out_json)
    print("saved:", out_md)


if __name__ == "__main__":
    main()
