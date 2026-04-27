from pathlib import Path
import sys
import soundfile as sf
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json

import torch
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_spans(spans) -> float:
    total_len = sum(len(s) for s in spans)
    if total_len == 0:
        return 0.0
    return float(sum(s.score * len(s) for s in spans) / total_len)

def load_audio_mono(audio_path: str, target_sr: int):
    data, sr = sf.read(audio_path, always_2d=True)
    waveform = torch.tensor(data.T, dtype=torch.float32)  # [C, T]

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr


def align_one(model, tokenizer, aligner, row, device: str):
    waveform, sr = load_audio_mono(row["audio_path"], bundle.sample_rate)

    transcript_words = row["romanized_words"]
    if not transcript_words:
        raise RuntimeError(f"{row['id']} has empty romanized_words")

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript_words))

    # Tutorial uses this ratio for mapping frame indices to seconds.
    ratio = waveform.size(1) / emission.size(1) / bundle.sample_rate

    word_alignments = []
    flat_char_alignments = []
    global_char_idx = 0

    for word_idx, (word, spans) in enumerate(zip(transcript_words, token_spans)):
        if not spans:
            continue

        word_start_frame = int(spans[0].start)
        word_end_frame = int(spans[-1].end)

        word_alignments.append(
            {
                "word_index": word_idx,
                "word": word,
                "start_frame": word_start_frame,
                "end_frame": word_end_frame,
                "start_sec": float(word_start_frame * ratio),
                "end_sec": float(word_end_frame * ratio),
                "score": score_spans(spans),
            }
        )

        for char_idx_in_word, (ch, span) in enumerate(zip(word, spans)):
            flat_char_alignments.append(
                {
                    "global_char_index": global_char_idx,
                    "word_index": word_idx,
                    "char_index_in_word": char_idx_in_word,
                    "char": ch,
                    "start_frame": int(span.start),
                    "end_frame": int(span.end),
                    "start_sec": float(span.start * ratio),
                    "end_sec": float(span.end * ratio),
                    "score": float(span.score),
                }
            )
            global_char_idx += 1

    return {
        "id": row["id"],
        "audio_path": row["audio_path"],
        "surah_name": row.get("surah_name"),
        "quranjson_verse_key": row.get("quranjson_verse_key"),
        "quranjson_verse_index": row.get("quranjson_verse_index"),

        "normalized_text": row["normalized_text"],
        "romanized_text": row["romanized_text"],
        "romanized_words": transcript_words,

        "gold_duration_labels": row.get("gold_duration_labels", []),
        "projected_duration_labels": row.get("projected_duration_labels", []),
        "duration_rule_spans_normalized": row.get("duration_rule_spans_normalized", []),

        "sample_rate": int(sr),
        "num_emission_frames": int(emission.size(1)),
        "word_alignments": word_alignments,
        "char_alignments_romanized": flat_char_alignments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_alignment_corpus_torchaudio.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/alignment/torchaudio_forced_alignment_preview.jsonl",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="If >= 0, align only one row by index",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="When --index is not used, align the first N rows",
    )
    parser.add_argument(
        "--device",
        default="",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)
    if args.index >= 0:
        rows = [rows[args.index]]
    else:
        rows = rows[: args.max_rows]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    out_rows = []
    for i, row in enumerate(rows, start=1):
        try:
            result = align_one(model, tokenizer, aligner, row, device)
            out_rows.append(result)
            print(
                f"[{i}/{len(rows)}] aligned {row['id']} "
                f"| words={len(result['word_alignments'])} "
                f"| chars={len(result['char_alignments_romanized'])}"
            )
        except Exception as e:
            print(f"[{i}/{len(rows)}] failed {row.get('id', '<unknown>')}: {e}")

    write_jsonl(output_path, out_rows)

    print(f"\nOutput: {output_path}")
    print(f"Rows aligned: {len(out_rows)}")


if __name__ == "__main__":
    main()
