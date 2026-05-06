from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import requests
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


IKHFA_TRIGGER_LETTERS = set("تثجدذزسشصضطظفقك")
IDGHAM_TRIGGER_LETTERS = set("يرملون")


def normalize_arabic(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def has_noon_pattern(text: str, triggers: set[str]) -> bool:
    compact = text.replace(" ", "")
    for idx, ch in enumerate(compact[:-1]):
        if ch == "ن" and compact[idx + 1] in triggers:
            return True
    return False


def transition_labels_from_text(text: str) -> list[str]:
    text_norm = normalize_arabic(text)
    labels = []

    if has_noon_pattern(text_norm, IKHFA_TRIGGER_LETTERS):
        labels.append("ikhfa")
    if has_noon_pattern(text_norm, IDGHAM_TRIGGER_LETTERS):
        labels.append("idgham")

    return labels


def multihot(labels: list[str]) -> list[float]:
    return [
        1.0 if "ikhfa" in labels else 0.0,
        1.0 if "idgham" in labels else 0.0,
    ]


def combo(labels: list[str]) -> str:
    if not labels:
        return "none"
    return "+".join(labels)


def safe_name(value: str) -> str:
    value = str(value).replace("\\", "_").replace("/", "_").replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in "_-")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def save_audio_without_torchcodec(audio_obj: object, output_path: Path) -> bool:
    """
    Save HF audio to WAV without using TorchCodec.

    With Audio(decode=False), datasets returns dicts containing either:
    - bytes
    - path

    We decode using soundfile instead.
    """
    if not isinstance(audio_obj, dict):
        return False

    audio_bytes = audio_obj.get("bytes")
    audio_path = audio_obj.get("path")

    try:
        if audio_bytes is not None:
            data, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            sf.write(str(output_path), data, int(sample_rate))
            return True

        if audio_path:
            audio_path = str(audio_path)

            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                response = requests.get(audio_path, timeout=60)
                response.raise_for_status()
                data, sample_rate = sf.read(io.BytesIO(response.content), always_2d=False)
                sf.write(str(output_path), data, int(sample_rate))
                return True

            local_path = Path(audio_path)
            if local_path.exists():
                data, sample_rate = sf.read(str(local_path), always_2d=False)
                sf.write(str(output_path), data, int(sample_rate))
                return True

    except Exception as exc:
        print(f"[warn] failed to save audio {output_path}: {exc}")
        return False

    return False

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Buraaq/quran-md-ayahs")
    parser.add_argument("--output-manifest", default="data/manifests/hf_quran_md_ayah_transition_pilot.jsonl")
    parser.add_argument("--audio-dir", default="data/raw/hf_quran_md_ayah_transition_pilot")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-per-reciter", type=int, default=80)
    parser.add_argument("--require-both", action="store_true")
    parser.add_argument("--sample-weight", type=float, default=0.25)
    args = parser.parse_args()

    output_manifest = PROJECT_ROOT / args.output_manifest
    audio_dir = PROJECT_ROOT / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HF dataset in streaming mode...")
    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    # Avoid Hugging Face automatic audio decoding, which uses TorchCodec.
    # We only need raw audio bytes/path and will decode/save with soundfile.
    ds = ds.cast_column("audio", Audio(decode=False))

    rows: list[dict[str, Any]] = []
    per_reciter: dict[str, int] = {}

    for sample in ds:
        ayah_ar = sample.get("ayah_ar") or ""
        text_norm = normalize_arabic(ayah_ar)
        labels = transition_labels_from_text(text_norm)

        if args.require_both and set(labels) != {"ikhfa", "idgham"}:
            continue

        if not labels:
            continue

        reciter_id = str(sample.get("reciter_id") or "unknown")
        if per_reciter.get(reciter_id, 0) >= args.max_per_reciter:
            continue

        audio = sample.get("audio")
        if not isinstance(audio, dict):
            continue

        surah_id = int(sample.get("surah_id"))
        ayah_id = int(sample.get("ayah_id"))
        sample_id = f"hf_quran_md_ayah_{reciter_id}_{surah_id:03d}_{ayah_id:03d}"
        audio_path = audio_dir / f"{safe_name(sample_id)}.wav"

        if not save_audio_without_torchcodec(audio, audio_path):
            continue

        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
            "text": text_norm,
            "source_text": ayah_ar,
            "surah_name": sample.get("surah_name_en", ""),
            "surah_name_ar": sample.get("surah_name_ar", ""),
            "quranjson_surah_number": surah_id,
            "quranjson_verse_key": f"verse_{ayah_id}",
            "reciter_id": reciter_id,
            "reciter_name": sample.get("reciter_name", ""),
            "transition_label_names": ["ikhfa", "idgham"],
            "transition_multilabel_rules": labels,
            "transition_multihot": multihot(labels),
            "transition_combo": combo(labels),
            "label_source": "weak_hf_quran_md_text_pattern",
            "sample_weight": float(args.sample_weight),
            "content_source": "hf_quran_md_ayahs",
        }

        rows.append(row)
        per_reciter[reciter_id] = per_reciter.get(reciter_id, 0) + 1

        if len(rows) >= args.max_samples:
            break

        if len(rows) % 100 == 0:
            print(f"collected {len(rows)} samples...")

    write_jsonl(output_manifest, rows)

    combo_counts: dict[str, int] = {}
    for row in rows:
        key = row["transition_combo"]
        combo_counts[key] = combo_counts.get(key, 0) + 1

    print("HF Quran-MD ayah pilot built")
    print("----------------------------")
    print(f"samples      : {len(rows)}")
    print(f"combo_counts : {combo_counts}")
    print(f"reciters     : {len(per_reciter)}")
    print(f"manifest     : {output_manifest}")
    print(f"audio_dir    : {audio_dir}")


if __name__ == "__main__":
    main()