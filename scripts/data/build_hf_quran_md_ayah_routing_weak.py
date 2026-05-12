from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import requests
import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


IKHFA_TRIGGER_LETTERS = set("تثجدذزسشصضطظفقك")
IDGHAM_TRIGGER_LETTERS = set("يرملون")
QALQALAH_LETTERS = set("قطبجد")
MADD_LETTERS = set("اوي")
GHUNNAH_LETTERS = set("نم")


def normalize_arabic(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def compact_text(text: str) -> str:
    return normalize_arabic(text).replace(" ", "")


def safe_name(value: str) -> str:
    value = str(value).replace("\\", "_").replace("/", "_").replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in "_-")


def count_noon_followed_by(text: str, trigger_letters: set[str]) -> int:
    compact = compact_text(text)
    count = 0

    for idx, ch in enumerate(compact[:-1]):
        if ch == "ن" and compact[idx + 1] in trigger_letters:
            count += 1

    return count


def count_word_final_qalqalah(text: str) -> int:
    words = normalize_arabic(text).split()
    return sum(1 for word in words if word and word[-1] in QALQALAH_LETTERS)


def count_any_qalqalah(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in QALQALAH_LETTERS)


def count_madd_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in MADD_LETTERS)


def count_ghunnah_letters(text: str) -> int:
    compact = compact_text(text)
    return sum(1 for ch in compact if ch in GHUNNAH_LETTERS)


def weak_rule_features(text: str) -> dict[str, Any]:
    text = normalize_arabic(text)

    ikhfa_count = count_noon_followed_by(text, IKHFA_TRIGGER_LETTERS)
    idgham_count = count_noon_followed_by(text, IDGHAM_TRIGGER_LETTERS)
    transition_count = ikhfa_count + idgham_count

    qalqalah_any_count = count_any_qalqalah(text)
    qalqalah_final_count = count_word_final_qalqalah(text)

    madd_count = count_madd_letters(text)
    ghunnah_count = count_ghunnah_letters(text)

    # Weak routing labels.
    # These are intentionally broad. They are NOT gold labels.
    use_transition = transition_count > 0
    use_burst = qalqalah_any_count > 0
    use_duration = madd_count > 0 or ghunnah_count > 0

    return {
        "use_duration": int(use_duration),
        "use_transition": int(use_transition),
        "use_burst": int(use_burst),
        "ikhfa_candidate_count": ikhfa_count,
        "idgham_candidate_count": idgham_count,
        "transition_candidate_count": transition_count,
        "qalqalah_any_count": qalqalah_any_count,
        "qalqalah_final_count": qalqalah_final_count,
        "madd_letter_count": madd_count,
        "ghunnah_letter_count": ghunnah_count,
    }


def save_audio_without_torchcodec(audio_obj: object, output_path: Path) -> bool:
    """
    Save HF audio to WAV without using TorchCodec.

    With Audio(decode=False), datasets returns dicts containing either:
    - bytes
    - path
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_sample_field(sample: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in sample and sample[name] not in (None, ""):
            return sample[name]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Buraaq/quran-md-ayahs")
    parser.add_argument("--output-manifest", default="data/manifests/hf_quran_md_ayah_routing_weak_pilot1000.jsonl")
    parser.add_argument("--audio-dir", default="data/raw/hf_quran_md_ayah_routing_weak_pilot1000")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-per-reciter", type=int, default=80)
    parser.add_argument("--max-per-ayah", type=int, default=0, help="Maximum saved recordings per ayah. 0 disables this cap.")
    parser.add_argument("--sample-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument(
        "--require-any-route",
        action="store_true",
        help="Keep only ayahs with at least one weak routing target.",
    )
    args = parser.parse_args()

    output_manifest = PROJECT_ROOT / args.output_manifest
    audio_dir = PROJECT_ROOT / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("Loading HF dataset in streaming mode...")
    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    # Avoid TorchCodec decoding.
    ds = ds.cast_column("audio", Audio(decode=False))

    rows: list[dict[str, Any]] = []
    per_reciter: dict[str, int] = {}
    per_ayah: dict[str, int] = {}

    for source_index, sample in enumerate(ds):
        ayah_ar = str(get_sample_field(sample, ["ayah_ar", "text", "arabic", "source_text"], ""))
        text_norm = normalize_arabic(ayah_ar)

        if not text_norm:
            continue

        reciter_id = str(get_sample_field(sample, ["reciter_id", "reciter"], "unknown"))
        if per_reciter.get(reciter_id, 0) >= args.max_per_reciter:
            continue

        weak = weak_rule_features(text_norm)

        if args.require_any_route:
            if not (weak["use_duration"] or weak["use_transition"] or weak["use_burst"]):
                continue

        audio = sample.get("audio")
        if not isinstance(audio, dict):
            continue

        surah_id = int(get_sample_field(sample, ["surah_id", "surah_number", "quranjson_surah_number"], 0))
        ayah_id = int(get_sample_field(sample, ["ayah_id", "ayah_number", "quranjson_verse_index"], 0))
        ayah_key = f"{surah_id:03d}:{ayah_id:03d}"

        if args.max_per_ayah and per_ayah.get(ayah_key, 0) >= args.max_per_ayah:
            continue

        sample_id = f"hf_quran_md_ayah_route_{reciter_id}_{surah_id:03d}_{ayah_id:03d}_{source_index:06d}"
        audio_path = audio_dir / f"{safe_name(sample_id)}.wav"

        if not save_audio_without_torchcodec(audio, audio_path):
            continue

        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "source_dataset": args.dataset_name,
            "source_index": source_index,
            "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
            "text": text_norm,
            "source_text": ayah_ar,
            "surah_name": get_sample_field(sample, ["surah_name_en", "surah_name"], ""),
            "surah_name_ar": get_sample_field(sample, ["surah_name_ar"], ""),
            "quranjson_surah_number": surah_id,
            "quranjson_verse_key": f"verse_{ayah_id}",
            "ayah_key": ayah_key,
            "reciter_id": reciter_id,
            "reciter_name": get_sample_field(sample, ["reciter_name"], ""),
            "target_names": ["use_duration", "use_transition", "use_burst"],
            "targets": [
                int(weak["use_duration"]),
                int(weak["use_transition"]),
                int(weak["use_burst"]),
            ],
            "weak_rule_features": weak,
            "label_source": "weak_hf_quran_md_text_pattern",
            "sample_weight": float(args.sample_weight),
            "content_source": "hf_quran_md_ayahs",
        }

        rows.append(row)
        per_reciter[reciter_id] = per_reciter.get(reciter_id, 0) + 1
        per_ayah[ayah_key] = per_ayah.get(ayah_key, 0) + 1

        if len(rows) % 100 == 0:
            print(f"collected {len(rows)} samples...")

        if len(rows) >= args.max_samples:
            break

    write_jsonl(output_manifest, rows)

    target_counts = Counter()
    combo_counts = Counter()
    reciter_counts = Counter()

    for row in rows:
        targets = row["targets"]
        names = row["target_names"]

        active = []
        for name, value in zip(names, targets):
            if value:
                target_counts[name] += 1
                active.append(name)

        combo_counts["+".join(active) if active else "none"] += 1
        reciter_counts[row["reciter_id"]] += 1

    print("HF weak routing pilot built")
    print("---------------------------")
    print(f"samples      : {len(rows)}")
    print(f"target_counts: {dict(target_counts)}")
    print(f"combo_counts : {dict(combo_counts)}")
    print(f"reciters     : {len(reciter_counts)}")
    print(f"unique_ayahs : {len(set(row.get('ayah_key', '') for row in rows))}")
    print(f"unique_texts : {len(set(row.get('text', '') for row in rows))}")
    print(f"top_reciters : {reciter_counts.most_common(10)}")
    print(f"manifest     : {output_manifest}")
    print(f"audio_dir    : {audio_dir}")


if __name__ == "__main__":
    main()