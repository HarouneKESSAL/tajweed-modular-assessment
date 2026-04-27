from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import json
import re

from datasets import Audio, load_dataset


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")
_NON_ARABIC_TEXT_RE = re.compile(r"[^\u0621-\u063A\u0641-\u064A\s]")


def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = text.replace("ـ", "")
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = (
        text.replace("ٱ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    text = _NON_ARABIC_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = _NON_SLUG_RE.sub("_", text)
    return text.strip("_")


def _parse_annotation_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"raw": value}
    return {"raw": value}


@dataclass
class RetasyRow:
    id: str
    source_dataset: str
    split: str
    hf_index: int

    surah_name: Optional[str]
    aya_text: str
    aya_text_norm: str

    duration_ms: Optional[int]
    golden: Optional[bool]
    final_label: Optional[str]

    reciter_id: Optional[str]
    reciter_country: Optional[str]
    reciter_gender: Optional[str]
    reciter_age: Optional[str]
    reciter_qiraah: Optional[str]

    judgments_num: Optional[int]
    annotation_metadata: Dict[str, Any]

    audio_path: Optional[str]
    original_audio_path: Optional[str]
    sample_rate: Optional[int]
    num_samples: Optional[int]


def iter_retasy_rows(split: str = "train") -> Iterator[Dict[str, Any]]:
    ds = load_dataset("RetaSy/quranic_audio_dataset", split=split)
    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    for row in ds:
        yield row


def _audio_info(audio_obj: Any) -> tuple[Optional[str], Optional[int], Optional[int], Optional[bytes]]:
    if not isinstance(audio_obj, dict):
        return None, None, None, None
    original_path = audio_obj.get("path")
    raw_bytes = audio_obj.get("bytes")
    return original_path, None, None, raw_bytes


def _infer_suffix(original_audio_path: Optional[str]) -> str:
    if original_audio_path:
        suffix = Path(original_audio_path).suffix
        if suffix:
            return suffix
    return ".wav"


def _save_audio_bytes(raw_bytes: Optional[bytes], out_path: Path) -> Optional[str]:
    if raw_bytes is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)
    return str(out_path)


def retasy_row_to_record(
    row: Dict[str, Any],
    *,
    hf_index: int,
    split: str,
    save_audio_dir: Optional[Path] = None,
) -> RetasyRow:
    surah_name = row.get("Surah")
    aya_text = row.get("Aya") or ""
    aya_text_norm = normalize_arabic_text(aya_text)

    original_audio_path, sample_rate, num_samples, raw_bytes = _audio_info(row.get("audio"))

    audio_path = None
    if save_audio_dir is not None:
        stem = f"{hf_index:06d}_{slugify(str(surah_name or 'unknown'))}"
        suffix = _infer_suffix(original_audio_path)
        audio_path = _save_audio_bytes(raw_bytes, save_audio_dir / f"{stem}{suffix}")

    return RetasyRow(
        id=f"retasy_{split}_{hf_index:06d}",
        source_dataset="RetaSy/quranic_audio_dataset",
        split=split,
        hf_index=hf_index,
        surah_name=surah_name,
        aya_text=aya_text,
        aya_text_norm=aya_text_norm,
        duration_ms=row.get("duration_ms"),
        golden=row.get("golden"),
        final_label=row.get("final_label"),
        reciter_id=row.get("reciter_id"),
        reciter_country=row.get("reciter_country"),
        reciter_gender=row.get("reciter_gender"),
        reciter_age=row.get("reciter_age"),
        reciter_qiraah=row.get("reciter_qiraah"),
        judgments_num=row.get("judgments_num"),
        annotation_metadata=_parse_annotation_metadata(row.get("annotation_metadata")),
        audio_path=audio_path,
        original_audio_path=original_audio_path,
        sample_rate=sample_rate,
        num_samples=num_samples,
    )


def build_retasy_manifest(
    out_jsonl: str | Path,
    *,
    split: str = "train",
    save_audio_dir: Optional[str | Path] = None,
) -> Path:
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_dir = Path(save_audio_dir) if save_audio_dir else None

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(iter_retasy_rows(split=split)):
            record = retasy_row_to_record(
                row,
                hf_index=i,
                split=split,
                save_audio_dir=audio_dir,
            )
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    build_retasy_manifest(
        "data/manifests/retasy_train.jsonl",
        split="train",
        save_audio_dir="data/raw/retasy_train_audio",
    )