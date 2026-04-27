from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
import json

from tajweed_assessment.utils.io import load_json, save_json

@dataclass
class ManifestEntry:
    sample_id: str
    audio_path: str
    feature_path: str = ""
    canonical_phonemes: list[str] | None = None
    canonical_rules: list[str] | None = None
    text: str = ""
    reciter_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

def save_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    save_json([entry.to_dict() for entry in entries], path)

def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _normalize_manifest_item(item: dict) -> dict:
    canonical_rules = (
        item.get("canonical_rules")
        or item.get("gold_duration_labels")
        or item.get("projected_duration_labels")
        or []
    )

    return {
        "sample_id": item.get("sample_id") or item.get("id") or item.get("audio_path") or "sample",
        "audio_path": item.get("audio_path", ""),
        "feature_path": item.get("feature_path", ""),
        "canonical_phonemes": item.get("canonical_phonemes"),
        "canonical_rules": canonical_rules,
        "text": item.get("text") or item.get("aya_text_norm") or item.get("normalized_text") or item.get("original_text") or "",
        "reciter_id": item.get("reciter_id"),
    }

def load_manifest(path: str | Path) -> List[ManifestEntry]:
    path = Path(path)
    items = _load_jsonl(path) if path.suffix.lower() == ".jsonl" else load_json(path)
    return [ManifestEntry(**_normalize_manifest_item(item)) for item in items]
