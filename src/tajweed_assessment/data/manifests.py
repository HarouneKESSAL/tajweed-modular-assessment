from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

from tajweed_assessment.utils.io import load_json, save_json

@dataclass
class ManifestEntry:
    sample_id: str
    audio_path: str
    feature_path: str = ""
    canonical_phonemes: list[str] | None = None
    canonical_rules: list[str] | None = None
    text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

def save_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    save_json([entry.to_dict() for entry in entries], path)

def load_manifest(path: str | Path) -> List[ManifestEntry]:
    return [ManifestEntry(**item) for item in load_json(path)]
