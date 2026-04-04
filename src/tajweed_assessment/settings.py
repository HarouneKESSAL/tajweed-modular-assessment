from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def manifests(self) -> Path:
        return self.data / "manifests"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data
