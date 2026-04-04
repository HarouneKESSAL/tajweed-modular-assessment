import json
from pathlib import Path
from typing import Any
import torch

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)

def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
