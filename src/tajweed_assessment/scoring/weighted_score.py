from pathlib import Path
from typing import Any

import yaml

from tajweed_assessment.scoring.error_types import TajweedError


def load_error_weights(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("error weight config must be a mapping")
    return config


def get_error_weight(config: dict[str, Any], error: TajweedError) -> float:
    category = config.get("categories", {}).get(error.module, {})
    item = category.get(error.error_type)
    if item is None:
        return 1.0
    return float(item.get("weight", 1.0))


def weighted_error_sum(errors: list[TajweedError], config: dict[str, Any]) -> float:
    total = 0.0
    for error in errors:
        confidence = max(0.0, min(1.0, float(error.confidence)))
        total += get_error_weight(config, error) * confidence
    return total


def final_score(errors: list[TajweedError], config: dict[str, Any]) -> float:
    scale = float(config.get("scale", 3.0))
    penalty = weighted_error_sum(errors, config) * scale
    return max(0.0, 100.0 - penalty)