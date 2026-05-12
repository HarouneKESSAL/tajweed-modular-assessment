from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.inference.learned_routing import load_learned_routing_predictor, load_learned_routing_predictor_from_config


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--threshold-config", default="configs/learned_router_thresholds.yaml")
    parser.add_argument("--threshold-profile", default="balanced_safe")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = load_jsonl(resolve_path(args.manifest))
    row = rows[args.sample_index]

    if args.checkpoint:
        predictor = load_learned_routing_predictor(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    else:
        predictor = load_learned_routing_predictor_from_config(
            config_path=args.threshold_config,
            profile=args.threshold_profile,
            device=args.device,
        )

    text = row.get("text") or row.get("normalized_text") or row.get("source_text") or ""
    result = predictor.predict(
        audio_path=row["audio_path"],
        text=text,
    )

    print("Sample ID:", row.get("id") or row.get("sample_id"))
    print("Text     :", text)
    print("Audio    :", row.get("audio_path"))
    print()
    print("Learned routing prediction:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
