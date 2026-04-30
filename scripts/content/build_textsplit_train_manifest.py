from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json

from content.train_chunked_content import load_jsonl, normalize_text_target, split_content_indices
from tajweed_assessment.settings import load_yaml


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_text(row: dict) -> str:
    return normalize_text_target(row.get("source_normalized_text") or row.get("normalized_text", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--augmented-manifest", default="data/manifests/retasy_content_chunks_with_subchunks.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_subchunks_textsplit_trainonly.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    base_rows = load_jsonl(PROJECT_ROOT / args.base_manifest)
    train_idx, val_idx = split_content_indices(
        base_rows,
        val_fraction=float(args.val_fraction),
        seed=int(train_cfg["seed"]),
        split_mode="text",
    )
    train_texts = {normalize_text_target(base_rows[idx].get("normalized_text", "")) for idx in train_idx}
    val_texts = {normalize_text_target(base_rows[idx].get("normalized_text", "")) for idx in val_idx}

    augmented_rows = load_jsonl(PROJECT_ROOT / args.augmented_manifest)
    kept_rows = [row for row in augmented_rows if source_text(row) in train_texts]
    leaked_rows = [row for row in kept_rows if source_text(row) in val_texts]
    if leaked_rows:
        raise RuntimeError(f"Leakage detected: {len(leaked_rows)} rows belong to held-out source texts")

    output_path = PROJECT_ROOT / args.output
    write_jsonl(kept_rows, output_path)
    summary = {
        "base_manifest": str(PROJECT_ROOT / args.base_manifest),
        "augmented_manifest": str(PROJECT_ROOT / args.augmented_manifest),
        "output": str(output_path),
        "base_rows": len(base_rows),
        "augmented_rows": len(augmented_rows),
        "kept_rows": len(kept_rows),
        "train_unique_text_count": len(train_texts),
        "val_unique_text_count": len(val_texts),
        "train_texts": sorted(train_texts),
        "held_out_texts": sorted(val_texts),
    }
    print_json(summary)


if __name__ == "__main__":
    main()
