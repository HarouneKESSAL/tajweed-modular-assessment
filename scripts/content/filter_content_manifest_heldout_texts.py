from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
from collections import Counter

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


def row_texts(row: dict) -> tuple[str, str]:
    chunk_text = normalize_text_target(row.get("normalized_text", ""))
    source_text = normalize_text_target(row.get("source_normalized_text") or row.get("aya_text_norm") or row.get("text") or "")
    return chunk_text, source_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--input", default="data/manifests/retasy_content_weak_chunks.jsonl")
    parser.add_argument("--output", default="data/manifests/retasy_content_weak_chunks_no_textsplit_val.jsonl")
    parser.add_argument(
        "--exclude-mode",
        choices=["chunk", "source_contains"],
        default="source_contains",
        help="chunk excludes exact held-out chunks; source_contains excludes rows whose source verse contains a held-out chunk.",
    )
    args = parser.parse_args()

    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    base_rows = load_jsonl(PROJECT_ROOT / args.base_manifest)
    _, val_idx = split_content_indices(
        base_rows,
        val_fraction=0.2,
        seed=int(train_cfg["seed"]),
        split_mode="text",
    )
    heldout_texts = {normalize_text_target(base_rows[idx].get("normalized_text", "")) for idx in val_idx}
    rows = load_jsonl(PROJECT_ROOT / args.input)

    kept_rows: list[dict] = []
    skipped = Counter()
    for row in rows:
        chunk_text, source_text = row_texts(row)
        if args.exclude_mode == "chunk":
            should_skip = chunk_text in heldout_texts
        else:
            should_skip = any(text and text in source_text for text in heldout_texts)
        if should_skip:
            skipped["heldout_text_overlap"] += 1
            continue
        kept_rows.append(row)

    output_path = PROJECT_ROOT / args.output
    write_jsonl(kept_rows, output_path)
    summary = {
        "input": str(PROJECT_ROOT / args.input),
        "output": str(output_path),
        "exclude_mode": args.exclude_mode,
        "heldout_texts": sorted(heldout_texts),
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "skipped": dict(skipped),
        "unique_kept_chunk_texts": len({row_texts(row)[0] for row in kept_rows}),
        "unique_kept_source_texts": len({row_texts(row)[1] for row in kept_rows}),
    }
    print_json(summary)


if __name__ == "__main__":
    main()
