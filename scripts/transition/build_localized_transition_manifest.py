from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import math
from collections import Counter

from tajweed_assessment.data.labels import normalize_rule_name
from tajweed_assessment.utils.io import save_json


TARGET_RULES = {"ikhfa", "idgham"}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def map_source_span_to_norm_indices(
    *,
    start: int,
    end: int,
    num_chars: int,
    text_length: int,
    source_max_end: int,
) -> tuple[int, int]:
    if num_chars <= 0:
        return 0, 0

    if 0 <= start < end <= text_length and text_length > 0:
        norm_start = start
        norm_end = end
    else:
        source_len = max(source_max_end, text_length, 1)
        norm_start = int(math.floor(start * num_chars / source_len))
        norm_end = int(math.ceil(end * num_chars / source_len))

    norm_start = max(0, min(norm_start, num_chars))
    norm_end = max(norm_start + 1, min(norm_end, num_chars))
    norm_end = min(norm_end, num_chars)
    return norm_start, norm_end


def project_transition_spans(transition_row: dict, alignment_row: dict) -> list[dict]:
    char_spans = alignment_row.get("arabic_char_time_spans", [])
    text = transition_row.get("text") or alignment_row.get("normalized_text") or ""
    text_length = len(str(text))
    num_chars = len(char_spans)
    source_max_end = max([int(span.get("end", 0)) for span in transition_row.get("rule_spans", [])] + [text_length, 1])

    projected = []
    for span in transition_row.get("rule_spans", []):
        coarse_rule = normalize_rule_name(span.get("rule", ""))
        if coarse_rule not in TARGET_RULES:
            continue

        start = int(span.get("start", 0))
        end = int(span.get("end", start + 1))
        norm_start, norm_end = map_source_span_to_norm_indices(
            start=start,
            end=end,
            num_chars=num_chars,
            text_length=text_length,
            source_max_end=source_max_end,
        )

        covered = [
            item
            for item in char_spans[norm_start:norm_end]
            if item.get("char") != " "
        ]
        timed = [item for item in covered if item.get("start_sec") is not None and item.get("end_sec") is not None]
        if not timed:
            continue

        start_sec = min(float(item["start_sec"]) for item in timed)
        end_sec = max(float(item["end_sec"]) for item in timed)
        text_piece = "".join(str(item.get("char", "")) for item in covered)
        projected.append(
            {
                "rule": coarse_rule,
                "norm_start": norm_start,
                "norm_end": norm_end,
                "text": text_piece,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "mapped_nonspace_chars": len(timed),
                "total_nonspace_chars": len(covered),
                "fully_timed": len(timed) == len(covered),
            }
        )

    return projected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-manifest", default="data/manifests/retasy_transition_subset.jsonl")
    parser.add_argument("--alignment-manifest", default="data/alignment/duration_time_projection_strict.jsonl")
    parser.add_argument("--output", default="data/alignment/transition_time_projection_strict.jsonl")
    args = parser.parse_args()

    transition_rows = load_jsonl(PROJECT_ROOT / args.transition_manifest)
    alignment_rows = load_jsonl(PROJECT_ROOT / args.alignment_manifest)
    alignment_by_id = {row.get("id"): row for row in alignment_rows}

    output_rows = []
    skipped_missing_alignment = 0
    label_counts = Counter()
    rows_with_any = 0

    for row in transition_rows:
        sample_id = row.get("sample_id") or row.get("id")
        alignment_row = alignment_by_id.get(sample_id)
        if alignment_row is None:
            skipped_missing_alignment += 1
            continue

        projected_spans = project_transition_spans(row, alignment_row)
        if projected_spans:
            rows_with_any += 1
            for span in projected_spans:
                label_counts[span["rule"]] += 1

        output_rows.append(
            {
                "id": sample_id,
                "audio_path": row.get("audio_path"),
                "surah_name": row.get("surah_name"),
                "quranjson_verse_key": row.get("quranjson_verse_key"),
                "normalized_text": alignment_row.get("normalized_text") or row.get("text", ""),
                "transition_label": normalize_rule_name((row.get("canonical_rules") or ["none"])[0]),
                "transition_rules": row.get("transition_rules", []),
                "transition_rule_time_spans": projected_spans,
                "alignment_stats": alignment_row.get("alignment_stats", {}),
                "reciter_id": row.get("reciter_id"),
            }
        )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input transition manifest : {PROJECT_ROOT / args.transition_manifest}")
    print(f"Input alignment manifest  : {PROJECT_ROOT / args.alignment_manifest}")
    print(f"Output manifest           : {output_path}")
    print(f"Rows written              : {len(output_rows)}")
    print(f"Rows with any transition  : {rows_with_any}")
    print(f"Skipped missing alignment : {skipped_missing_alignment}")
    print(f"Label counts              : {dict(label_counts)}")


if __name__ == "__main__":
    main()

