from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


HARD_RECITER_EXTRA_REPEATS = {
    "husary_mujawwad": 4,
    "minshawy_mujawwad": 4,
    "muhsin_al_qasim": 3,
    "warsh_husary": 2,
    "hussary.teacher": 2,
    "banna": 1,
    "warsh_yassin": 1,
}


def normalize_arabic(text: str) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_arabic(text))


def reciter_from_id(sample_id: str) -> str:
    m = re.match(r"^hf_quran_md_ayah_route_(.+?)_\d{3}_\d{3}_", str(sample_id))
    return m.group(1) if m else "unknown"


def get_text(row: dict[str, Any]) -> str:
    for key in ["text", "normalized_text", "target_text", "transcript", "sentence"]:
        if row.get(key):
            return str(row[key])
    return ""


def get_id(row: dict[str, Any], index: int) -> str:
    for key in ["id", "sample_id", "audio_id"]:
        if row.get(key):
            return str(row[key])
    return f"row_{index:08d}"


def extra_repeats_for_row(row: dict[str, Any], index: int, *, train_split: str) -> int:
    split = str(row.get("split", train_split))
    if split != train_split:
        return 0

    sample_id = get_id(row, index)
    reciter = reciter_from_id(sample_id)
    text = get_text(row)
    clen = len(compact(text))

    extra = 0

    # Main v2 target: hard reciters from v1 error report.
    extra += HARD_RECITER_EXTRA_REPEATS.get(reciter, 0)

    # Secondary target: longer ayahs had lower exact rate.
    if 41 <= clen <= 60:
        extra += 2
    elif clen > 60:
        extra += 3
    elif 11 <= clen <= 20:
        extra += 1

    # Avoid exploding the dataset too much.
    return min(extra, 6)


def make_augmented_copy(row: dict[str, Any], base_id: str, repeat_index: int) -> dict[str, Any]:
    out = dict(row)
    aug_id = f"{base_id}__v2aug_{repeat_index:02d}"

    if "id" in out:
        out["id"] = aug_id
    if "sample_id" in out:
        out["sample_id"] = aug_id
    if "audio_id" in out:
        out["audio_id"] = aug_id

    out["augmentation_source_id"] = base_id
    out["augmentation_kind"] = "v2_hard_reciter_length_oversample"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--train-split", default="train")
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    output_rows = []
    original_train = 0
    original_val = 0
    added = 0

    repeat_counter = Counter()
    reciter_counter = Counter()
    length_counter = Counter()

    for i, row in enumerate(rows):
        split = str(row.get("split", args.train_split))
        if split == args.train_split:
            original_train += 1
        else:
            original_val += 1

        output_rows.append(row)

        base_id = get_id(row, i)
        reciter = reciter_from_id(base_id)
        text_len = len(compact(get_text(row)))

        extra = extra_repeats_for_row(row, i, train_split=args.train_split)
        repeat_counter[extra] += 1

        if split == args.train_split and extra > 0:
            reciter_counter[reciter] += extra
            if text_len <= 10:
                length_counter["001_010"] += extra
            elif text_len <= 20:
                length_counter["011_020"] += extra
            elif text_len <= 40:
                length_counter["021_040"] += extra
            elif text_len <= 60:
                length_counter["041_060"] += extra
            else:
                length_counter["061_plus"] += extra

        for r in range(extra):
            output_rows.append(make_augmented_copy(row, base_id, r + 1))
            added += 1

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in output_rows) + "\n",
        encoding="utf-8",
    )

    summary = {
        "input_manifest": str(input_path),
        "output_manifest": str(out_manifest),
        "original_rows": len(rows),
        "original_train_rows": original_train,
        "original_non_train_rows": original_val,
        "added_train_duplicates": added,
        "output_rows": len(output_rows),
        "repeat_distribution": dict(sorted(repeat_counter.items())),
        "added_by_reciter": dict(reciter_counter.most_common()),
        "added_by_length_bucket": dict(sorted(length_counter.items())),
        "hard_reciter_extra_repeats": HARD_RECITER_EXTRA_REPEATS,
    }

    out_json = Path(args.output_summary_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Whisper Quran ASR v2 weighted manifest")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for k in [
        "original_rows",
        "original_train_rows",
        "original_non_train_rows",
        "added_train_duplicates",
        "output_rows",
    ]:
        lines.append(f"| {k} | {summary[k]} |")

    lines.append("")
    lines.append("## Added duplicates by reciter")
    lines.append("")
    lines.append("| reciter | added rows |")
    lines.append("|---|---:|")
    for k, v in summary["added_by_reciter"].items():
        lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## Added duplicates by length bucket")
    lines.append("")
    lines.append("| bucket | added rows |")
    lines.append("|---|---:|")
    for k, v in summary["added_by_length_bucket"].items():
        lines.append(f"| {k} | {v} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
