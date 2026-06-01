from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_wavs(segment_dir: Path) -> list[Path]:
    candidates = [
        segment_dir / "natural",
        segment_dir / "expected_count",
        segment_dir / "free_detect",
        segment_dir,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            wavs = sorted(candidate.glob("*.wav"))
            if wavs:
                return wavs

    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/uploads/segments"))
    parser.add_argument("--surah", required=True, type=int)
    parser.add_argument("--ayah-start", required=True, type=int)
    parser.add_argument("--ayah-end", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    expected_count = args.ayah_end - args.ayah_start + 1

    rows = []
    skipped = []

    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir():
            continue

        wavs = find_wavs(folder)

        if len(wavs) != expected_count:
            skipped.append((folder, len(wavs)))
            continue

        for offset, wav in enumerate(wavs):
            rows.append(
                {
                    "audio_path": str(wav).replace("\\", "/"),
                    "surah": args.surah,
                    "ayah": args.ayah_start + offset,
                    "source_folder": str(folder).replace("\\", "/"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    with args.output.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} labeled rows to {args.output}")
    print(f"Used {len(rows) // expected_count if expected_count else 0} folders")
    print(f"Skipped {len(skipped)} folders")

    if skipped:
        print("\nSkipped folders because segment count did not match:")
        for folder, count in skipped[:30]:
            print(f"  {folder} -> {count} wav files")


if __name__ == "__main__":
    main()