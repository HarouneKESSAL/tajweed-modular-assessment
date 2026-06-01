from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-dir", required=True, type=Path)
    parser.add_argument("--surah", required=True, type=int)
    parser.add_argument("--ayah-start", required=True, type=int)
    parser.add_argument("--ayah-end", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    wavs = sorted(args.segment_dir.glob("*.wav"))
    expected_count = args.ayah_end - args.ayah_start + 1

    if len(wavs) != expected_count:
        raise SystemExit(
            f"Expected {expected_count} wav files for ayahs "
            f"{args.ayah_start}-{args.ayah_end}, but found {len(wavs)} in {args.segment_dir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"

    with args.output.open(mode, encoding="utf-8") as f:
        for offset, wav in enumerate(wavs):
            row = {
                "audio_path": str(wav).replace("\\", "/"),
                "surah": args.surah,
                "ayah": args.ayah_start + offset,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(wavs)} labeled rows to {args.output}")


if __name__ == "__main__":
    main()