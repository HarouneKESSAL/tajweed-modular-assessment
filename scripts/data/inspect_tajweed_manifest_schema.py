from __future__ import annotations

import json
from pathlib import Path


paths = [
    Path("data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl"),
]

for path in paths:
    print()
    print("=" * 80)
    print(path)
    print("=" * 80)

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= 3:
            break

    for i, row in enumerate(rows):
        print()
        print(f"ROW {i}")
        print("keys:", list(row.keys()))
        print(json.dumps(row, ensure_ascii=False, indent=2)[:5000])
