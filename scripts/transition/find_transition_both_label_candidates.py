from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


MANIFESTS = [
    Path("data/manifests/retasy_train.jsonl"),
    Path("data/manifests/retasy_quranjson_train.jsonl"),
    Path("data/manifests/hf_quran_md_ayahs_unique48_r2.jsonl"),
]


IKHFA_TRIGGER_LETTERS = set("تثجدذزسشصضطظفقك")
IDGHAM_TRIGGER_LETTERS = set("يرملون")


def norm(text: str) -> str:
    marks = set(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670])
    text = "".join(ch for ch in str(text) if ch not in marks)
    text = text.replace("ـ", "")
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    text = text.replace("ى", "ي")
    return " ".join(text.split())


def has_noon_sakinah_or_tanween_like_pattern(text: str, triggers: set[str]) -> bool:
    """
    Approximate search only.

    Looks for ن followed soon by a trigger letter, allowing spaces.
    This is not a final Tajweed labeler.
    """
    compact = text.replace(" ", "")
    for i, ch in enumerate(compact[:-1]):
        if ch == "ن":
            nxt = compact[i + 1]
            if nxt in triggers:
                return True
    return False


def get_text(row: dict) -> str:
    for key in ["aya_text_norm", "aya_text", "quranjson_verse_text", "verse_text_norm", "verse_text", "normalized_text", "source_text"]:
        if row.get(key):
            return norm(str(row[key]))
    return ""


def get_audio_path(row: dict) -> str:
    for key in ["audio_path", "path", "wav_path", "mp3_path"]:
        if row.get(key):
            return str(row[key])
    return ""


def main() -> None:
    candidates = []
    source_counts = Counter()

    for path in MANIFESTS:
        if not path.exists():
            continue

        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                text = get_text(row)
                if not text:
                    continue

                approx_ikhfa = has_noon_sakinah_or_tanween_like_pattern(text, IKHFA_TRIGGER_LETTERS)
                approx_idgham = has_noon_sakinah_or_tanween_like_pattern(text, IDGHAM_TRIGGER_LETTERS)

                if approx_ikhfa and approx_idgham:
                    out = dict(row)
                    out["id"] = row.get("id") or row.get("sample_id")
                    out["candidate_text_norm"] = text
                    out["candidate_audio_path"] = get_audio_path(row)
                    out["approx_transition_rules"] = ["ikhfa", "idgham"]
                    out["transition_multihot"] = [1.0, 1.0]
                    out["transition_combo"] = "ikhfa+idgham_candidate"
                    out["source_manifest"] = str(path)
                    candidates.append(out)
                    source_counts[str(path)] += 1

    out_path = Path("data/analysis/transition_both_label_candidates.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Transition both-label candidate search")
    print("--------------------------------------")
    print(f"candidates: {len(candidates)}")
    print(f"source_counts: {dict(source_counts)}")
    print(f"saved: {out_path}")

    for row in candidates[:10]:
        print("")
        print("id:", row.get("id"))
        print("source:", row.get("source_manifest"))
        print("audio:", row.get("candidate_audio_path"))
        print("text:", row.get("candidate_text_norm"))


if __name__ == "__main__":
    main()
