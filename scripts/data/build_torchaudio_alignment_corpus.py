from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import re
import shlex
import subprocess
from collections import Counter


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_uroman(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"([^a-z' ])", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def sanitize_alignment_text(text: str) -> str:
    """
    Remove replacement characters and anything outside Arabic letters/spaces
    before sending to uroman.
    """
    if not text:
        return ""

    text = text.replace("\ufffd", " ")
    text = re.sub(r"[^\u0621-\u063A\u0641-\u064A\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def run_uroman(text: str, uroman_cmd: str = "") -> str:
    """
    Uses the current Python interpreter by default, so it works inside the venv.
    """
    import subprocess
    import sys
    import shlex

    if uroman_cmd:
        cmd = shlex.split(uroman_cmd)
    else:
        cmd = [sys.executable, "-m", "uroman"]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        shell=False,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"uroman failed with code {proc.returncode}\n"
            f"STDERR:\n{stderr}"
        )

    stdout = proc.stdout.decode("utf-8", errors="replace")
    return stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/manifests/retasy_duration_alignment_strict.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/retasy_duration_alignment_corpus_torchaudio.jsonl",
    )
    parser.add_argument(
    "--uroman-cmd",
    default="",
    help="Optional custom uroman command. Defaults to: current_python -m uroman",
    ) 
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 means all rows",
    )
    args = parser.parse_args()


    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_rows = []
    word_count_match_counter = Counter()
    empty_romanized = 0
    skipped_rows = 0
    for row in rows:
        normalized_text = sanitize_alignment_text(row["normalized_text"])

        if not normalized_text:
            empty_romanized += 1
            continue

        try:
            romanized_raw = run_uroman(normalized_text, args.uroman_cmd)
            romanized_text = normalize_uroman(romanized_raw)
        except Exception as e:
            skipped_rows += 1
            print(f"Skipping {row['id']} during uroman: {e}")
            continue

        if not romanized_text:
            empty_romanized += 1

        ar_words = normalized_text.split()
        romanized_words = romanized_text.split()

        word_count_match = len(ar_words) == len(romanized_words)
        word_count_match_counter[word_count_match] += 1

        out_rows.append(
            {
                "id": row["id"],
                "audio_path": row["audio_path"],
                "hf_index": row["hf_index"],
                "surah_name": row.get("surah_name"),
                "quranjson_verse_key": row.get("quranjson_verse_key"),
                "quranjson_verse_index": row.get("quranjson_verse_index"),

                "original_text": row["original_text"],
                "normalized_text": normalized_text,
                "normalized_words_ar": ar_words,

                "romanized_text_raw": romanized_raw,
                "romanized_text": romanized_text,
                "romanized_words": romanized_words,

                "word_count_match": word_count_match,

                "gold_duration_labels": row.get("gold_duration_labels", []),
                "projected_duration_labels": row.get("projected_duration_labels", []),
                "projection_exact_label_match": row.get("projection_exact_label_match", False),

                # Keep these for later time projection
                "duration_rule_spans_normalized": row.get("duration_rule_spans_normalized", []),
                "normalized_char_labels": row.get("normalized_char_labels", []),
            }
        )

    write_jsonl(output_path, out_rows)

    print(f"Input manifest       : {input_path}")
    print(f"Output manifest      : {output_path}")
    print(f"Rows written         : {len(out_rows)}")
    print(f"Empty romanized rows : {empty_romanized}")
    print(f"Skipped rows         : {skipped_rows}")
    print(f"Word-count match     : {dict(word_count_match_counter)}")
    print(f"Skipped rows         : {skipped_rows}")

if __name__ == "__main__":
    main()
