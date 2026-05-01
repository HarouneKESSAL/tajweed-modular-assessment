from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re
import subprocess
import sys


_UROMAN_KEEP_RE = re.compile(r"([^a-z' ])")
_WHITESPACE_RE = re.compile(r"\s+")
_GLOBAL_ROMANIZE_CACHE: Dict[Tuple[str, str], str] = {}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_uroman(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = _UROMAN_KEEP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def run_uroman(text: str, uroman_cmd: str = "") -> str:
    """
    Uses the current Python interpreter by default, so it works in the active venv.
    """
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


def romanize_char(ch: str, cache: Dict[str, str], uroman_cmd: str = "") -> str:
    if ch in cache:
        return cache[ch]
    global_key = (uroman_cmd, ch)
    if global_key in _GLOBAL_ROMANIZE_CACHE:
        cache[ch] = _GLOBAL_ROMANIZE_CACHE[global_key]
        return cache[ch]

    if ch == " ":
        cache[ch] = ""
        _GLOBAL_ROMANIZE_CACHE[global_key] = ""
        return ""

    raw = run_uroman(ch, uroman_cmd=uroman_cmd)
    norm = normalize_uroman(raw).replace(" ", "")
    cache[ch] = norm
    _GLOBAL_ROMANIZE_CACHE[global_key] = norm
    return norm


def build_arabic_to_romanized_source(
    normalized_text: str,
    *,
    uroman_cmd: str = "",
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    Returns:
      source_chars: flat romanized chars generated from Arabic normalized chars
      source_to_norm_idx: source char index -> normalized Arabic char index
      per_norm_char_meta: one entry per normalized Arabic char
    """
    cache: Dict[str, str] = {}
    source_chars: List[str] = []
    source_to_norm_idx: List[int] = []
    per_norm_char_meta: List[Dict[str, Any]] = []

    for norm_idx, ch in enumerate(normalized_text):
        if ch == " ":
            per_norm_char_meta.append(
                {
                    "norm_index": norm_idx,
                    "char": ch,
                    "romanized_piece": "",
                    "source_start": None,
                    "source_end": None,
                    "source_len": 0,
                }
            )
            continue

        roman = romanize_char(ch, cache=cache, uroman_cmd=uroman_cmd)
        start = len(source_chars)

        for rc in roman:
            source_chars.append(rc)
            source_to_norm_idx.append(norm_idx)

        end = len(source_chars)

        per_norm_char_meta.append(
            {
                "norm_index": norm_idx,
                "char": ch,
                "romanized_piece": roman,
                "source_start": start,
                "source_end": end,
                "source_len": end - start,
            }
        )

    return source_chars, source_to_norm_idx, per_norm_char_meta


def align_source_to_target(
    source_chars: List[str],
    target_chars: List[str],
) -> Tuple[List[Optional[int]], Dict[str, Any]]:
    """
    Align source romanized chars to target aligned romanized chars.

    Returns:
      source_to_target_idx: source char index -> target char index or None
      stats
    """
    n = len(source_chars)
    m = len(target_chars)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        s = source_chars[i - 1]
        for j in range(1, m + 1):
            t = target_chars[j - 1]
            cost = 0 if s == t else 1

            sub = dp[i - 1][j - 1] + cost
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            dp[i][j] = min(sub, delete, insert)

    source_to_target_idx: List[Optional[int]] = [None] * n
    exact_match_count = 0
    substitution_count = 0
    deletion_count = 0
    insertion_count = 0

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = source_chars[i - 1]
            t = target_chars[j - 1]
            cost = 0 if s == t else 1

            if dp[i][j] == dp[i - 1][j - 1] + cost:
                source_to_target_idx[i - 1] = j - 1
                if cost == 0:
                    exact_match_count += 1
                else:
                    substitution_count += 1
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            source_to_target_idx[i - 1] = None
            deletion_count += 1
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertion_count += 1
            j -= 1
            continue

        # safety fallback
        if i > 0:
            source_to_target_idx[i - 1] = None
            deletion_count += 1
            i -= 1
        elif j > 0:
            insertion_count += 1
            j -= 1

    stats = {
        "source_len": n,
        "target_len": m,
        "edit_distance": dp[n][m],
        "exact_match_count": exact_match_count,
        "substitution_count": substitution_count,
        "deletion_count": deletion_count,
        "insertion_count": insertion_count,
        "mapped_source_count": sum(x is not None for x in source_to_target_idx),
    }

    return source_to_target_idx, stats


@dataclass
class ArabicCharTimeSpan:
    norm_index: int
    char: str
    romanized_piece: str
    source_len: int
    mapped_target_count: int
    target_indices: List[int]
    start_sec: Optional[float]
    end_sec: Optional[float]
    fully_mapped: bool


@dataclass
class DurationRuleTimeSpan:
    rule: str
    coarse_group: Optional[str]
    norm_start: int
    norm_end: int
    text: str
    start_sec: Optional[float]
    end_sec: Optional[float]
    total_nonspace_chars: int
    mapped_nonspace_chars: int
    fully_timed: bool


def project_row_to_time(
    row: Dict[str, Any],
    *,
    uroman_cmd: str = "",
) -> Dict[str, Any]:
    normalized_text = row["normalized_text"]
    target_char_alignments = row["char_alignments_romanized"]
    target_chars = [item["char"] for item in target_char_alignments]

    source_chars, source_to_norm_idx, per_norm_char_meta = build_arabic_to_romanized_source(
        normalized_text,
        uroman_cmd=uroman_cmd,
    )

    source_to_target_idx, align_stats = align_source_to_target(source_chars, target_chars)

    # Build Arabic normalized char -> time spans
    arabic_char_time_spans: List[ArabicCharTimeSpan] = []
    for meta in per_norm_char_meta:
        norm_index = meta["norm_index"]
        ch = meta["char"]
        source_start = meta["source_start"]
        source_end = meta["source_end"]
        source_len = meta["source_len"]

        if ch == " " or source_len == 0 or source_start is None or source_end is None:
            arabic_char_time_spans.append(
                ArabicCharTimeSpan(
                    norm_index=norm_index,
                    char=ch,
                    romanized_piece=meta["romanized_piece"],
                    source_len=source_len,
                    mapped_target_count=0,
                    target_indices=[],
                    start_sec=None,
                    end_sec=None,
                    fully_mapped=(ch == " "),
                )
            )
            continue

        target_indices = [
            source_to_target_idx[src_idx]
            for src_idx in range(source_start, source_end)
            if source_to_target_idx[src_idx] is not None
        ]

        target_indices = sorted(set(target_indices))
        fully_mapped = len(target_indices) == source_len

        if target_indices:
            start_sec = float(target_char_alignments[target_indices[0]]["start_sec"])
            end_sec = float(target_char_alignments[target_indices[-1]]["end_sec"])
        else:
            start_sec = None
            end_sec = None

        arabic_char_time_spans.append(
            ArabicCharTimeSpan(
                norm_index=norm_index,
                char=ch,
                romanized_piece=meta["romanized_piece"],
                source_len=source_len,
                mapped_target_count=len(target_indices),
                target_indices=target_indices,
                start_sec=start_sec,
                end_sec=end_sec,
                fully_mapped=fully_mapped,
            )
        )

    # Build duration-rule time spans
    duration_rule_time_spans: List[DurationRuleTimeSpan] = []
    for span in row.get("duration_rule_spans_normalized", []):
        norm_start = int(span["norm_start"])
        norm_end = int(span["norm_end"])

        selected = [
            arabic_char_time_spans[i]
            for i in range(norm_start, norm_end)
            if 0 <= i < len(arabic_char_time_spans) and arabic_char_time_spans[i].char != " "
        ]

        mapped = [x for x in selected if x.start_sec is not None and x.end_sec is not None]

        if mapped:
            start_sec = min(x.start_sec for x in mapped if x.start_sec is not None)
            end_sec = max(x.end_sec for x in mapped if x.end_sec is not None)
        else:
            start_sec = None
            end_sec = None

        duration_rule_time_spans.append(
            DurationRuleTimeSpan(
                rule=span["rule"],
                coarse_group=span.get("coarse_group"),
                norm_start=norm_start,
                norm_end=norm_end,
                text=span.get("text", normalized_text[norm_start:norm_end]),
                start_sec=start_sec,
                end_sec=end_sec,
                total_nonspace_chars=len(selected),
                mapped_nonspace_chars=len(mapped),
                fully_timed=(len(selected) > 0 and len(mapped) == len(selected)),
            )
        )

    gold_labels = row.get("gold_duration_labels", [])
    projected_labels = row.get("projected_duration_labels", [])

    return {
        "id": row["id"],
        "audio_path": row["audio_path"],
        "surah_name": row.get("surah_name"),
        "quranjson_verse_key": row.get("quranjson_verse_key"),
        "quranjson_verse_index": row.get("quranjson_verse_index"),

        "normalized_text": normalized_text,
        "romanized_text": row.get("romanized_text", ""),
        "gold_duration_labels": gold_labels,
        "projected_duration_labels": projected_labels,

        "alignment_stats": align_stats,
        "arabic_char_time_spans": [asdict(x) for x in arabic_char_time_spans],
        "duration_rule_time_spans": [asdict(x) for x in duration_rule_time_spans],
    }
