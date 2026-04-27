from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import json
import re


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE_RE = re.compile(r"\s+")
_SURAH_FILE_RE = re.compile(r"surah_(\d+)\.json$", re.IGNORECASE)


def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = text.replace("ـ", "")
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = (
        text.replace("ٱ", "ا")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


@dataclass
class RuleSpan:
    start: int
    end: int
    rule: str
    extra: Dict[str, Any]


@dataclass
class VerseRuleRecord:
    source_json_path: str
    surah_number: Optional[int]
    verse_key: str
    verse_index: Optional[int]
    verse_text: str
    verse_text_norm: str
    rule_spans: List[RuleSpan]


def _guess_surah_number(path: Path) -> Optional[int]:
    m = _SURAH_FILE_RE.search(path.name)
    return int(m.group(1)) if m else None


def _guess_verse_index(verse_key: str) -> Optional[int]:
    m = re.search(r"(\d+)$", verse_key)
    return int(m.group(1)) if m else None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_verse_container(doc: Any) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValueError(f"Expected dict root, got {type(doc)}")

    verse = doc.get("verse")
    if not isinstance(verse, dict):
        raise ValueError("Expected root['verse'] to be a dict")

    return verse


def _extract_text_from_surah_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("text", "content", "aya_text", "ayah_text", "verse_text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

    return ""


def _load_surah_text_map(repo_root: str | Path, surah_number: int) -> Dict[str, str]:
    """
    Reads source/surah/surah_{n}.json and returns:
      verse_key -> verse_text
    """
    repo_root = Path(repo_root)
    path = repo_root / "source" / "surah" / f"surah_{surah_number}.json"
    if not path.exists():
        return {}

    doc = _load_json(path)
    verse_container = _get_verse_container(doc)

    text_map: Dict[str, str] = {}
    for verse_key, payload in verse_container.items():
        text_map[str(verse_key)] = _extract_text_from_surah_payload(payload)

    return text_map


def _looks_like_rule_span(item: Dict[str, Any]) -> bool:
    has_start = any(k in item for k in ("start", "from", "begin", "s"))
    has_end = any(k in item for k in ("end", "to", "finish", "e"))
    has_rule = any(k in item for k in ("rule", "type", "name", "label"))
    return has_start and has_end and has_rule


def _normalize_rule_span(raw: Dict[str, Any]) -> RuleSpan:
    start = raw.get("start", raw.get("from", raw.get("begin", raw.get("s"))))
    end = raw.get("end", raw.get("to", raw.get("finish", raw.get("e"))))
    rule = raw.get("rule", raw.get("type", raw.get("name", raw.get("label"))))

    extra = {
        k: v
        for k, v in raw.items()
        if k not in {
            "start", "from", "begin", "s",
            "end", "to", "finish", "e",
            "rule", "type", "name", "label",
        }
    }

    return RuleSpan(
        start=int(start),
        end=int(end),
        rule=str(rule),
        extra=extra,
    )


def iter_tajweed_files(repo_root: str | Path) -> Iterator[Path]:
    repo_root = Path(repo_root)

    preferred = [
        repo_root / "source" / "tajweed",
        repo_root / "source" / "tajwid",
    ]

    seen: set[Path] = set()

    for directory in preferred:
        if not directory.exists():
            continue
        for path in directory.rglob("surah_*.json"):
            if path not in seen:
                seen.add(path)
                yield path


def load_quranjson_rule_records(repo_root: str | Path) -> List[VerseRuleRecord]:
    repo_root = Path(repo_root)
    records: List[VerseRuleRecord] = []

    for json_path in iter_tajweed_files(repo_root):
        surah_number = _guess_surah_number(json_path)
        if surah_number is None:
            continue

        doc = _load_json(json_path)
        verse_container = _get_verse_container(doc)
        verse_text_map = _load_surah_text_map(repo_root, surah_number)

        for verse_key, payload in verse_container.items():
            if not isinstance(payload, list):
                continue

            spans: List[RuleSpan] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if not _looks_like_rule_span(item):
                    # skip metadata items like:
                    # {'index': '01', 'verse': {'start': 'verse_1', 'end': 'verse_7'}}
                    continue
                spans.append(_normalize_rule_span(item))

            verse_text = verse_text_map.get(str(verse_key), "")
            records.append(
                VerseRuleRecord(
                    source_json_path=str(json_path),
                    surah_number=surah_number,
                    verse_key=str(verse_key),
                    verse_index=_guess_verse_index(str(verse_key)),
                    verse_text=verse_text,
                    verse_text_norm=normalize_arabic_text(verse_text),
                    rule_spans=spans,
                )
            )

    return records


def build_rule_text_index(records: Iterable[VerseRuleRecord]) -> Dict[str, List[VerseRuleRecord]]:
    index: Dict[str, List[VerseRuleRecord]] = {}
    for rec in records:
        if rec.verse_text_norm:
            index.setdefault(rec.verse_text_norm, []).append(rec)
    return index


def export_quranjson_rule_records(repo_root: str | Path, out_jsonl: str | Path) -> Path:
    records = load_quranjson_rule_records(repo_root)
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    export_quranjson_rule_records(
        repo_root="external/quranjson-tajwid",
        out_jsonl="data/manifests/quranjson_rules.jsonl",
    )