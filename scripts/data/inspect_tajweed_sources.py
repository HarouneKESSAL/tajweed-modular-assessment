from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(".")
SEARCH_DIRS = [
    ROOT / "data" / "external",
    ROOT / "external",
    ROOT / "data",
]


def iter_json_files() -> list[Path]:
    files: list[Path] = []
    for base in SEARCH_DIRS:
        if base.exists():
            files.extend(base.rglob("*.json"))
    return sorted(set(files))


def short(value: Any, limit: int = 500) -> str:
    text = repr(value)
    text = text.replace("\n", " ")
    return text[:limit]


def walk_limited(obj: Any, path: str = "$", depth: int = 0, max_depth: int = 5, out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []

    if depth > max_depth:
        return out

    if isinstance(obj, dict):
        keys = list(obj.keys())
        out.append(f"{path}: dict keys={keys[:30]}")
        for key in keys[:8]:
            walk_limited(obj[key], f"{path}.{key}", depth + 1, max_depth, out)
    elif isinstance(obj, list):
        out.append(f"{path}: list len={len(obj)}")
        for i, item in enumerate(obj[:3]):
            walk_limited(item, f"{path}[{i}]", depth + 1, max_depth, out)
    else:
        out.append(f"{path}: {type(obj).__name__} {short(obj, 160)}")

    return out


def collect_strings(obj: Any, max_items: int = 3000) -> list[str]:
    strings: list[str] = []

    def rec(x: Any) -> None:
        if len(strings) >= max_items:
            return
        if isinstance(x, str):
            strings.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(obj)
    return strings


def main() -> None:
    out_dir = ROOT / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "external_tajweed_json_inspection.md"

    lines: list[str] = []
    lines.append("# External Tajweed JSON inspection")
    lines.append("")

    files = iter_json_files()
    lines.append(f"Found JSON files: {len(files)}")
    lines.append("")

    html_class_counter = Counter()
    style_color_counter = Counter()
    likely_files = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            continue

        strings = collect_strings(data)
        joined_sample = "\n".join(strings[:500])

        has_arabic = bool(re.search(r"[\u0600-\u06FF]", joined_sample))
        has_html = "<span" in joined_sample or "class=" in joined_sample or "style=" in joined_sample
        has_tajweed_words = bool(re.search(r"tajw|tajweed|madd|ghunn|ikhfa|idgham|qalqalah|iqlab", joined_sample, re.I))

        classes = re.findall(r'class=["\']([^"\']+)["\']', joined_sample)
        for cls in classes:
            for c in cls.split():
                html_class_counter[c] += 1

        colors = re.findall(r'(?:color|background-color)\s*:\s*([^;"\']+)', joined_sample, flags=re.I)
        for color in colors:
            style_color_counter[color.strip()] += 1

        score = int(has_arabic) + int(has_html) + int(has_tajweed_words)
        if score:
            likely_files.append((score, path, has_arabic, has_html, has_tajweed_words))

    likely_files.sort(key=lambda x: (-x[0], str(x[1])))

    lines.append("## Likely Quran/Tajweed JSON files")
    lines.append("")
    for score, path, has_arabic, has_html, has_tajweed_words in likely_files[:30]:
        lines.append(f"- `{path}` score={score} arabic={has_arabic} html={has_html} tajweed_terms={has_tajweed_words}")

    lines.append("")
    lines.append("## Top HTML classes")
    lines.append("")
    for k, v in html_class_counter.most_common(80):
        lines.append(f"- `{k}`: {v}")

    lines.append("")
    lines.append("## Top style colors")
    lines.append("")
    for k, v in style_color_counter.most_common(80):
        lines.append(f"- `{k}`: {v}")

    lines.append("")
    lines.append("## Structure previews")
    lines.append("")

    for _, path, *_ in likely_files[:8]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        lines.append(f"### `{path}`")
        lines.append("")
        lines.append("```text")
        for row in walk_limited(data, max_depth=4)[:120]:
            lines.append(row)
        lines.append("```")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8")[:8000])


if __name__ == "__main__":
    main()
