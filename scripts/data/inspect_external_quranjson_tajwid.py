from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")
EXTERNAL_ROOT = ROOT / "external" / "quranjson-tajwid"
OUT_MD = ROOT / "data" / "analysis" / "external_quranjson_tajwid_targeted_inspection.md"
OUT_FILES = ROOT / "data" / "analysis" / "external_quranjson_tajwid_json_files.txt"


def short(value: Any, limit: int = 220) -> str:
    text = repr(value).replace("\n", " ")
    return text[:limit]


def iter_json_files() -> list[Path]:
    if not EXTERNAL_ROOT.exists():
        return []
    return sorted(EXTERNAL_ROOT.rglob("*.json"))


def walk_preview(obj: Any, path: str = "$", depth: int = 0, max_depth: int = 5, out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []

    if depth > max_depth:
        return out

    if isinstance(obj, dict):
        keys = list(obj.keys())
        out.append(f"{path}: dict keys={keys[:50]}")
        for key in keys[:12]:
            walk_preview(obj[key], f"{path}.{key}", depth + 1, max_depth, out)
    elif isinstance(obj, list):
        out.append(f"{path}: list len={len(obj)}")
        for i, item in enumerate(obj[:4]):
            walk_preview(item, f"{path}[{i}]", depth + 1, max_depth, out)
    else:
        out.append(f"{path}: {type(obj).__name__} {short(obj)}")

    return out


def collect_stats(obj: Any) -> dict[str, Any]:
    keys = Counter()
    arabic_samples: list[tuple[str, str]] = []
    interesting: list[str] = []
    html_classes = Counter()
    style_colors = Counter()

    interesting_key = re.compile(
        r"(taj|tajweed|rule|color|class|style|type|text|aya|ayah|verse|sura|surah|word|char|letter)",
        re.I,
    )

    def rec(x: Any, path: str = "$") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                k_str = str(k)
                keys[k_str] += 1

                if interesting_key.search(k_str) and len(interesting) < 120:
                    interesting.append(f"{path}.{k_str}: {short(v)}")

                rec(v, f"{path}.{k_str}")

        elif isinstance(x, list):
            for i, item in enumerate(x[:2500]):
                rec(item, f"{path}[{i}]")

        elif isinstance(x, str):
            if re.search(r"[\u0600-\u06FF]", x) and len(arabic_samples) < 100:
                arabic_samples.append((path, x[:260]))

            for cls in re.findall(r'class=["\']([^"\']+)["\']', x):
                for part in cls.split():
                    html_classes[part] += 1

            for color in re.findall(r'(?:color|background-color)\s*:\s*([^;"\']+)', x, flags=re.I):
                style_colors[color.strip()] += 1

    rec(obj)

    return {
        "keys": keys,
        "arabic_samples": arabic_samples,
        "interesting": interesting,
        "html_classes": html_classes,
        "style_colors": style_colors,
    }


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    files = iter_json_files()
    OUT_FILES.write_text("\n".join(str(p) for p in files), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Targeted external quranjson-tajwid inspection")
    lines.append("")
    lines.append(f"External root: `{EXTERNAL_ROOT}`")
    lines.append(f"JSON files found: {len(files)}")
    lines.append("")

    if not files:
        lines.append("No JSON files found. Check that external/quranjson-tajwid exists.")
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")
        print(OUT_MD)
        print(OUT_MD.read_text(encoding="utf-8"))
        return

    scored: list[tuple[int, Path, dict[str, Any], Any]] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        stats = collect_stats(data)
        arabic_count = len(stats["arabic_samples"])
        interesting_count = len(stats["interesting"])
        html_count = sum(stats["html_classes"].values())
        color_count = sum(stats["style_colors"].values())

        score = arabic_count * 5 + interesting_count + html_count + color_count
        scored.append((score, path, stats, data))

    scored.sort(key=lambda x: (-x[0], str(x[1])))

    lines.append("## Candidate JSON files")
    lines.append("")
    for score, path, stats, _ in scored[:40]:
        lines.append(
            f"- `{path}` score={score} "
            f"arabic_samples={len(stats['arabic_samples'])} "
            f"interesting={len(stats['interesting'])} "
            f"classes={len(stats['html_classes'])} "
            f"colors={len(stats['style_colors'])}"
        )

    for score, path, stats, data in scored[:10]:
        lines.append("")
        lines.append(f"## `{path}`")
        lines.append("")
        lines.append(f"score: {score}")
        lines.append("")

        lines.append("### Structure preview")
        lines.append("")
        lines.append("```text")
        for row in walk_preview(data)[:180]:
            lines.append(row)
        lines.append("```")

        lines.append("")
        lines.append("### Top keys")
        lines.append("")
        for k, v in stats["keys"].most_common(80):
            lines.append(f"- `{k}`: {v}")

        lines.append("")
        lines.append("### Arabic samples")
        lines.append("")
        for p, sample in stats["arabic_samples"][:60]:
            lines.append(f"- `{p}`: `{sample}`")

        lines.append("")
        lines.append("### Interesting paths")
        lines.append("")
        for item in stats["interesting"][:80]:
            lines.append(f"- `{item}`")

        lines.append("")
        lines.append("### HTML classes")
        lines.append("")
        for k, v in stats["html_classes"].most_common(60):
            lines.append(f"- `{k}`: {v}")

        lines.append("")
        lines.append("### Style colors")
        lines.append("")
        for k, v in stats["style_colors"].most_common(60):
            lines.append(f"- `{k}`: {v}")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_MD)
    print(OUT_MD.read_text(encoding="utf-8")[:12000])


if __name__ == "__main__":
    main()
