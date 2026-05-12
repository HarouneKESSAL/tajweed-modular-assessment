from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{100.0 * float(v):.2f}%"


def get_nested(d: dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_muqattaat_exact_rate(data: dict):
    if not isinstance(data, dict):
        return None
    if "exact_compact_after_muqattaat_norm" in data:
        return data.get("exact_compact_after_muqattaat_norm")
    if "exact_after_rate" in data:
        return data.get("exact_after_rate")
    overall = data.get("overall_after_muqattaat")
    if isinstance(overall, dict):
        return overall.get("exact_after_rate")
    return None


def get_muqattaat_cer(data: dict):
    if not isinstance(data, dict):
        return None
    if "cer_after_muqattaat_norm" in data:
        return data.get("cer_after_muqattaat_norm")
    if "cer_after" in data:
        return data.get("cer_after")
    overall = data.get("overall_after_muqattaat")
    if isinstance(overall, dict):
        return overall.get("cer_after")
    return None


def get_muqattaat_changed_count(data: dict):
    if not isinstance(data, dict):
        return None
    for key in ["muqattaat_changed_count", "num_muqattaat_changed"]:
        if key in data:
            return data.get(key)
    examples = data.get("muqattaat_changed_examples")
    if isinstance(examples, list):
        return len(examples)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-summary-json", required=True)
    parser.add_argument("--content-muqattaat-json")
    parser.add_argument("--modular-suite-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    content = load_json(args.content_summary_json)
    modular = load_json(args.modular_suite_json)
    muq = load_json(args.content_muqattaat_json) if args.content_muqattaat_json else {}

    duration = modular.get("duration", {})
    transition = modular.get("transition", {})
    burst = modular.get("burst", {})
    weighted = modular.get("weighted_scoring", {})

    content_gate = {
        "checkpoint": content.get("model_dir"),
        "manifest": content.get("manifest"),
        "samples": content.get("samples"),
        "exact_norm_rate": content.get("exact_norm_rate"),
        "exact_compact_rate": content.get("exact_compact_rate"),
        "avg_char_accuracy": content.get("avg_char_accuracy"),
        "cer": content.get("cer"),
        "wer": content.get("wer"),
    }

    if muq:
        content_gate["exact_compact_after_muqattaat_norm"] = muq.get(
            "exact_compact_after_muqattaat_norm"
        )
        content_gate["cer_after_muqattaat_norm"] = muq.get(
            "cer_after_muqattaat_norm"
        )
        content_gate["muqattaat_changed_count"] = muq.get("changed_count")

    tajweed_modules = {
        "duration": {
            "samples": duration.get("samples"),
            "positions": duration.get("positions") or duration.get("total_positions"),
            "correct_positions": duration.get("correct_positions"),
            "accuracy": duration.get("accuracy"),
            "class_summary": duration.get("class_summary") or duration.get("rule_summary"),
            "hybrid_support": duration.get("hybrid_support"),
        },
        "transition": {
            "samples": transition.get("samples"),
            "accuracy": transition.get("accuracy"),
            "class_summary": transition.get("class_summary"),
        },
        "burst": {
            "samples": burst.get("samples"),
            "accuracy": burst.get("accuracy"),
            "class_summary": burst.get("class_summary"),
        },
    }

    old_weighted_score_note = {
        "estimated_average_score": weighted.get("estimated_average_score") or weighted.get("score"),
        "warning": (
            "This weighted score may include the old chunk-content module if the modular-suite JSON was "
            "generated before Whisper content gate integration. Use module accuracies + Whisper content gate "
            "as the honest current system view."
        ),
        "weighted_scoring_raw": weighted,
    }

    conclusions = []
    conclusions.append(
        "Whisper-medium v2 weighted + muqattaat normfix is the current best learner-style content ASR gate."
    )
    conclusions.append(
        "The ASR ayah manifest should not be used directly for Tajweed diagnosis because it lacks rule-level annotations."
    )
    conclusions.append(
        "Tajweed module results must come from annotated duration / transition / burst manifests."
    )
    conclusions.append(
        "The old chunk-content CTC path is deprecated for the learner content-ASR goal."
    )

    report = {
        "content_gate": content_gate,
        "tajweed_modules": tajweed_modules,
        "old_weighted_score_note": old_weighted_score_note,
        "conclusions": conclusions,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Whole-system status report")
    lines.append("")
    lines.append("This report separates the two correct layers:")
    lines.append("")
    lines.append("1. **Content gate**: learner-style Whisper ASR checks whether the recited ayah content matches.")
    lines.append("2. **Tajweed modules**: duration / transition / burst are evaluated only on annotated Tajweed manifests.")
    lines.append("")
    lines.append("## Content gate: Whisper Quran ASR")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| checkpoint | `{content_gate.get('checkpoint')}` |")
    lines.append(f"| samples | {fmt(content_gate.get('samples'))} |")
    lines.append(f"| exact_norm_rate | {pct(content_gate.get('exact_norm_rate'))} |")
    lines.append(f"| exact_compact_rate | {pct(content_gate.get('exact_compact_rate'))} |")
    if "exact_compact_after_muqattaat_norm" in content_gate:
        lines.append(
            f"| exact_compact_after_muqattaat_norm | "
            f"{pct(content_gate.get('exact_compact_after_muqattaat_norm'))} |"
        )
    lines.append(f"| avg_char_accuracy | {pct(content_gate.get('avg_char_accuracy'))} |")
    lines.append(f"| CER | {pct(content_gate.get('cer'))} |")
    if "cer_after_muqattaat_norm" in content_gate:
        lines.append(
            f"| CER_after_muqattaat_norm | {pct(content_gate.get('cer_after_muqattaat_norm'))} |"
        )
    lines.append(f"| WER | {pct(content_gate.get('wer'))} |")
    if "muqattaat_changed_count" in content_gate:
        lines.append(f"| muqattaat_changed_count | {fmt(content_gate.get('muqattaat_changed_count'))} |")

    lines.append("")
    lines.append("## Tajweed module evaluations")
    lines.append("")
    lines.append("| module | samples | units/positions | accuracy |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| duration | {fmt(tajweed_modules['duration'].get('samples'))} | "
        f"{fmt(tajweed_modules['duration'].get('positions'))} | "
        f"{pct(tajweed_modules['duration'].get('accuracy'))} |"
    )
    lines.append(
        f"| transition | {fmt(tajweed_modules['transition'].get('samples'))} | n/a | "
        f"{pct(tajweed_modules['transition'].get('accuracy'))} |"
    )
    lines.append(
        f"| burst | {fmt(tajweed_modules['burst'].get('samples'))} | n/a | "
        f"{pct(tajweed_modules['burst'].get('accuracy'))} |"
    )

    lines.append("")
    lines.append("## Module class summaries")
    for module_name in ["duration", "transition", "burst"]:
        cls = tajweed_modules[module_name].get("class_summary") or {}
        lines.append("")
        lines.append(f"### {module_name}")
        lines.append("")
        if not cls:
            lines.append("_No class summary found._")
            continue
        lines.append("| class | correct | total | accuracy |")
        lines.append("|---|---:|---:|---:|")
        for label, stats in cls.items():
            lines.append(
                f"| {label} | {fmt(stats.get('correct'))} | "
                f"{fmt(stats.get('total'))} | {pct(stats.get('accuracy'))} |"
            )

    lines.append("")
    lines.append("## Important note about weighted score")
    lines.append("")
    lines.append(
        "The modular-suite weighted score may still include the old chunk-content CTC module. "
        "Do **not** treat that as the final integrated score after the Whisper gate change."
    )
    if old_weighted_score_note.get("estimated_average_score") is not None:
        lines.append("")
        lines.append(f"- Old modular-suite estimated score: `{fmt(old_weighted_score_note.get('estimated_average_score'), 3)}`")

    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    for c in conclusions:
        lines.append(f"- {c}")

    lines.append("")
    lines.append("## Current recommended architecture")
    lines.append("")
    lines.append("```text")
    lines.append("Full ayah audio")
    lines.append("→ Whisper-medium v2 Quran ASR content gate")
    lines.append("→ Quran normalization + muqattaat normalization")
    lines.append("→ if content accepted: run Tajweed modules on annotated Tajweed inputs")
    lines.append("→ if content rejected: stop with content mismatch / review required")
    lines.append("```")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8")[:7000])


if __name__ == "__main__":
    main()
