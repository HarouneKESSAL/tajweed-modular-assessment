from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("data/analysis/thesis_ablation_v2")
MODULAR_JSON = ROOT / "modules/modular_tajweed_baseline_burst047.json"
BURST_SWEEP_JSON = ROOT / "burst/burst_threshold_fine_ablation.json"


def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(x):
    if x is None:
        return "n/a"
    return f"{100 * float(x):.2f}%"


def safe(x):
    return "n/a" if x is None else x


def first_dict(d: dict, names: list[str]) -> dict:
    for name in names:
        obj = d.get(name)
        if isinstance(obj, dict):
            return obj
    return {}


def normalize_rows(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["rows", "results", "thresholds", "ablation_results"]:
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
    return []


def main() -> None:
    mod = load(MODULAR_JSON)
    duration = mod.get("duration", {})
    transition = mod.get("transition", {})
    burst = mod.get("burst", {})

    dh = duration.get("hybrid_support", {}) if isinstance(duration.get("hybrid_support"), dict) else {}
    th = transition.get("hybrid_support", {}) if isinstance(transition.get("hybrid_support"), dict) else {}

    burst_rows = normalize_rows(load(BURST_SWEEP_JSON)) if BURST_SWEEP_JSON.exists() else []

    out_json = ROOT / "MODULE_INTERNAL_ABLATION_REPORT.json"
    out_md = ROOT / "MODULE_INTERNAL_ABLATION_REPORT.md"

    report = {
        "duration": {
            "accuracy": duration.get("accuracy"),
            "rule_summary": duration.get("rule_summary"),
            "hybrid_support": dh,
        },
        "transition": {
            "accuracy": transition.get("accuracy"),
            "class_summary": transition.get("class_summary"),
            "hybrid_support": th,
        },
        "burst": {
            "accuracy": burst.get("accuracy"),
            "class_summary": burst.get("class_summary"),
            "threshold_rows": burst_rows,
        },
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Module-internal Ablation Report")
    lines.append("")
    lines.append("This report separates the independent Tajweed modules from the deprecated chunked CTC content module.")
    lines.append("Duration, transition, and burst are evaluated from their own annotated manifests.")
    lines.append("")

    lines.append("## Module-level summary")
    lines.append("")
    lines.append("| module | accuracy | interpretation |")
    lines.append("|---|---:|---|")
    lines.append(f"| duration | {pct(duration.get('accuracy'))} | strongest module |")
    lines.append(f"| transition | {pct(transition.get('accuracy'))} | usable, idgham weaker |")
    lines.append(f"| burst/qalqalah | {pct(burst.get('accuracy'))} | weakest Tajweed module |")

    lines.append("")
    lines.append("## Duration: within-module class ablation")
    lines.append("")
    lines.append("| rule | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, s in (duration.get("rule_summary") or {}).items():
        lines.append(f"| {label} | {safe(s.get('correct'))} | {safe(s.get('total'))} | {pct(s.get('accuracy'))} |")

    lines.append("")
    lines.append("## Duration: localizer support diagnostic")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in [
        "localized_available",
        "localized_same_as_sequence",
        "localized_same_rate",
        "localized_supports_gold",
        "localized_supports_gold_rate",
        "localized_supports_sequence",
        "localized_supports_sequence_rate",
        "localized_disagrees_with_sequence",
    ]:
        v = dh.get(key)
        lines.append(f"| {key} | {pct(v) if isinstance(v, float) else safe(v)} |")

    lines.append("")
    lines.append("## Duration: gold support by class")
    lines.append("")
    lines.append("| rule | supported | total | support rate |")
    lines.append("|---|---:|---:|---:|")
    for label, s in first_dict(dh, ["gold_supported_by_class", "gold_support_by_class"]).items():
        lines.append(f"| {label} | {safe(s.get('supported'))} | {safe(s.get('total'))} | {pct(s.get('rate'))} |")

    lines.append("")
    lines.append("## Transition: within-module class ablation")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, s in (transition.get("class_summary") or {}).items():
        lines.append(f"| {label} | {safe(s.get('correct'))} | {safe(s.get('total'))} | {pct(s.get('accuracy'))} |")

    lines.append("")
    lines.append("## Transition: localizer support diagnostic")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in [
        "localized_available",
        "localized_same_as_whole_verse",
        "localized_same_rate",
        "localized_supports_gold",
        "localized_supports_gold_rate",
        "localized_supports_whole_verse",
        "localized_supports_whole_verse_rate",
        "localized_disagrees_with_whole_verse",
    ]:
        v = th.get(key)
        lines.append(f"| {key} | {pct(v) if isinstance(v, float) else safe(v)} |")

    lines.append("")
    lines.append("## Transition: gold support by class")
    lines.append("")
    lines.append("| class | supported | total | support rate |")
    lines.append("|---|---:|---:|---:|")
    for label, s in first_dict(th, ["gold_supported_by_class", "gold_support_by_class"]).items():
        lines.append(f"| {label} | {safe(s.get('supported'))} | {safe(s.get('total'))} | {pct(s.get('rate'))} |")

    lines.append("")
    lines.append("## Burst: class ablation at selected threshold")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, s in (burst.get("class_summary") or {}).items():
        lines.append(f"| {label} | {safe(s.get('correct'))} | {safe(s.get('total'))} | {pct(s.get('accuracy'))} |")

    lines.append("")
    lines.append("## Burst: threshold ablation")
    lines.append("")
    lines.append("| threshold | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in burst_rows:
        lines.append(
            f"| {r.get('threshold', 'argmax')} | {pct(r.get('accuracy'))} | {pct(r.get('qalqalah_precision'))} | "
            f"{pct(r.get('qalqalah_recall'))} | {pct(r.get('qalqalah_f1'))} | {safe(r.get('fp'))} | "
            f"{safe(r.get('fn'))} | {pct(r.get('none_accuracy'))} | {pct(r.get('qalqalah_accuracy'))} |"
        )

    lines.append("")
    lines.append("## Module-internal conclusion")
    lines.append("")
    lines.append(
        "Duration is already very strong, with both ghunnah and madd above thesis-ready performance. "
        "Transition is acceptable, but idgham is weaker than ikhfa and none. "
        "Burst/qalqalah remains the weakest Tajweed module; threshold 0.47 is the best simple calibration, "
        "but future improvement likely requires hard-example retraining or better localized windows."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
