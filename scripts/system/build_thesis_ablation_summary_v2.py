from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("data/analysis/thesis_ablation_v2")


def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(x):
    if x is None:
        return "n/a"
    return f"{100 * float(x):.2f}%"


def val(x, default="n/a"):
    return default if x is None else x


def get_content_after(path: str | Path) -> dict:
    d = load(path)
    o = d["overall_after_muqattaat"]
    return {
        "samples": o.get("samples"),
        "exact_after": o.get("exact_after_rate"),
        "cer_after": o.get("cer_after"),
        "char_acc_after": o.get("avg_char_accuracy_after"),
        "errors": d.get("num_errors_after_muqattaat"),
        "near_misses": d.get("num_near_misses_ge_095"),
        "strong_near_misses": d.get("num_strong_near_misses_ge_098"),
        "muqattaat_remaining": d.get("error_type_counts", {}).get("muqattaat_remaining_errors"),
    }


def first_dict(d: dict, names: list[str]) -> dict:
    for name in names:
        obj = d.get(name)
        if isinstance(obj, dict):
            return obj
    return {}


def normalize_burst_threshold_rows(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["rows", "results", "thresholds", "ablation_results"]:
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        if isinstance(data.get("best"), dict):
            return [data["best"]]
    return []


def best_by(rows: list[dict], key: str) -> dict | None:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: float(r.get(key, 0.0)))


def write_class_table(lines: list[str], title: str, summary: dict) -> None:
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| class | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for label, s in summary.items():
        if isinstance(s, dict):
            lines.append(
                f"| {label} | {val(s.get('correct'))} | {val(s.get('total'))} | {pct(s.get('accuracy'))} |"
            )


def write_support_table(lines: list[str], title: str, summary: dict) -> None:
    lines.append("")
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| class | supported | total | support rate |")
    lines.append("|---|---:|---:|---:|")
    for label, s in summary.items():
        if isinstance(s, dict):
            lines.append(
                f"| {label} | {val(s.get('supported'))} | {val(s.get('total'))} | {pct(s.get('rate'))} |"
            )


def main() -> None:
    v1 = get_content_after(ROOT / "content/whisper_medium_v1_content_gate_errors.json")
    v2 = get_content_after(ROOT / "content/whisper_medium_v2_weighted_content_gate_errors_normfix.json")

    mod = load(ROOT / "modules/modular_tajweed_baseline_burst047.json")
    duration = mod.get("duration", {})
    transition = mod.get("transition", {})
    burst = mod.get("burst", {})

    duration_hybrid = duration.get("hybrid_support", {}) if isinstance(duration.get("hybrid_support"), dict) else {}
    transition_hybrid = transition.get("hybrid_support", {}) if isinstance(transition.get("hybrid_support"), dict) else {}

    burst_threshold_path = ROOT / "burst/burst_threshold_fine_ablation.json"
    burst_threshold_rows = normalize_burst_threshold_rows(load(burst_threshold_path)) if burst_threshold_path.exists() else []
    best_burst_acc = best_by(burst_threshold_rows, "accuracy")
    best_burst_f1 = best_by(burst_threshold_rows, "qalqalah_f1")

    summary = {
        "content_gate": {
            "v1_medium": v1,
            "v2_weighted_normfix": v2,
            "selected": "v2_weighted_normfix",
        },
        "tajweed_modules": {
            "duration": duration,
            "transition": transition,
            "burst": burst,
        },
        "burst_threshold_ablation": {
            "best_accuracy": best_burst_acc,
            "best_qalqalah_f1": best_burst_f1,
            "all_rows": burst_threshold_rows,
        },
        "notes": [
            "Duration, transition, and burst are independent Tajweed modules.",
            "Chunked CTC content is deprecated and kept only as legacy ablation evidence.",
            "Final content gate uses Whisper-medium v2 weighted with muqattaat normalization.",
            "Near-exact content matches are review evidence, not automatic acceptance.",
            "Within-module duration and transition results are reported as class/localizer diagnostic ablations, not as retrained module-off experiments.",
        ],
    }

    out_json = ROOT / "THESIS_ABLATION_SUMMARY.json"
    out_md = ROOT / "THESIS_ABLATION_SUMMARY.md"

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Thesis Ablation Summary v2")
    lines.append("")
    lines.append("## Content gate ablation")
    lines.append("")
    lines.append("| system | samples | exact after muqattaat | char accuracy | CER | errors | near misses >= 0.95 | strong near misses >= 0.98 | muqattaat remaining |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in [("Whisper-medium v1", v1), ("Whisper-medium v2 weighted + normfix", v2)]:
        lines.append(
            f"| {name} | {d['samples']} | {pct(d['exact_after'])} | {pct(d['char_acc_after'])} | "
            f"{pct(d['cer_after'])} | {d['errors']} | {d['near_misses']} | {d['strong_near_misses']} | {d['muqattaat_remaining']} |"
        )

    lines.append("")
    lines.append("## Selected content gate")
    lines.append("")
    lines.append("Whisper-medium v2 weighted + muqattaat normfix is selected because it improves exact acceptance, character accuracy, CER, and remaining error count compared with v1.")
    lines.append("")

    lines.append("## Tajweed module baseline")
    lines.append("")
    lines.append("| module | metric | value |")
    lines.append("|---|---|---:|")
    lines.append(f"| duration | accuracy | {pct(duration.get('accuracy'))} |")
    lines.append(f"| transition | accuracy | {pct(transition.get('accuracy'))} |")
    lines.append(f"| burst | accuracy | {pct(burst.get('accuracy'))} |")

    write_class_table(lines, "Duration within-module ablation: rule breakdown", duration.get("rule_summary") or {})
    write_support_table(lines, "Duration within-module diagnostic: gold support by localized detector", first_dict(duration_hybrid, ["gold_supported_by_class", "gold_support_by_class"]))
    write_support_table(lines, "Duration within-module diagnostic: sequence support by localized detector", first_dict(duration_hybrid, ["sequence_supported_by_class", "sequence_support_by_class"]))

    lines.append("")
    lines.append("## Duration localizer diagnostics")
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
        value = duration_hybrid.get(key)
        lines.append(f"| {key} | {pct(value) if isinstance(value, float) else val(value)} |")

    write_class_table(lines, "Transition within-module ablation: class breakdown", transition.get("class_summary") or {})
    write_support_table(lines, "Transition within-module diagnostic: gold support by localized detector", first_dict(transition_hybrid, ["gold_supported_by_class", "gold_support_by_class"]))
    write_support_table(lines, "Transition within-module diagnostic: whole-verse support by localized detector", first_dict(transition_hybrid, ["whole_verse_supported_by_class", "whole_verse_support_by_class", "whole_verse_support"]))

    lines.append("")
    lines.append("## Transition localizer diagnostics")
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
        value = transition_hybrid.get(key)
        lines.append(f"| {key} | {pct(value) if isinstance(value, float) else val(value)} |")

    write_class_table(lines, "Burst within-module ablation: class breakdown at selected threshold", burst.get("class_summary") or {})

    lines.append("")
    lines.append("## Burst threshold ablation")
    lines.append("")
    lines.append("| threshold | rule | accuracy | precision | recall | f1 | FP | FN | none_acc | qalqalah_acc |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in burst_threshold_rows:
        threshold = r.get("threshold", "argmax")
        rule = r.get("decision_rule", r.get("rule", ""))
        lines.append(
            f"| {threshold} | {rule} | {pct(r.get('accuracy'))} | {pct(r.get('qalqalah_precision'))} | "
            f"{pct(r.get('qalqalah_recall'))} | {pct(r.get('qalqalah_f1'))} | "
            f"{val(r.get('fp'))} | {val(r.get('fn'))} | {pct(r.get('none_accuracy'))} | {pct(r.get('qalqalah_accuracy'))} |"
        )

    lines.append("")
    lines.append("## Best burst threshold")
    lines.append("")
    if best_burst_acc:
        lines.append(f"- Best accuracy threshold: `{best_burst_acc.get('threshold')}` with accuracy {pct(best_burst_acc.get('accuracy'))}.")
    if best_burst_f1:
        lines.append(f"- Best qalqalah F1 threshold: `{best_burst_f1.get('threshold')}` with F1 {pct(best_burst_f1.get('qalqalah_f1'))}.")
    lines.append("")

    lines.append("## Legacy chunked CTC content note")
    lines.append("")
    lines.append(
        "The chunked CTC content module is kept only as a legacy ablation baseline. "
        "It is not used as the final learner-facing content gate. The final content gate is Whisper-medium v2 weighted + muqattaat normfix."
    )

    lines.append("")
    lines.append("## Thesis conclusion")
    lines.append("")
    lines.append(
        "The ablation study shows that duration is the strongest Tajweed module, transition is acceptable but idgham remains weaker, "
        "and burst/qalqalah is the weakest Tajweed module. The main architectural improvement is replacing the deprecated chunked CTC "
        "content module with a learner-style full-ayah Whisper-medium ASR content gate. The selected v2 content gate uses targeted "
        "oversampling and muqattaat normalization, and only exact normalized matches are automatically accepted."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
