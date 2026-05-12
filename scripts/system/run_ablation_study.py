from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_SUITE = PROJECT_ROOT / "scripts" / "system" / "evaluate_modular_suite.py"


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    expected_question: str
    args: tuple[str, ...]


OFFICIAL_COMMON_ARGS: tuple[str, ...] = (
    "--duration-fusion-checkpoint",
    "duration_fusion_calibrator_approved.pt",
    "--transition-checkpoint",
    "transition_module_hardcase.pt",
    "--chunked-content-checkpoint",
    "content_chunked_module_hd96_reciter.pt",
    "--content-split-mode",
    "text",
    "--content-decoder-config",
    "checkpoints/content_chunked_decoder_open_hd96.json",
)


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant(
        name="official_full_system",
        description="Official promoted configuration with duration fusion, hardcase transition, burst, and tuned chunked content.",
        expected_question="What is the current full system performance?",
        args=OFFICIAL_COMMON_ARGS,
    ),
    AblationVariant(
        name="duration_without_fusion",
        description="Official system but with learned duration fusion disabled.",
        expected_question="How much does learned duration fusion add?",
        args=OFFICIAL_COMMON_ARGS + ("--disable-duration-fusion",),
    ),
    AblationVariant(
        name="duration_sequence_only",
        description="Official system but with localized duration support and duration fusion disabled.",
        expected_question="How much do localized duration evidence and fusion add over the sequence model?",
        args=OFFICIAL_COMMON_ARGS + ("--disable-localized-duration", "--disable-duration-fusion"),
    ),
    AblationVariant(
        name="transition_original_checkpoint",
        description="Official system but using the original transition checkpoint instead of the hardcase checkpoint.",
        expected_question="How much did transition hardcase training add?",
        args=(
            "--duration-fusion-checkpoint",
            "duration_fusion_calibrator_approved.pt",
            "--transition-checkpoint",
            "transition_module.pt",
            "--chunked-content-checkpoint",
            "content_chunked_module_hd96_reciter.pt",
            "--content-split-mode",
            "text",
            "--content-decoder-config",
            "checkpoints/content_chunked_decoder_open_hd96.json",
        ),
    ),
    AblationVariant(
        name="transition_without_thresholds",
        description="Official system but transition tuned thresholds are disabled.",
        expected_question="How much do transition thresholds add compared with plain argmax?",
        args=OFFICIAL_COMMON_ARGS + ("--disable-transition-thresholds",),
    ),
    AblationVariant(
        name="transition_without_localizer",
        description="Official system but localized transition support is disabled.",
        expected_question="Does transition localized evidence affect the reported support analysis?",
        args=OFFICIAL_COMMON_ARGS + ("--disable-localized-transition",),
    ),
    AblationVariant(
        name="content_without_blank_penalty",
        description="Official system but chunked content uses open greedy decoding with blank penalty 1.0.",
        expected_question="How much does the tuned CTC blank penalty add?",
        args=(
            "--duration-fusion-checkpoint",
            "duration_fusion_calibrator_approved.pt",
            "--transition-checkpoint",
            "transition_module_hardcase.pt",
            "--chunked-content-checkpoint",
            "content_chunked_module_hd96_reciter.pt",
            "--content-split-mode",
            "text",
            "--content-decoder-config",
            "data/analysis/ablation_content_decoder_open_bp10.json",
        ),
    ),
    AblationVariant(
        name="content_tiny_aux_candidate",
        description="Official system but chunked content uses the tiny auxiliary multitask candidate.",
        expected_question="Does the tiny auxiliary content candidate improve the official chunked baseline?",
        args=(
            "--duration-fusion-checkpoint",
            "duration_fusion_calibrator_approved.pt",
            "--transition-checkpoint",
            "transition_module_hardcase.pt",
            "--chunked-content-checkpoint",
            "content_multitask_word_chunk_tiny_aux_v2.pt",
            "--content-split-mode",
            "text",
            "--content-decoder-config",
            "checkpoints/content_chunked_decoder_open_hd96.json",
        ),
    ),
    AblationVariant(
        name="content_original_chunked",
        description="Official system but using the first/default chunked content checkpoint with open decoding.",
        expected_question="How much did the stronger HD96 chunked content baseline add over the earlier chunked model without using a fixed answer list?",
        args=(
            "--duration-fusion-checkpoint",
            "duration_fusion_calibrator_approved.pt",
            "--transition-checkpoint",
            "transition_module_hardcase.pt",
            "--chunked-content-checkpoint",
            "content_chunked_module.pt",
            "--content-split-mode",
            "text",
            "--content-decoder-config",
            "checkpoints/content_chunked_decoder_open.json",
        ),
    ),
)


def ensure_bp10_decoder_config() -> None:
    path = PROJECT_ROOT / "data" / "analysis" / "ablation_content_decoder_open_bp10.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": "open_recognition",
        "blank_penalty": 1.0,
        "use_cleanup": False,
        "decoder": "greedy",
        "beam_width": 1,
        "lexicon_source": "train",
        "notes": "Ablation config: same open greedy decoder as official content, but blank penalty is reset to 1.0.",
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_variants(names: list[str]) -> list[AblationVariant]:
    if not names or names == ["all"]:
        return list(VARIANTS)
    by_name = {variant.name: variant for variant in VARIANTS}
    missing = [name for name in names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise SystemExit(f"Unknown ablation variant(s): {missing}. Available: {available}")
    return [by_name[name] for name in names]


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    duration = summary.get("duration") or {}
    transition = summary.get("transition") or {}
    burst = summary.get("burst") or {}
    content = summary.get("content") or {}
    content_reference = summary.get("content_reference_full_verse") or {}
    weighted_scoring = summary.get("weighted_scoring") or {}

    duration_rules = duration.get("rule_summary") or {}
    transition_classes = transition.get("class_summary") or {}
    burst_classes = burst.get("class_summary") or {}

    return {
        "duration_accuracy": duration.get("accuracy"),
        "duration_ghunnah_accuracy": (duration_rules.get("ghunnah") or {}).get("accuracy"),
        "duration_madd_accuracy": (duration_rules.get("madd") or {}).get("accuracy"),
        "duration_localized_available": (duration.get("hybrid_support") or {}).get("localized_available"),
        "transition_accuracy": transition.get("accuracy"),
        "transition_none_accuracy": (transition_classes.get("none") or {}).get("accuracy"),
        "transition_ikhfa_accuracy": (transition_classes.get("ikhfa") or {}).get("accuracy"),
        "transition_idgham_accuracy": (transition_classes.get("idgham") or {}).get("accuracy"),
        "transition_localized_available": (transition.get("hybrid_support") or {}).get("localized_available"),
        "burst_accuracy": burst.get("accuracy"),
        "burst_none_accuracy": (burst_classes.get("none") or {}).get("accuracy"),
        "burst_qalqalah_accuracy": (burst_classes.get("qalqalah") or {}).get("accuracy"),
        "content_exact_match": content.get("exact_match"),
        "content_char_accuracy": content.get("char_accuracy"),
        "content_edit_distance": content.get("edit_distance"),
        "content_reference_full_verse_exact": content_reference.get("exact_match"),
        "content_reference_full_verse_char_accuracy": content_reference.get("char_accuracy"),
        "estimated_average_score": weighted_scoring.get("estimated_average_score"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def delta(value: Any, baseline: Any, lower_is_better: bool = False) -> str:
    if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)):
        return "-"
    diff = float(value) - float(baseline)
    if lower_is_better:
        diff = -diff
    if abs(diff) < 0.0005:
        diff = 0.0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def write_markdown_report(
    *,
    output_path: Path,
    results: list[dict[str, Any]],
    smoke: bool,
) -> None:
    official = next((row for row in results if row["variant"] == "official_full_system"), None)
    official_metrics = official.get("metrics", {}) if official else {}
    lines = [
        "# Modular Ablation Study",
        "",
        f"Mode: {'smoke / limited samples' if smoke else 'full evaluation'}",
        "",
        "This report compares the official full system against variants where one component is removed or replaced.",
        "A positive delta means the variant is better than the official baseline for that metric, except edit distance where lower is better.",
        "",
        "## Summary Table",
        "",
        "| Variant | Score | Δ | Duration acc | Δ | Transition acc | Δ | Burst acc | Δ | Content exact | Δ | Content char | Δ | Content edit | Δ better |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in results:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    fmt(metrics.get("estimated_average_score")),
                    delta(metrics.get("estimated_average_score"), official_metrics.get("estimated_average_score")),
                    fmt(metrics.get("duration_accuracy")),
                    delta(metrics.get("duration_accuracy"), official_metrics.get("duration_accuracy")),
                    fmt(metrics.get("transition_accuracy")),
                    delta(metrics.get("transition_accuracy"), official_metrics.get("transition_accuracy")),
                    fmt(metrics.get("burst_accuracy")),
                    delta(metrics.get("burst_accuracy"), official_metrics.get("burst_accuracy")),
                    fmt(metrics.get("content_exact_match")),
                    delta(metrics.get("content_exact_match"), official_metrics.get("content_exact_match")),
                    fmt(metrics.get("content_char_accuracy")),
                    delta(metrics.get("content_char_accuracy"), official_metrics.get("content_char_accuracy")),
                    fmt(metrics.get("content_edit_distance")),
                    delta(metrics.get("content_edit_distance"), official_metrics.get("content_edit_distance"), lower_is_better=True),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Variant Details", ""])
    for row in results:
        lines.extend(
            [
                f"### {row['variant']}",
                "",
                f"Question: {row['expected_question']}",
                "",
                f"Description: {row['description']}",
                "",
                f"Output JSON: `{row['output_json']}`",
                "",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_variant(
    *,
    variant: AblationVariant,
    output_dir: Path,
    smoke: bool,
    extra_suite_args: list[str],
) -> dict[str, Any]:
    output_json = output_dir / f"{variant.name}.json"
    cmd = [
        sys.executable,
        str(EVALUATE_SUITE),
        *variant.args,
        "--output-json",
        str(output_json.relative_to(PROJECT_ROOT)),
    ]
    if smoke:
        cmd.extend(
            [
                "--duration-limit",
                "40",
                "--transition-limit",
                "40",
                "--burst-limit",
                "40",
                "--content-limit",
                "40",
            ]
        )
    cmd.extend(extra_suite_args)

    print(f"\n=== Running {variant.name} ===")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Ablation variant failed: {variant.name}")

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "variant": variant.name,
        "description": variant.description,
        "expected_question": variant.expected_question,
        "output_json": str(output_json.relative_to(PROJECT_ROOT)),
        "metrics": extract_metrics(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["all"],
        help="Variant names to run, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis/ablation_study",
        help="Directory for per-variant JSON files and the summary report.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast limited-sample ablation to validate wiring. Do not use smoke numbers in the thesis.",
    )
    parser.add_argument(
        "--suite-arg",
        action="append",
        default=[],
        help="Extra argument to pass through to evaluate_modular_suite.py. Repeat for multiple args.",
    )
    args = parser.parse_args()

    ensure_bp10_decoder_config()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_variants(args.variants)
    results = [
        run_variant(
            variant=variant,
            output_dir=output_dir,
            smoke=args.smoke,
            extra_suite_args=args.suite_arg,
        )
        for variant in selected
    ]

    summary = {
        "mode": "smoke" if args.smoke else "full",
        "variants": results,
    }
    summary_json = output_dir / ("summary_smoke.json" if args.smoke else "summary.json")
    summary_md = output_dir / ("summary_smoke.md" if args.smoke else "summary.md")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(output_path=summary_md, results=results, smoke=args.smoke)

    print("")
    print(f"Saved ablation JSON to {summary_json}")
    print(f"Saved ablation Markdown to {summary_md}")


if __name__ == "__main__":
    main()
