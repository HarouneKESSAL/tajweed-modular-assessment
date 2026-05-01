from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import subprocess
from datetime import datetime


DEFAULT_OUTPUT_JSON = "data/analysis/final_demo_report.json"
DEFAULT_OUTPUT_MD = "data/analysis/final_demo_report.md"


def load_json(path: str) -> dict:
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return {"available": False, "path": str(full_path)}
    return json.loads(full_path.read_text(encoding="utf-8"))


def write_json(payload: dict, path: str) -> Path:
    output_path = PROJECT_ROOT / path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def refresh_suite(output_json: str) -> None:
    cmd = [
        sys.executable,
        "scripts/system/evaluate_modular_suite.py",
        "--chunked-content-checkpoint",
        "content_chunked_module_hd96_reciter.pt",
        "--content-split-mode",
        "text",
        "--content-decoder-config",
        "checkpoints/content_chunked_decoder_open_hd96.json",
        "--output-json",
        output_json,
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_payload(*, refreshed_suite_path: str | None = None) -> dict:
    suite_path = refreshed_suite_path or "data/analysis/modular_suite_content_open_hd96_textsplit.json"
    suite = load_json(suite_path)
    duration_gate = load_json("data/analysis/duration_pipeline_verse_holdout_comparison.json")
    content_open = load_json("data/analysis/chunked_content_open_hd96_reciter_textsplit_bp16.json")
    distillation = load_json("data/analysis/chunked_content_distillation_comparison.json")
    learner_readiness = load_json("data/analysis/content_learner_readiness_report.json")

    duration = suite.get("duration", {})
    transition = suite.get("transition", {})
    burst = suite.get("burst", {})
    full_verse = suite.get("content_reference_full_verse", {})
    content = suite.get("content", {})
    learned_duration = duration_gate.get("learned", {})

    official = {
        "duration": {
            "status": "promoted",
            "checkpoint": suite.get("duration_checkpoint", "checkpoints/duration_module.pt"),
            "suite_accuracy": duration.get("accuracy"),
            "suite_ghunnah_accuracy": duration.get("rule_summary", {}).get("ghunnah", {}).get("accuracy"),
            "suite_madd_accuracy": duration.get("rule_summary", {}).get("madd", {}).get("accuracy"),
            "strict_verse_holdout_accuracy": learned_duration.get("accuracy"),
            "strict_verse_holdout_ghunnah_accuracy": learned_duration.get("ghunnah_accuracy"),
            "strict_verse_holdout_madd_accuracy": learned_duration.get("madd_accuracy"),
            "note": "Use the verse-held-out score when explaining generalization.",
        },
        "transition": {
            "status": "promoted",
            "checkpoint": suite.get("transition_checkpoint", "checkpoints/transition_module_hardcase.pt"),
            "accuracy": transition.get("accuracy"),
            "class_accuracy": {
                name: values.get("accuracy")
                for name, values in transition.get("class_summary", {}).items()
            },
        },
        "burst": {
            "status": "baseline",
            "checkpoint": "checkpoints/burst_module.pt",
            "accuracy": burst.get("accuracy"),
            "class_accuracy": {
                name: values.get("accuracy")
                for name, values in burst.get("class_summary", {}).items()
            },
        },
        "content": {
            "status": "official_chunked_open_baseline",
            "checkpoint": content_open.get("checkpoint") or "checkpoints/content_chunked_module_hd96_reciter.pt",
            "decoder": content_open.get("decoder", content.get("decoder", {}).get("decoder")),
            "blank_penalty": content_open.get("blank_penalty", content.get("decoder", {}).get("blank_penalty")),
            "lexicon_coverage": content_open.get("eval_text_coverage", content.get("decoder", {}).get("eval_text_coverage")),
            "exact_match": content_open.get("exact_match", content.get("exact_match")),
            "char_accuracy": content_open.get("char_accuracy", content.get("char_accuracy")),
            "edit_distance": content_open.get("edit_distance", content.get("edit_distance")),
            "full_verse_reference": {
                "exact_match": full_verse.get("exact_match"),
                "char_accuracy": full_verse.get("char_accuracy"),
                "edit_distance": full_verse.get("edit_distance"),
            },
            "note": "Content is still chunk-based; learner-level open content recognition is not promoted.",
        },
    }

    experimental = {
        "distilled_content_bridge": {
            "decision": distillation.get("decision"),
            "strict_exact_match": distillation.get("best_distilled_candidate", {}).get("strict_exact_match"),
            "strict_char_accuracy": distillation.get("best_distilled_candidate", {}).get("strict_char_accuracy"),
            "new_aligned_chunk_exact_match": distillation.get("best_distilled_candidate", {}).get("extra_exact_match"),
            "new_aligned_chunk_char_accuracy": distillation.get("best_distilled_candidate", {}).get("extra_char_accuracy"),
            "interpretation": "Preserved old behavior, but did not learn new aligned chunks enough.",
        },
        "learner_level_content": {
            "status": "not_ready",
            "answer": learner_readiness.get("current_answer"),
            "expanded_recognizer_validation_exact": learner_readiness.get("separate_expanded_recognizer_probe", {}).get("validation_exact_match"),
            "expanded_recognizer_validation_char_accuracy": learner_readiness.get("separate_expanded_recognizer_probe", {}).get("validation_char_accuracy"),
            "promotion_gate": learner_readiness.get("promotion_gate_for_learner_level_content", {}),
        },
    }

    return {
        "project": "tajweed-modular-assessment",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Presentation-friendly final demo/evaluation report.",
        "official_baseline": official,
        "experimental_findings": experimental,
        "recommended_demo_commands": [
            ".\\.venv\\Scripts\\python scripts\\system\\generate_demo_report.py",
            ".\\.venv\\Scripts\\python scripts\\system\\run_inference.py --manifest data\\manifests\\retasy_transition_subset.jsonl --sample-index 1 --show-matches",
            ".\\.venv\\Scripts\\python scripts\\system\\evaluate_modular_suite.py --chunked-content-checkpoint content_chunked_module_hd96_reciter.pt --content-split-mode text --content-decoder-config checkpoints\\content_chunked_decoder_open_hd96.json --output-json data\\analysis\\demo_modular_suite_open_hd96.json",
        ],
        "talk_track": [
            "We implemented the PDF architecture as a modular Tajweed assessment pipeline.",
            "Routing chooses the correct specialist module: duration, transition, burst, or content.",
            "MFCC features support rule modules; wav2vec-style SSL features support content recognition.",
            "We promoted only changes that passed explicit comparison gates.",
            "Duration, transition, and burst are usable specialist baselines.",
            "Content improved from weak full-verse recognition to a stronger chunked open baseline.",
            "Learner-level open content recognition remains future work because expanded recognizers do not yet generalize enough.",
        ],
        "source_files": {
            "suite": str(PROJECT_ROOT / suite_path),
            "duration_gate": str(PROJECT_ROOT / "data/analysis/duration_pipeline_verse_holdout_comparison.json"),
            "content_open": str(PROJECT_ROOT / "data/analysis/chunked_content_open_hd96_reciter_textsplit_bp16.json"),
            "distillation": str(PROJECT_ROOT / "data/analysis/chunked_content_distillation_comparison.json"),
            "learner_readiness": str(PROJECT_ROOT / "data/analysis/content_learner_readiness_report.json"),
        },
    }


def build_markdown(payload: dict) -> str:
    official = payload["official_baseline"]
    experimental = payload["experimental_findings"]
    lines = [
        "# Final Demo Report",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Official Baseline",
        "",
        "| Module | Status | Main Result | Notes |",
        "| --- | --- | --- | --- |",
        (
            f"| Duration | {official['duration']['status']} | "
            f"verse-held-out acc `{fmt(official['duration']['strict_verse_holdout_accuracy'])}`; "
            f"suite acc `{fmt(official['duration']['suite_accuracy'])}` | "
            "Use verse-held-out score for generalization. |"
        ),
        (
            f"| Transition | {official['transition']['status']} | "
            f"acc `{fmt(official['transition']['accuracy'])}` | "
            "Hard-case checkpoint promoted. |"
        ),
        (
            f"| Burst | {official['burst']['status']} | "
            f"acc `{fmt(official['burst']['accuracy'])}` | "
            "Qalqalah/burst baseline. |"
        ),
        (
            f"| Content | {official['content']['status']} | "
            f"exact `{fmt(official['content']['exact_match'])}`, char acc `{fmt(official['content']['char_accuracy'])}` | "
            "Chunked open baseline; learner-level content not promoted. |"
        ),
        "",
        "## Experimental Findings",
        "",
        (
            "- Distilled content bridge tied the baseline exact match at "
            f"`{fmt(experimental['distilled_content_bridge']['strict_exact_match'])}`, "
            f"but new aligned chunk char accuracy stayed low at "
            f"`{fmt(experimental['distilled_content_bridge']['new_aligned_chunk_char_accuracy'])}`."
        ),
        (
            "- Learner-level content recognition is not ready: expanded recognizer validation exact match was "
            f"`{fmt(experimental['learner_level_content']['expanded_recognizer_validation_exact'])}` "
            f"and char accuracy was `{fmt(experimental['learner_level_content']['expanded_recognizer_validation_char_accuracy'])}`."
        ),
        "",
        "## What To Say",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["talk_track"])
    lines.extend(["", "## Demo Commands", ""])
    lines.extend(f"```powershell\n{cmd}\n```" for cmd in payload["recommended_demo_commands"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--refresh-suite", action="store_true")
    parser.add_argument("--refreshed-suite-json", default="data/analysis/demo_modular_suite_open_hd96.json")
    args = parser.parse_args()

    refreshed_suite_path = None
    if args.refresh_suite:
        refresh_suite(args.refreshed_suite_json)
        refreshed_suite_path = args.refreshed_suite_json

    payload = build_payload(refreshed_suite_path=refreshed_suite_path)
    json_path = write_json(payload, args.output_json)
    md_path = PROJECT_ROOT / args.output_md
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Saved demo JSON report to {json_path}")
    print(f"Saved demo Markdown report to {md_path}")
    print("\nOfficial baseline:")
    print(f"- duration verse-held-out accuracy: {fmt(payload['official_baseline']['duration']['strict_verse_holdout_accuracy'])}")
    print(f"- transition accuracy: {fmt(payload['official_baseline']['transition']['accuracy'])}")
    print(f"- burst accuracy: {fmt(payload['official_baseline']['burst']['accuracy'])}")
    print(
        "- content exact/char accuracy: "
        f"{fmt(payload['official_baseline']['content']['exact_match'])}/"
        f"{fmt(payload['official_baseline']['content']['char_accuracy'])}"
    )


if __name__ == "__main__":
    main()
