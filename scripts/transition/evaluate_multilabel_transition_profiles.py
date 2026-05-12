from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tajweed_assessment.evaluation.transition_multilabel_profiles import (
    evaluate_transition_multilabel_profiles,
    save_transition_multilabel_profile_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold-config", default="configs/transition_multilabel_thresholds.yaml")
    parser.add_argument("--profiles", nargs="+", default=["gold_safe", "ikhfa_recall_safe", "merged_best", "retasy_extended_best"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = evaluate_transition_multilabel_profiles(
        manifest_path=args.manifest,
        threshold_config=args.threshold_config,
        profiles=args.profiles,
        limit=args.limit,
        device=args.device,
    )
    save_transition_multilabel_profile_report(args.output_json, result)

    print("Multi-label transition profile evaluation")
    print("-----------------------------------------")
    print(f"samples: {result['samples']}")
    for profile, metrics in result["results"].items():
        print(
            f"{profile}: "
            f"exact={metrics['exact_match']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"predicted={metrics['predicted_combo_counts']}"
        )
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
