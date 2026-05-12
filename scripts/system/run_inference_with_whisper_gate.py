from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.content.run_whisper_quran_content_gate import (  # noqa: E402
    get_audio_path,
    get_row_id,
    get_text,
    load_jsonl,
    project_path,
    run_whisper_asr,
    score_content,
)


def row_has_tajweed_scoring_annotations(row: dict) -> bool:
    """Return True only when a row has explicit labels/targets for Tajweed scoring."""
    annotation_keys = [
        "canonical_phonemes",
        "expected_phonemes",
        "phonemes",
        "duration_label",
        "duration_labels",
        "transition_label",
        "transition_labels",
        "burst_label",
        "burst_labels",
        "rule_label",
        "rule_labels",
        "tajweed_errors",
        "expected_errors",
    ]

    for key in annotation_keys:
        value = row.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
        if isinstance(value, dict) and len(value) > 0:
            return True
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (int, float)):
            return True

    return False


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        sep = argv.index("--")
        gate_argv = argv[:sep]
        passthrough = argv[sep + 1 :]
    else:
        gate_argv = argv
        passthrough = []

    parser = argparse.ArgumentParser(
        description="Run Whisper Quran content gate before existing Tajweed inference."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--whisper-content-model-dir", required=True)
    parser.add_argument("--whisper-content-device", default="cuda")
    parser.add_argument("--whisper-content-max-new-tokens", type=int, default=128)
    parser.add_argument("--content-gate-mode", choices=["strict", "near_exact"], default="strict")
    parser.add_argument("--content-gate-min-char-accuracy", type=float, default=0.98)
    parser.add_argument("--content-gate-max-cer", type=float, default=0.02)
    parser.add_argument("--content-gate-output-json")
    parser.add_argument("--allow-content-reject", action="store_true")
    parser.add_argument(
        "--allow-unannotated-tajweed",
        action="store_true",
        help="Allow run_inference.py even when the manifest row has no Tajweed scoring annotations.",
    )
    args = parser.parse_args(gate_argv)

    all_rows = load_jsonl(project_path(args.manifest))

    if args.split == "all":
        selected_global_index = args.sample_index
        row = all_rows[selected_global_index]
    else:
        split_matches = [
            (idx, r)
            for idx, r in enumerate(all_rows)
            if str(r.get("split", "train")) == args.split
        ]
        selected_global_index, row = split_matches[args.sample_index]

    sample_id = get_row_id(row)
    audio_path = get_audio_path(row)
    gold_text = get_text(row)

    pred_text = run_whisper_asr(
        model_dir=args.whisper_content_model_dir,
        audio_path=audio_path,
        device=args.whisper_content_device,
        max_new_tokens=args.whisper_content_max_new_tokens,
    )

    score = score_content(
        gold_text,
        pred_text,
        mode=args.content_gate_mode,
        min_char_accuracy=args.content_gate_min_char_accuracy,
        max_cer=args.content_gate_max_cer,
    )

    gate_result = {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "model_dir": str(project_path(args.whisper_content_model_dir)),
        "gold_text": gold_text,
        "pred_text": pred_text,
        "content_gate": score,
    }

    print("Whisper content gate")
    print("--------------------")
    print(f"Sample ID : {sample_id}")
    print(f"Accepted  : {score['accepted']}")
    print(f"Verdict   : {score['verdict']}")
    print(f"Exact     : {score['exact_compact']}")
    print(f"Char acc  : {score['char_accuracy']:.4f}")
    print(f"CER       : {score['cer']:.4f}")
    print(f"Gold      : {score['gold_norm']}")
    print(f"Pred      : {score['pred_norm']}")
    print("")

    if args.content_gate_output_json:
        out = project_path(args.content_gate_output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gate_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved gate JSON: {out}")
        print("")

    if not score["accepted"] and not args.allow_content_reject:
        print("STOP: content gate rejected this audio. Tajweed scoring was not run.")
        print("Use --allow-content-reject only for debugging.")
        raise SystemExit(2)

    if not row_has_tajweed_scoring_annotations(row) and not args.allow_unannotated_tajweed:
        print("STOP: content was accepted, but this manifest row has no explicit Tajweed scoring annotations.")
        print("This avoids a false 100 score from empty canonical_phonemes.")
        print("")
        print("Use a Tajweed module manifest for duration / transition / burst scoring,")
        print("or pass --allow-unannotated-tajweed only for debugging.")
        raise SystemExit(3)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "system" / "run_inference.py"),
        "--manifest",
        args.manifest,
        "--sample-index",
        str(selected_global_index),
    ] + passthrough

    print(f"Split index       : {args.sample_index}")
    print(f"Global row index  : {selected_global_index}")
    print("")
    print("Running Tajweed inference...")
    print(" ".join(cmd))
    print("")

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
