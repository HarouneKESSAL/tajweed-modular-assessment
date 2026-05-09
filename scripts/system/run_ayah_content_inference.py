from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule


def resolve_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def compact(text: str) -> str:
    return str(text or "").replace(" ", "")


def edit_distance(a: str, b: str) -> int:
    a = compact(a)
    b = compact(b)

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if ca == cb else 1),
            )
            prev = old

    return dp[-1]


def char_accuracy(gold: str, pred: str) -> float:
    g = compact(gold)
    if not g:
        return 1.0 if not compact(pred) else 0.0
    return max(0.0, 1.0 - edit_distance(gold, pred) / len(g))


def to_int_id_to_char(raw: dict[Any, str]) -> dict[int, str]:
    if isinstance(next(iter(raw.keys())), str):
        return {int(k): v for k, v in raw.items()}
    return raw


def greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    id_to_char: dict[int, str],
    blank_penalty: float,
) -> list[str]:
    adjusted = logits.clone()
    adjusted[..., 0] -= blank_penalty

    ids = adjusted.argmax(dim=-1).detach().cpu()
    out: list[str] = []

    for seq, length in zip(ids, input_lengths.detach().cpu().tolist()):
        chars = []
        prev = 0
        for idx in seq[:length].tolist():
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(id_to_char.get(idx, ""))
            prev = idx
        out.append("".join(chars))

    return out


def load_decoder_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"decoder": "greedy", "blank_penalty": 1.2}
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_model(checkpoint_path: Path, device: str) -> tuple[ContentVerificationModule, dict[int, str], dict[str, int]]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    char_to_id = ckpt["char_to_id"]
    id_to_char = to_int_id_to_char(ckpt["id_to_char"])
    hidden_dim = int(ckpt.get("config", {}).get("model", {}).get("hidden_dim", 96))

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, id_to_char, char_to_id


def select_row(manifest_path: Path, split: str, sample_index: int) -> dict[str, Any]:
    rows = load_jsonl(manifest_path)

    if split != "all":
        rows = [r for r in rows if r.get("split", "train") == split]

    if not rows:
        raise RuntimeError(f"No rows found for split={split!r} in {manifest_path}")

    if sample_index < 0 or sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} out of range for {len(rows)} rows")

    return rows[sample_index]


def run_ayah_content_for_row(
    row: dict[str, Any],
    checkpoint_path: str | Path,
    decoder_config_path: str | Path,
    feature_cache_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    device = device if device == "cpu" or torch.cuda.is_available() else "cpu"

    checkpoint_path = resolve_path(checkpoint_path)
    decoder_config_path = resolve_path(decoder_config_path)
    feature_cache_dir = resolve_path(feature_cache_dir)

    decoder_cfg = load_decoder_config(decoder_config_path)
    blank_penalty = float(decoder_cfg.get("blank_penalty", 1.2))

    model, id_to_char, char_to_id = load_model(checkpoint_path, device)

    sample_id = str(row.get("id") or row.get("sample_id") or "ayah_sample")
    audio_path = row.get("audio_path")
    text = row.get("normalized_text") or row.get("text") or row.get("source_text") or ""

    if not audio_path:
        raise RuntimeError(f"Row {sample_id} has no audio_path.")
    if not text:
        raise RuntimeError(f"Row {sample_id} has no text/normalized_text/source_text.")

    work_manifest = resolve_path("data/interim/ayah_content_inference_one.jsonl")
    work_manifest.parent.mkdir(parents=True, exist_ok=True)

    one_row = dict(row)
    one_row["id"] = f"ayah_content_{sanitize_id(sample_id)}"
    one_row["audio_path"] = audio_path
    one_row["normalized_text"] = text
    one_row["text"] = text
    one_row["split"] = "eval"

    work_manifest.write_text(json.dumps(one_row, ensure_ascii=False) + "\n", encoding="utf-8")

    ds = ChunkedContentDataset(
        work_manifest,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=feature_cache_dir,
        char_to_id=char_to_id,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_content_batch)

    with torch.no_grad():
        batch = next(iter(loader))
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        gold = batch["texts"][0]

        logits = model(x, input_lengths)
        pred = greedy_decode(logits, input_lengths, id_to_char, blank_penalty)[0]

    result = {
        "id": sample_id,
        "audio_path": audio_path,
        "checkpoint": str(checkpoint_path),
        "decoder_config": str(decoder_config_path),
        "blank_penalty": blank_penalty,
        "gold": compact(gold),
        "pred": compact(pred),
        "char_accuracy": char_accuracy(gold, pred),
        "edit_distance": edit_distance(gold, pred),
        "gold_len": len(compact(gold)),
        "pred_len": len(compact(pred)),
        "device": device,
    }

    return result


def run_ayah_content_for_manifest_sample(
    manifest_path: str | Path,
    sample_index: int,
    split: str,
    checkpoint_path: str | Path,
    decoder_config_path: str | Path,
    feature_cache_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    row = select_row(resolve_path(manifest_path), split=split, sample_index=sample_index)
    return run_ayah_content_for_row(
        row=row,
        checkpoint_path=checkpoint_path,
        decoder_config_path=decoder_config_path,
        feature_cache_dir=feature_cache_dir,
        device=device,
    )


def print_ayah_content_report(result: dict[str, Any]) -> None:
    print("Ayah content inference")
    print("----------------------")
    print(f"Sample ID    : {result['id']}")
    print(f"Audio        : {result['audio_path']}")
    print(f"Checkpoint   : {result['checkpoint']}")
    print(f"Decoder cfg  : {result['decoder_config']}")
    print(f"Blank penalty: {result['blank_penalty']}")
    print()
    print(f"Gold         : {result['gold']}")
    print(f"Prediction   : {result['pred']}")
    print()
    print("Metrics:")
    print(json.dumps(
        {
            "char_accuracy": result["char_accuracy"],
            "edit_distance": result["edit_distance"],
            "gold_len": result["gold_len"],
            "pred_len": result["pred_len"],
        },
        ensure_ascii=False,
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/content_ayah_hf_v2_balanced_hd96.pt")
    parser.add_argument("--decoder-config", default="checkpoints/content_ayah_decoder_bp12.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/ayah_content_inference_ssl_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result = run_ayah_content_for_manifest_sample(
        manifest_path=args.manifest,
        split=args.split,
        sample_index=args.sample_index,
        checkpoint_path=args.checkpoint,
        decoder_config_path=args.decoder_config,
        feature_cache_dir=args.feature_cache_dir,
        device=args.device,
    )

    print_ayah_content_report(result)

    if args.output_json:
        out = resolve_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
