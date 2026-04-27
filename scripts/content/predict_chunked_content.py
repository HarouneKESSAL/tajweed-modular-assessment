from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json

import torch

from content.train_chunked_content import load_jsonl, normalize_text_target
from content.evaluate_chunked_content import (
    apply_blank_penalty,
    char_accuracy,
    content_postprocess,
    decode_ids,
    decode_sequences,
    levenshtein,
)
from tajweed_assessment.data.audio import load_audio
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.features.ssl import Wav2VecFeatureExtractor
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint, load_json


def print_json(payload: dict) -> None:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    ensure_ascii = encoding.lower() in {"cp1252", "charmap"}
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2))


def default_checkpoint_name() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "content_chunked_module_hd96_reciter.pt"
    if preferred.exists():
        return "content_chunked_module_hd96_reciter.pt"
    return "content_chunked_module.pt"


def default_decoder_config() -> str:
    preferred = PROJECT_ROOT / "checkpoints" / "content_chunked_decoder_open_hd96.json"
    if preferred.exists():
        return "checkpoints/content_chunked_decoder_open_hd96.json"
    return "checkpoints/content_chunked_decoder_open.json"


def load_decoder_config(path: str) -> dict:
    config_path = PROJECT_ROOT / path
    if not config_path.exists():
        return {"decoder": "greedy", "beam_width": 1, "blank_penalty": 0.0, "use_cleanup": False}
    data = load_json(config_path)
    return {
        "decoder": str(data.get("decoder", "greedy")),
        "beam_width": int(data.get("beam_width", 1)),
        "blank_penalty": float(data.get("blank_penalty", 0.0)),
        "use_cleanup": bool(data.get("use_cleanup", False)),
        "path": str(config_path),
    }


def load_content_model(checkpoint_name: str) -> tuple[ContentVerificationModule, dict]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / checkpoint_name
    checkpoint = load_checkpoint(ckpt_path)
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    ckpt_model_cfg = checkpoint.get("config", {}).get("model", {})
    hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg["hidden_dim"]))
    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(checkpoint["char_to_id"]) + 1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def clip_waveform(waveform: torch.Tensor, sample_rate: int, start_sec: float | None, end_sec: float | None) -> torch.Tensor:
    start = 0 if start_sec is None else max(0, int(float(start_sec) * sample_rate))
    end = waveform.size(1) if end_sec is None else min(waveform.size(1), int(float(end_sec) * sample_rate))
    if end <= start:
        raise ValueError(f"Invalid clip range: start_sec={start_sec}, end_sec={end_sec}")
    return waveform[:, start:end]


def predict_content(
    *,
    audio_path: Path,
    expected_text: str,
    checkpoint_name: str,
    decoder_config_path: str,
    start_sec: float | None,
    end_sec: float | None,
) -> dict:
    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )
    model, checkpoint = load_content_model(checkpoint_name)
    decoder_config = load_decoder_config(decoder_config_path)
    id_to_char = {int(k): v for k, v in checkpoint["id_to_char"].items()}

    waveform, sample_rate = load_audio(audio_path, sample_rate=int(data_cfg["sample_rate"]), speed_config=speed_config)
    clip = clip_waveform(waveform, sample_rate, start_sec, end_sec)
    extractor = Wav2VecFeatureExtractor(bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"))
    features = extractor(clip).unsqueeze(0)
    lengths = torch.tensor([features.size(1)], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(features, lengths).cpu()
        decoded = decode_sequences(
            apply_blank_penalty(log_probs, float(decoder_config["blank_penalty"])),
            lengths,
            decoder=str(decoder_config["decoder"]),
            beam_width=int(decoder_config["beam_width"]),
            lexicon_targets=None,
        )[0]
    prediction = decode_ids(decoded, id_to_char)
    if decoder_config.get("use_cleanup"):
        prediction = content_postprocess(prediction)

    expected = normalize_text_target(expected_text)
    result = {
        "audio_path": str(audio_path),
        "checkpoint": str(PROJECT_ROOT / "checkpoints" / checkpoint_name),
        "decoder_config": decoder_config,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "predicted_text": prediction,
        "expected_text": expected,
    }
    if expected:
        result["exact_match"] = prediction == expected
        result["char_accuracy"] = char_accuracy(expected, prediction)
        result["edit_distance"] = levenshtein(expected, prediction)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", default="", help="Direct audio path. If omitted, --manifest and --sample-index are used.")
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--expected-text", default="")
    parser.add_argument("--checkpoint", default=default_checkpoint_name())
    parser.add_argument("--decoder-config", default=default_decoder_config())
    parser.add_argument("--start-sec", type=float, default=None)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    row = {}
    if args.audio_path:
        audio_path = PROJECT_ROOT / args.audio_path if not Path(args.audio_path).is_absolute() else Path(args.audio_path)
        expected_text = args.expected_text
    else:
        rows = load_jsonl(PROJECT_ROOT / args.manifest)
        if args.sample_index < 0 or args.sample_index >= len(rows):
            raise IndexError(f"sample-index out of range: 0 <= idx < {len(rows)}")
        row = rows[args.sample_index]
        audio_path = Path(row["audio_path"])
        expected_text = args.expected_text or str(row.get("normalized_text", ""))
        if args.start_sec is None:
            args.start_sec = float(row.get("start_sec", 0.0))
        if args.end_sec is None:
            args.end_sec = float(row.get("end_sec", 0.0))

    result = predict_content(
        audio_path=audio_path,
        expected_text=expected_text,
        checkpoint_name=args.checkpoint,
        decoder_config_path=args.decoder_config,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )
    if row:
        result["sample"] = {
            "id": row.get("id"),
            "surah_name": row.get("surah_name"),
            "quranjson_verse_key": row.get("quranjson_verse_key"),
        }
    print_json(result)
    if args.output_json:
        output_path = PROJECT_ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved prediction JSON to {output_path}")


if __name__ == "__main__":
    main()
