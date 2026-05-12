from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from content.train_chunked_content import (
    ChunkedContentDataset,
    collate_content_batch,
    normalize_text_target,
    split_content_indices,
)
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.training.metrics import greedy_decode_from_log_probs


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain mapping: {path}")
    return data


def decode_ids(ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(int(i), "") for i in ids)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + int(ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def char_accuracy(gold: str, pred: str) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(gold, pred) / len(gold))


def get_batch_targets(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    target_key = None
    for key in ("targets", "labels", "y"):
        if key in batch:
            target_key = key
            break

    length_key = None
    for key in ("target_lengths", "label_lengths", "y_lengths"):
        if key in batch:
            length_key = key
            break

    if target_key is None or length_key is None:
        raise KeyError(f"Could not find CTC target keys in batch: {sorted(batch.keys())}")

    targets = batch[target_key]
    target_lengths = batch[length_key]

    if targets.ndim == 2:
        flat_targets = []
        for i, length in enumerate(target_lengths.tolist()):
            flat_targets.append(targets[i, : int(length)])
        targets = torch.cat(flat_targets) if flat_targets else torch.empty(0, dtype=torch.long)

    return targets.long(), target_lengths.long()


def build_speed_config(data_cfg: dict[str, Any]) -> SpeedNormalizationConfig:
    return SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )


def make_dataset(
    *,
    manifest_path: Path,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    char_to_id: dict[str, int] | None,
    feature_cache_dir: Path,
) -> ChunkedContentDataset:
    return ChunkedContentDataset(
        manifest_path,
        sample_rate=int(data_cfg["sample_rate"]),
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=build_speed_config(data_cfg),
        feature_cache_dir=feature_cache_dir,
        char_to_id=char_to_id,
    )


def evaluate_model(
    *,
    model: ContentVerificationModule,
    loader: DataLoader,
    id_to_char: dict[int, str],
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total = 0
    exact = 0
    char_acc_sum = 0.0
    edit_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            log_probs = model(x, input_lengths)
            decoded = greedy_decode_from_log_probs(log_probs.cpu(), batch["input_lengths"])

            for gold_text, pred_ids in zip(batch["texts"], decoded):
                gold = normalize_text_target(gold_text)
                pred = decode_ids(pred_ids, id_to_char)

                total += 1
                exact += int(pred == gold)
                char_acc_sum += char_accuracy(gold, pred)
                edit_sum += levenshtein(gold, pred)

    return {
        "samples": float(total),
        "exact_match": exact / total if total else 0.0,
        "char_accuracy": char_acc_sum / total if total else 0.0,
        "edit_distance": edit_sum / total if total else 0.0,
    }


def train_stage(
    *,
    stage_name: str,
    manifest_path: Path,
    model: ContentVerificationModule,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    char_to_id: dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_checkpoint: Path,
    feature_cache_dir: Path,
) -> dict[str, Any]:
    dataset = make_dataset(
        manifest_path=manifest_path,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        char_to_id=char_to_id,
        feature_cache_dir=feature_cache_dir,
    )

    train_idx, val_idx = split_content_indices(
        dataset.rows,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        seed=int(train_cfg.get("seed", 1337)),
    )

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_content_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    id_to_char = {idx: ch for ch, idx in char_to_id.items()}

    best_metrics: dict[str, float] | None = None
    best_char = -1.0

    print("")
    print(f"Stage: {stage_name}")
    print("-" * (len(stage_name) + 7))
    print(f"manifest      : {manifest_path}")
    print(f"train_samples : {len(train_idx)}")
    print(f"val_samples   : {len(val_idx)}")
    print(f"epochs        : {epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []

        for step, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets, target_lengths = get_batch_targets(batch)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(x, input_lengths)

            loss = criterion(
                log_probs.transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))

            if step % 10 == 0:
                print(
                    f"{stage_name} epoch={epoch} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={losses[-1]:.4f}"
                )

        metrics = evaluate_model(
            model=model,
            loader=val_loader,
            id_to_char=id_to_char,
            device=device,
        )
        mean_loss = sum(losses) / max(1, len(losses))

        print(
            f"{stage_name} epoch={epoch} train_loss={mean_loss:.4f} "
            f"val_exact={metrics['exact_match']:.3f} "
            f"val_char={metrics['char_accuracy']:.3f} "
            f"val_edit={metrics['edit_distance']:.3f}"
        )

        if metrics["char_accuracy"] > best_char:
            best_char = metrics["char_accuracy"]
            best_metrics = metrics

            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_to_id": char_to_id,
                    "id_to_char": {str(k): v for k, v in id_to_char.items()},
                    "config": {
                        "stage_name": stage_name,
                        "data": data_cfg,
                        "model": {
                            **model_cfg,
                            "hidden_dim": int(model.encoder.lstm.hidden_size),
                            "num_phonemes": len(char_to_id) + 1,
                        },
                        "train": train_cfg,
                        "manifest": str(manifest_path),
                    },
                    "val_metrics": metrics,
                },
                output_checkpoint,
            )
            print(f"saved checkpoint: {output_checkpoint}")

    return {
        "stage_name": stage_name,
        "manifest": str(manifest_path),
        "best_metrics": best_metrics or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-manifests",
        nargs="+",
        required=True,
        help="Stage manifests in training order.",
    )
    parser.add_argument(
        "--stage-names",
        nargs="+",
        default=[],
        help="Optional names for stages.",
    )
    parser.add_argument(
        "--epochs-per-stage",
        nargs="+",
        type=int,
        default=[],
        help="Epochs for each stage. If omitted, uses train config epochs.",
    )
    parser.add_argument("--vocab-manifest", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_content.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output-checkpoint", default="checkpoints/content_multitask_word_chunk.pt")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional existing content checkpoint to fine-tune from.",
    )
    parser.add_argument("--output-json", default="data/analysis/content_multitask_word_chunk_train.json")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_multitask_ssl_cache")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_cfg = load_yaml_file(resolve_path(args.data_config))
    model_cfg = load_yaml_file(resolve_path(args.model_config))
    train_cfg = load_yaml_file(resolve_path(args.train_config))

    torch.manual_seed(int(train_cfg.get("seed", 1337)))

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    init_ckpt = None
    if args.init_checkpoint:
        init_path = resolve_path(args.init_checkpoint)
        init_ckpt = torch.load(init_path, map_location="cpu")
        if "char_to_id" not in init_ckpt:
            raise RuntimeError(f"Initial checkpoint has no char_to_id: {init_path}")
        char_to_id = dict(init_ckpt["char_to_id"])

        ckpt_model_cfg = init_ckpt.get("config", {}).get("model", {})
        hidden_dim = int(ckpt_model_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 96)))

        print(f"using checkpoint vocabulary from: {init_path}")
        print(f"checkpoint vocab size including blank: {len(char_to_id) + 1}")
        print(f"checkpoint hidden_dim: {hidden_dim}")
    else:
        vocab_dataset = make_dataset(
            manifest_path=resolve_path(args.vocab_manifest),
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            char_to_id=None,
            feature_cache_dir=resolve_path(args.feature_cache_dir),
        )
        char_to_id = dict(vocab_dataset.char_to_id)
        hidden_dim = int(model_cfg["hidden_dim"])

    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    )

    if init_ckpt is not None:
        state = init_ckpt.get("model_state_dict", init_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("initialized model weights from checkpoint")
        print(f"missing keys: {len(missing)}")
        print(f"unexpected keys: {len(unexpected)}")

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 2))
    learning_rate = args.learning_rate or float(train_cfg.get("learning_rate", 0.001))

    stage_manifests = [resolve_path(path) for path in args.stage_manifests]
    stage_names = args.stage_names or [path.stem for path in stage_manifests]

    if len(stage_names) != len(stage_manifests):
        raise ValueError("--stage-names must match --stage-manifests length")

    if args.epochs_per_stage:
        epochs_per_stage = args.epochs_per_stage
        if len(epochs_per_stage) != len(stage_manifests):
            raise ValueError("--epochs-per-stage must match --stage-manifests length")
    else:
        epochs_per_stage = [int(train_cfg.get("epochs", 3))] * len(stage_manifests)

    print("Multitask content training")
    print("--------------------------")
    print(f"device       : {device}")
    print(f"vocab_size   : {len(char_to_id) + 1}")
    print(f"batch_size   : {batch_size}")
    print(f"learning_rate: {learning_rate}")

    stage_results = []
    for stage_name, manifest_path, epochs in zip(stage_names, stage_manifests, epochs_per_stage):
        stage_result = train_stage(
            stage_name=stage_name,
            manifest_path=manifest_path,
            model=model,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            char_to_id=char_to_id,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_checkpoint=resolve_path(args.output_checkpoint),
            feature_cache_dir=resolve_path(args.feature_cache_dir),
        )
        stage_results.append(stage_result)

    out = {
        "output_checkpoint": str(resolve_path(args.output_checkpoint)),
        "stage_results": stage_results,
    }

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"saved training summary: {output_json}")


if __name__ == "__main__":
    main()
