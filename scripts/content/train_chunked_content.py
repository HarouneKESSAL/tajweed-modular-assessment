from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tajweed_assessment.data.audio import load_audio
from tajweed_assessment.data.labels import BLANK_ID
from tajweed_assessment.data.speed import SpeedNormalizationConfig
from tajweed_assessment.features.ssl import Wav2VecFeatureExtractor
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.metrics import (
    greedy_decode_from_log_probs,
    phoneme_sequence_accuracy,
    phoneme_token_accuracy,
)
from tajweed_assessment.utils.io import load_checkpoint, load_json
from tajweed_assessment.utils.seed import seed_everything


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text_target(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip() if not ch.isspace())


def build_char_vocab(rows):
    charset = sorted({ch for row in rows for ch in normalize_text_target(row.get("normalized_text", ""))})
    char_to_id = {ch: idx + 1 for idx, ch in enumerate(charset)}
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}
    return char_to_id, id_to_char


class ChunkedContentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16000,
        bundle_name: str = "WAV2VEC2_BASE",
        speed_config=None,
        feature_cache_dir: str | Path | None = None,
        indices=None,
        char_to_id=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        rows = load_jsonl(self.manifest_path)
        rows = [row for row in rows if normalize_text_target(row.get("normalized_text")) and row.get("audio_path")]
        if indices is not None:
            rows = [rows[i] for i in indices]
        self.rows = rows
        self.sample_rate = sample_rate
        self.speed_config = speed_config
        self.ssl = Wav2VecFeatureExtractor(bundle_name=bundle_name)
        self.char_to_id, self.id_to_char = (char_to_id, {idx: ch for ch, idx in char_to_id.items()}) if char_to_id is not None else build_char_vocab(self.rows)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def _cache_path(self, row):
        if self.feature_cache_dir is None:
            return None
        return self.feature_cache_dir / f"{row['id']}_chunk_ssl.pt"

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = normalize_text_target(row.get("normalized_text"))
        target = torch.tensor([self.char_to_id[ch] for ch in text], dtype=torch.long)
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path)
        else:
            waveform, _ = load_audio(row["audio_path"], sample_rate=self.sample_rate, speed_config=self.speed_config)
            start = max(0, int(float(row.get("start_sec", 0.0)) * self.sample_rate))
            end = min(waveform.size(1), int(float(row.get("end_sec", 0.0)) * self.sample_rate))
            if end <= start:
                end = min(waveform.size(1), start + int(0.25 * self.sample_rate))
            clip = waveform[:, start:end]
            x = self.ssl(clip)
            if cache_path is not None:
                torch.save(x.cpu(), cache_path)
        return {
            "x": x,
            "targets": target,
            "text": text,
            "reciter_id": row.get("reciter_id") or "Unknown",
        }


def collate_content_batch(batch):
    xs = [item["x"] for item in batch]
    targets = [item["targets"] for item in batch]
    texts = [item["text"] for item in batch]
    reciters = [item["reciter_id"] for item in batch]
    x_pad = pad_sequence(xs, batch_first=True)
    input_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets, dim=0)
    return {
        "x": x_pad,
        "input_lengths": input_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "raw_targets": targets,
        "texts": texts,
        "reciter_ids": reciters,
    }


def split_content_indices(rows, val_fraction: float = 0.2, seed: int = 7, split_mode: str = "reciter"):
    groups = {}
    for idx, row in enumerate(rows):
        if split_mode == "text":
            key = normalize_text_target(row.get("normalized_text", "")) or "EMPTY"
        else:
            key = row.get("reciter_id") or "Unknown"
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    for key, indices in items:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(indices)
    train_idx, val_idx = [], []
    for key, indices in groups.items():
        (val_idx if key in val_groups else train_idx).extend(indices)
    if not train_idx or not val_idx:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))
    return train_idx, val_idx


def chunk_entry_id(row: dict) -> str:
    return str(row.get("id") or row.get("audio_path") or "")


def load_hardcase_weight_map(path: str | Path) -> dict[str, float]:
    data = load_json(path)
    if isinstance(data, dict):
        if isinstance(data.get("weights"), dict):
            return {str(key): float(value) for key, value in data["weights"].items()}
        if isinstance(data.get("hardcases"), list):
            return {
                str(item.get("sample_id") or item.get("id") or item.get("audio_path")): float(item.get("weight", 1.0))
                for item in data["hardcases"]
                if isinstance(item, dict) and (item.get("sample_id") or item.get("id") or item.get("audio_path"))
            }
    if isinstance(data, list):
        return {
            str(item.get("sample_id") or item.get("id") or item.get("audio_path")): float(item.get("weight", 1.0))
            for item in data
            if isinstance(item, dict) and (item.get("sample_id") or item.get("id") or item.get("audio_path"))
        }
    return {}


def build_chunked_sample_weights(rows, indices, hardcase_weight_map: dict[str, float] | None = None) -> list[float]:
    hardcase_weight_map = hardcase_weight_map or {}
    weights: list[float] = []
    for idx in indices:
        row = rows[idx]
        weight = float(hardcase_weight_map.get(chunk_entry_id(row), 1.0))
        weights.append(max(1.0, weight))
    return weights


def augment_content_features(
    x: torch.Tensor,
    lengths: torch.Tensor,
    *,
    time_mask_count: int = 0,
    time_mask_width: int = 0,
    feature_mask_count: int = 0,
    feature_mask_width: int = 0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    if time_mask_count <= 0 and feature_mask_count <= 0 and noise_std <= 0.0:
        return x
    augmented = x.clone()
    batch_size, _, feature_dim = augmented.shape
    for batch_idx in range(batch_size):
        valid_length = max(1, int(lengths[batch_idx].item()))
        for _ in range(max(0, time_mask_count)):
            width = min(max(1, time_mask_width), valid_length)
            if width >= valid_length:
                start = 0
            else:
                start = random.randint(0, valid_length - width)
            augmented[batch_idx, start : start + width, :] = 0.0
        for _ in range(max(0, feature_mask_count)):
            width = min(max(1, feature_mask_width), feature_dim)
            if width >= feature_dim:
                start = 0
            else:
                start = random.randint(0, feature_dim - width)
            augmented[batch_idx, :valid_length, start : start + width] = 0.0
    if noise_std > 0.0:
        augmented = augmented + torch.randn_like(augmented) * float(noise_std)
    return augmented


def train_content_epoch(model, loader, optimizer, loss_fn, device="cpu", augmentation: dict | None = None):
    model.train()
    augmentation = augmentation or {}
    total_loss = total_token_acc = total_seq_acc = 0.0
    for batch in loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        x = augment_content_features(
            x,
            input_lengths,
            time_mask_count=int(augmentation.get("time_mask_count", 0)),
            time_mask_width=int(augmentation.get("time_mask_width", 0)),
            feature_mask_count=int(augmentation.get("feature_mask_count", 0)),
            feature_mask_width=int(augmentation.get("feature_mask_width", 0)),
            noise_std=float(augmentation.get("noise_std", 0.0)),
        )
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        log_probs = model(x, input_lengths)
        loss = loss_fn(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        decoded = greedy_decode_from_log_probs(log_probs.detach().cpu(), batch["input_lengths"], blank_id=BLANK_ID)
        total_token_acc += phoneme_token_accuracy(decoded, batch["raw_targets"])
        total_seq_acc += phoneme_sequence_accuracy(log_probs.detach().cpu(), batch["input_lengths"], batch["raw_targets"], blank_id=BLANK_ID)
    n = max(len(loader), 1)
    return {"loss": total_loss / n, "token_acc": total_token_acc / n, "seq_acc": total_seq_acc / n}


@torch.no_grad()
def evaluate_content_epoch(model, loader, loss_fn, device="cpu"):
    model.eval()
    total_loss = total_token_acc = total_seq_acc = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        log_probs = model(x, input_lengths)
        loss = loss_fn(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        total_loss += float(loss.item())
        decoded = greedy_decode_from_log_probs(log_probs.detach().cpu(), batch["input_lengths"], blank_id=BLANK_ID)
        total_token_acc += phoneme_token_accuracy(decoded, batch["raw_targets"])
        total_seq_acc += phoneme_sequence_accuracy(log_probs.detach().cpu(), batch["input_lengths"], batch["raw_targets"], blank_id=BLANK_ID)
    n = max(len(loader), 1)
    return {"loss": total_loss / n, "token_acc": total_token_acc / n, "seq_acc": total_seq_acc / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_content_chunks.jsonl")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_chunk_ssl_cache")
    parser.add_argument("--checkpoint", default="checkpoints/content_chunked_module.pt")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint to initialize model weights before training.")
    parser.add_argument("--hardcase-json", default="")
    parser.add_argument("--split-mode", choices=["reciter", "text"], default="reciter")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--time-mask-count", type=int, default=0)
    parser.add_argument("--time-mask-width", type=int, default=0)
    parser.add_argument("--feature-mask-count", type=int, default=0)
    parser.add_argument("--feature-mask-width", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")
    if args.epochs > 0:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size > 0:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.learning_rate > 0:
        train_cfg["learning_rate"] = float(args.learning_rate)
    if args.hidden_dim > 0:
        model_cfg["hidden_dim"] = int(args.hidden_dim)
    seed_everything(train_cfg["seed"])
    paths = ProjectPaths(PROJECT_ROOT)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    full_dataset = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    train_idx, val_idx = split_content_indices(
        full_dataset.rows,
        val_fraction=0.2,
        seed=train_cfg["seed"],
        split_mode=args.split_mode,
    )
    train_ds = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=train_idx,
        char_to_id=full_dataset.char_to_id,
    )
    val_ds = ChunkedContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
        indices=val_idx,
        char_to_id=full_dataset.char_to_id,
    )

    print(f"Chunked content rows: total={len(full_dataset)} train={len(train_ds)} val={len(val_ds)}")
    print(f"Chunked vocab size: {len(full_dataset.char_to_id) + 1}")
    print(f"Chunked split mode: {args.split_mode}")
    print(f"Chunked hidden dim: {model_cfg['hidden_dim']}")
    augmentation_cfg = {
        "time_mask_count": int(args.time_mask_count),
        "time_mask_width": int(args.time_mask_width),
        "feature_mask_count": int(args.feature_mask_count),
        "feature_mask_width": int(args.feature_mask_width),
        "noise_std": float(args.noise_std),
    }
    print(f"Chunked augmentation: {augmentation_cfg}")

    train_sampler = None
    if args.hardcase_json:
        hardcase_weight_map = load_hardcase_weight_map(PROJECT_ROOT / args.hardcase_json)
        sample_weights = build_chunked_sample_weights(full_dataset.rows, train_idx, hardcase_weight_map)
        boosted = sum(1 for weight in sample_weights if weight > 1.0)
        if boosted > 0:
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            mean_weight = sum(sample_weights) / max(len(sample_weights), 1)
            max_weight = max(sample_weights) if sample_weights else 1.0
            print(
                f"Loaded hardcase weights: boosted_train_rows={boosted}/{len(sample_weights)} "
                f"mean_weight={mean_weight:.2f} max_weight={max_weight:.2f}"
            )
        else:
            print("Loaded hardcase weights, but no train rows received weight > 1.0.")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_content_batch,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_content_batch)

    model = ContentVerificationModule(hidden_dim=model_cfg["hidden_dim"], num_phonemes=len(full_dataset.char_to_id) + 1)
    if args.init_checkpoint:
        init_state = load_checkpoint(PROJECT_ROOT / args.init_checkpoint)
        init_char_to_id = init_state.get("char_to_id", {})
        if init_char_to_id != full_dataset.char_to_id:
            raise ValueError("init-checkpoint character vocabulary does not match the training manifest vocabulary")
        model.load_state_dict(init_state["model_state_dict"])
        print(f"Initialized model from {PROJECT_ROOT / args.init_checkpoint}")
    device = train_cfg.get("device", "cpu")
    model = model.to(device)

    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    checkpoint = ModelCheckpoint(checkpoint_path.parent, filename=checkpoint_path.name)
    best_score = float("-inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_content_epoch(model, train_loader, optimizer, loss_fn, device=device, augmentation=augmentation_cfg)
        val_metrics = evaluate_content_epoch(model, val_loader, loss_fn, device=device)
        score = 0.7 * val_metrics["token_acc"] + 0.3 * val_metrics["seq_acc"]
        if score > best_score:
            best_score = score
            checkpoint.step(
                -score,
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "model": model_cfg,
                        "train": train_cfg,
                        "split_mode": args.split_mode,
                        "augmentation": augmentation_cfg,
                    },
                    "char_to_id": full_dataset.char_to_id,
                    "id_to_char": full_dataset.id_to_char,
                },
            )
            print(f"saved best checkpoint to {checkpoint_path}")
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_token_acc={train_metrics['token_acc']:.3f} "
            f"train_seq_acc={train_metrics['seq_acc']:.3f} val_loss={val_metrics['loss']:.4f} "
            f"val_token_acc={val_metrics['token_acc']:.3f} val_seq_acc={val_metrics['seq_acc']:.3f} val_score={score:.3f}"
        )

    state = load_checkpoint(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    final_metrics = evaluate_content_epoch(model, val_loader, loss_fn, device=device)
    print(f"best checkpoint: {checkpoint_path}")
    print(f"final_val_loss={final_metrics['loss']:.4f}")
    print(f"final_val_token_acc={final_metrics['token_acc']:.3f}")
    print(f"final_val_seq_acc={final_metrics['seq_acc']:.3f}")


if __name__ == "__main__":
    main()

