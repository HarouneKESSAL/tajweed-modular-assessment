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
from torch.utils.data import DataLoader, Dataset

from tajweed_assessment.data.labels import BLANK_ID
from tajweed_assessment.features.ssl import Wav2VecFeatureExtractor
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule
from tajweed_assessment.settings import ProjectPaths, load_yaml
from tajweed_assessment.training.callbacks import ModelCheckpoint
from tajweed_assessment.training.metrics import (
    greedy_decode_from_log_probs,
    phoneme_sequence_accuracy,
    phoneme_token_accuracy,
)
from tajweed_assessment.utils.io import load_checkpoint
from tajweed_assessment.utils.seed import seed_everything


def normalize_text_target(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip() if not ch.isspace())


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_char_vocab(rows):
    charset = sorted(
        {
            ch
            for row in rows
            for ch in normalize_text_target(
                row.get("normalized_text") or row.get("aya_text_norm") or row.get("text") or ""
            )
        }
    )
    # Reserve 0 for CTC blank.
    char_to_id = {ch: idx + 1 for idx, ch in enumerate(charset)}
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}
    return char_to_id, id_to_char


class ManifestContentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        bundle_name: str = "WAV2VEC2_BASE",
        speed_config=None,
        feature_cache_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        rows = load_jsonl(self.manifest_path)

        filtered = []
        for row in rows:
            if "match_status" in row and row.get("match_status") != "matched_unique":
                continue
            if row.get("final_label") == "not_related_quran":
                continue
            text = normalize_text_target(
                row.get("normalized_text") or row.get("aya_text_norm") or row.get("text") or ""
            )
            if not text:
                continue
            if not row.get("audio_path"):
                continue
            filtered.append(row)

        self.rows = filtered
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.speed_config = speed_config
        self.char_to_id, self.id_to_char = build_char_vocab(self.rows)
        self.ssl = Wav2VecFeatureExtractor(bundle_name=bundle_name)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_path(self, row) -> Path | None:
        if self.feature_cache_dir is None:
            return None
        return self.feature_cache_dir / f"{row.get('id') or row.get('sample_id')}_content_ssl.pt"

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        text = normalize_text_target(
            row.get("normalized_text") or row.get("aya_text_norm") or row.get("text") or ""
        )
        target = torch.tensor([self.char_to_id[ch] for ch in text], dtype=torch.long)
        cache_path = self._cache_path(row)
        if cache_path is not None and cache_path.exists():
            x = torch.load(cache_path)
        else:
            x = self.ssl.forward_path(
                row["audio_path"],
                sample_rate=self.sample_rate,
            )
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


def split_content_indices(rows, val_fraction: float = 0.2, seed: int = 7):
    groups = {}
    for idx, row in enumerate(rows):
        group_key = row.get("reciter_id") or "Unknown"
        groups.setdefault(group_key, []).append(idx)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0

    for key, indices in group_items:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(indices)

    train_idx = []
    val_idx = []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if len(train_idx) == 0 or len(val_idx) == 0:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))

    return train_idx, val_idx


def train_content_epoch(model, loader, optimizer, loss_fn, device: str = "cpu"):
    model.train()
    total_loss = total_token_acc = total_seq_acc = 0.0

    for batch in loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs = model(x, input_lengths)
        loss = loss_fn(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        decoded = greedy_decode_from_log_probs(log_probs.detach().cpu(), batch["input_lengths"], blank_id=BLANK_ID)
        total_token_acc += phoneme_token_accuracy(decoded, batch["raw_targets"])
        total_seq_acc += phoneme_sequence_accuracy(
            log_probs.detach().cpu(),
            batch["input_lengths"],
            batch["raw_targets"],
            blank_id=BLANK_ID,
        )

    n = len(loader)
    return {
        "loss": total_loss / n,
        "token_acc": total_token_acc / n,
        "seq_acc": total_seq_acc / n,
    }


@torch.no_grad()
def evaluate_content_epoch(model, loader, loss_fn, device: str = "cpu"):
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
        total_seq_acc += phoneme_sequence_accuracy(
            log_probs.detach().cpu(),
            batch["input_lengths"],
            batch["raw_targets"],
            blank_id=BLANK_ID,
        )

    n = len(loader)
    return {
        "loss": total_loss / n,
        "token_acc": total_token_acc / n,
        "seq_acc": total_seq_acc / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_quranjson_train.jsonl")
    parser.add_argument("--feature-cache-dir", default="data/interim/content_ssl_cache")
    args = parser.parse_args()

    data_cfg = load_yaml(PROJECT_ROOT / "configs" / "data.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_content.yaml")
    train_cfg = load_yaml(PROJECT_ROOT / "configs" / "train.yaml")

    seed_everything(train_cfg["seed"])
    paths = ProjectPaths(PROJECT_ROOT)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)

    from tajweed_assessment.data.speed import SpeedNormalizationConfig

    speed_config = SpeedNormalizationConfig(
        enabled=bool(data_cfg.get("normalize_speed", False)),
        target_speech_rate=float(data_cfg.get("target_speech_rate", 12.0)),
        min_speed_factor=float(data_cfg.get("min_speed_factor", 1.0)),
        max_speed_factor=float(data_cfg.get("max_speed_factor", 1.35)),
    )

    full_dataset = ManifestContentDataset(
        PROJECT_ROOT / args.manifest,
        sample_rate=data_cfg["sample_rate"],
        n_mfcc=data_cfg["n_mfcc"],
        bundle_name=model_cfg.get("ssl_model_name", "WAV2VEC2_BASE"),
        speed_config=speed_config,
        feature_cache_dir=PROJECT_ROOT / args.feature_cache_dir,
    )
    train_idx, val_idx = split_content_indices(full_dataset.rows, val_fraction=0.2, seed=train_cfg["seed"])
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)

    print(f"Content rows: total={len(full_dataset)} train={len(train_idx)} val={len(val_idx)}")
    print(f"Content vocab size: {len(full_dataset.char_to_id) + 1}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_content_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_content_batch,
    )

    model = ContentVerificationModule(
        hidden_dim=model_cfg["hidden_dim"],
        num_phonemes=len(full_dataset.char_to_id) + 1,
    )
    device = train_cfg.get("device", "cpu")
    model = model.to(device)

    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    checkpoint = ModelCheckpoint(paths.checkpoints, filename="content_module.pt")
    checkpoint_path = paths.checkpoints / "content_module.pt"

    best_score = float("-inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = train_content_epoch(model, train_loader, optimizer, loss_fn, device=device)
        val_metrics = evaluate_content_epoch(model, val_loader, loss_fn, device=device)
        score = 0.7 * val_metrics["token_acc"] + 0.3 * val_metrics["seq_acc"]

        if score > best_score:
            best_score = score
            checkpoint.step(
                -score,
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"model": model_cfg, "train": train_cfg},
                    "char_to_id": full_dataset.char_to_id,
                    "id_to_char": full_dataset.id_to_char,
                },
            )
            print(f"saved best checkpoint to {checkpoint_path}")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_token_acc={train_metrics['token_acc']:.3f} "
            f"train_seq_acc={train_metrics['seq_acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_token_acc={val_metrics['token_acc']:.3f} "
            f"val_seq_acc={val_metrics['seq_acc']:.3f} "
            f"val_score={score:.3f}"
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

