from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tajweed_assessment.data.labels import normalize_rule_name, rule_to_id
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.inference.pipeline import TajweedInferencePipeline
from tajweed_assessment.models.duration.madd_ghunnah_module import DurationRuleModule
from tajweed_assessment.models.fusion.duration_fusion_calibrator import (
    DURATION_FUSION_LABELS,
    DurationFusionCalibrator,
    build_duration_char_vocab,
    build_duration_fusion_numeric_features,
    encode_duration_context_chars,
)
from tajweed_assessment.settings import load_yaml
from tajweed_assessment.utils.io import load_checkpoint
from tajweed_assessment.utils.seed import seed_everything


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LocalizedDurationBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        return self.classifier(out)


def load_duration_module(checkpoint_name: str = "duration_module.pt") -> DurationRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_duration.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / checkpoint_name)
    model = DurationRuleModule(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_phonemes=model_cfg["num_phonemes"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_localized_duration_module() -> tuple[nn.Module, tuple[str, ...], dict[str, float] | None]:
    ckpt_path = PROJECT_ROOT / "checkpoints" / "localized_duration_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("localized duration checkpoint not found at checkpoints/localized_duration_model.pt")
    ckpt = load_checkpoint(ckpt_path)
    config = ckpt.get("config", {})
    label_vocab = tuple(ckpt.get("label_vocab", ["ghunnah", "madd"]))
    model = LocalizedDurationBiLSTM(
        input_dim=int(config.get("n_mfcc", 13)) * 3,
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_labels=len(label_vocab),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    decoder_path = PROJECT_ROOT / "checkpoints" / "localized_duration_decoder.json"
    thresholds = None
    if decoder_path.exists():
        with decoder_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("decoder_config", data) if isinstance(data, dict) else {}
        thresholds = {str(label): float(raw.get(label, {}).get("threshold", 0.5)) for label in label_vocab}
    return model, label_vocab, thresholds


def build_canonical_rules(row: dict) -> list[int]:
    labels = row.get("normalized_char_labels", [])
    canonical_rules = []
    for item in labels:
        rules = item.get("rules", [])
        label = normalize_rule_name(rules[0]) if rules else "none"
        canonical_rules.append(rule_to_id.get(label, rule_to_id["none"]))
    return canonical_rules


def build_canonical_phonemes(row: dict) -> list[int]:
    phonemes = row.get("canonical_phonemes") or []
    if not phonemes:
        return []
    from tajweed_assessment.data.labels import phoneme_to_id
    return [phoneme_to_id[p] for p in phonemes if p in phoneme_to_id]


def build_canonical_chars(row: dict) -> list[str]:
    labels = row.get("normalized_char_labels", [])
    if labels:
        return [str(item.get("char", "")) for item in labels]
    text = row.get("normalized_text") or row.get("text") or ""
    return list(str(text))


def split_indices_by_reciter(rows: list[dict], val_fraction: float = 0.2, seed: int = 7) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = str(row.get("reciter_id") or "Unknown")
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    for key in group_keys:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(groups[key])
    train_idx, val_idx = [], []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    if not train_idx or not val_idx:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))
    return train_idx, val_idx


def split_indices_by_verse(rows: list[dict], val_fraction: float = 0.2, seed: int = 7) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        text = str(row.get("normalized_text") or row.get("text") or "")
        verse_key = str(row.get("quranjson_verse_key") or "")
        surah = str(row.get("surah_name") or "")
        key = f"{surah}|{verse_key}|{text}"
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    for key in group_keys:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(groups[key])
    train_idx, val_idx = [], []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    if not train_idx or not val_idx:
        split = max(1, int(0.8 * len(rows)))
        train_idx = list(range(split))
        val_idx = list(range(split, len(rows)))
    return train_idx, val_idx


class FusionExampleDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        item = self.examples[idx]
        return {
            "numeric": torch.tensor(item["numeric_features"], dtype=torch.float32),
            "prev_char_id": torch.tensor(item["prev_char_id"], dtype=torch.long),
            "curr_char_id": torch.tensor(item["curr_char_id"], dtype=torch.long),
            "next_char_id": torch.tensor(item["next_char_id"], dtype=torch.long),
            "target": torch.tensor(item["target"], dtype=torch.long),
        }


def collate_fusion_batch(batch: list[dict]) -> dict:
    return {
        "numeric": torch.stack([item["numeric"] for item in batch]),
        "prev_char_id": torch.stack([item["prev_char_id"] for item in batch]),
        "curr_char_id": torch.stack([item["curr_char_id"] for item in batch]),
        "next_char_id": torch.stack([item["next_char_id"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def safe_acc(correct: int, total: int) -> float:
    return correct / max(total, 1)


def evaluate_rule_based_baselines(examples: list[dict], conservative_threshold: float = 0.98) -> dict:
    label_to_id = {label: idx for idx, label in enumerate(DURATION_FUSION_LABELS)}
    sequence_correct = 0
    conservative_correct = 0
    total = 0
    for item in examples:
        numeric = item["numeric_features"]
        seq_pred = "madd" if numeric[0] >= numeric[1] else "ghunnah"
        seq_target = int(item["target"])
        seq_pred_id = label_to_id[seq_pred]
        sequence_correct += int(seq_pred_id == seq_target)

        ghunnah_prob = float(numeric[4])
        localizer_has_ghunnah = bool(numeric[7] >= 0.5)
        conservative_pred = seq_pred
        if seq_pred == "madd" and str(item.get("curr_char") or "") == "ن" and localizer_has_ghunnah and ghunnah_prob >= conservative_threshold:
            conservative_pred = "ghunnah"
        conservative_correct += int(label_to_id[conservative_pred] == seq_target)
        total += 1
    return {
        "sequence_accuracy": safe_acc(sequence_correct, total),
        "conservative_accuracy": safe_acc(conservative_correct, total),
        "total": total,
    }


def build_examples(
    rows: list[dict],
    *,
    duration_model: DurationRuleModule,
    localized_duration_model: nn.Module,
    localized_duration_labels: tuple[str, ...],
    localized_duration_thresholds: dict[str, float] | None,
    char_vocab: dict[str, int],
) -> list[dict]:
    pipeline = TajweedInferencePipeline(
        duration_module=duration_model,
        localized_duration_module=localized_duration_model,
        localized_duration_thresholds=localized_duration_thresholds,
        localized_duration_labels=localized_duration_labels,
        duration_localizer_override_threshold=1.1,
        device="cpu",
    )

    examples: list[dict] = []
    label_to_id = {label: idx for idx, label in enumerate(DURATION_FUSION_LABELS)}
    for idx, row in enumerate(rows, start=1):
        mfcc = extract_mfcc_features(row["audio_path"])
        canonical_rules = build_canonical_rules(row)
        canonical_phonemes = build_canonical_phonemes(row)
        canonical_chars = build_canonical_chars(row)

        result = pipeline.run_modular(
            canonical_phonemes=canonical_phonemes,
            canonical_rules=canonical_rules,
            canonical_chars=canonical_chars,
            word=row.get("normalized_text") or row.get("text") or row.get("id") or "sample",
            duration_x=mfcc,
            localized_duration_x=mfcc,
        )

        for judgment in result.get("module_judgments", []):
            if judgment.get("source_module") != "duration":
                continue
            gold_rule = str(judgment.get("rule"))
            if gold_rule not in label_to_id:
                continue
            pos = int(judgment["position"])
            prev_id, curr_id, next_id = encode_duration_context_chars(canonical_chars, pos, char_vocab)
            numeric = build_duration_fusion_numeric_features(
                sequence_predicted_rule=str(judgment.get("sequence_predicted_rule") or judgment.get("predicted_rule")),
                sequence_confidence=judgment.get("confidence"),
                localized_clip_probabilities=judgment.get("localized_clip_probabilities"),
                localized_predicted_labels=judgment.get("localized_predicted_labels"),
            )
            examples.append(
                {
                    "sample_id": row.get("id"),
                    "target": label_to_id[gold_rule],
                    "gold_rule": gold_rule,
                    "numeric_features": numeric,
                    "prev_char_id": prev_id,
                    "curr_char_id": curr_id,
                    "next_char_id": next_id,
                    "curr_char": canonical_chars[pos] if pos < len(canonical_chars) else "",
                }
            )

        if idx % 25 == 0:
            print(f"Built examples for {idx}/{len(rows)} samples...")

    return examples


def evaluate(model: DurationFusionCalibrator, loader: DataLoader, device: str) -> dict:
    model.eval()
    total = 0
    correct = 0
    class_total = [0 for _ in DURATION_FUSION_LABELS]
    class_correct = [0 for _ in DURATION_FUSION_LABELS]
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["numeric"].to(device),
                batch["prev_char_id"].to(device),
                batch["curr_char_id"].to(device),
                batch["next_char_id"].to(device),
            )
            targets = batch["target"].to(device)
            preds = logits.argmax(dim=-1)
            total += int(targets.numel())
            correct += int((preds == targets).sum().item())
            for idx, _label in enumerate(DURATION_FUSION_LABELS):
                mask = targets == idx
                class_total[idx] += int(mask.sum().item())
                class_correct[idx] += int(((preds == targets) & mask).sum().item())
    return {
        "accuracy": safe_acc(correct, total),
        "class_accuracy": {
            label: safe_acc(class_correct[idx], class_total[idx])
            for idx, label in enumerate(DURATION_FUSION_LABELS)
        },
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_duration_alignment_corpus_torchaudio_strict.jsonl")
    parser.add_argument("--duration-checkpoint", default="duration_module.pt")
    parser.add_argument("--checkpoint-name", default="duration_fusion_calibrator_experimental.pt")
    parser.add_argument("--split-mode", choices=["reciter", "verse"], default="reciter")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--char-embedding-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-json", default="")
    args = parser.parse_args()

    seed_everything(args.seed)
    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    if args.split_mode == "verse":
        train_idx, val_idx = split_indices_by_verse(rows, val_fraction=0.2, seed=args.seed)
    else:
        train_idx, val_idx = split_indices_by_reciter(rows, val_fraction=0.2, seed=args.seed)
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    char_vocab = build_duration_char_vocab(build_canonical_chars(row) for row in rows)

    duration_model = load_duration_module(args.duration_checkpoint)
    localized_duration_model, localized_duration_labels, localized_duration_thresholds = load_localized_duration_module()

    print(f"Split mode: {args.split_mode}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows  : {len(val_rows)}")
    print(f"Char vocab: {len(char_vocab)}")

    train_examples = build_examples(
        train_rows,
        duration_model=duration_model,
        localized_duration_model=localized_duration_model,
        localized_duration_labels=localized_duration_labels,
        localized_duration_thresholds=localized_duration_thresholds,
        char_vocab=char_vocab,
    )
    val_examples = build_examples(
        val_rows,
        duration_model=duration_model,
        localized_duration_model=localized_duration_model,
        localized_duration_labels=localized_duration_labels,
        localized_duration_thresholds=localized_duration_thresholds,
        char_vocab=char_vocab,
    )

    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples  : {len(val_examples)}")
    baseline_metrics = evaluate_rule_based_baselines(val_examples)
    print(
        f"Verse-holdout baselines: sequence_acc={baseline_metrics['sequence_accuracy']:.3f} "
        f"conservative_acc={baseline_metrics['conservative_accuracy']:.3f}"
    )

    train_dataset = FusionExampleDataset(train_examples)
    val_dataset = FusionExampleDataset(val_examples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fusion_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fusion_batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DurationFusionCalibrator(
        num_numeric_features=len(train_examples[0]["numeric_features"]),
        char_vocab_size=len(char_vocab),
        char_embedding_dim=args.char_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    class_counts = torch.zeros(len(DURATION_FUSION_LABELS), dtype=torch.float32)
    for item in train_examples:
        class_counts[item["target"]] += 1
    class_weights = class_counts.sum() / class_counts.clamp(min=1.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val_acc = -1.0
    best_val_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["numeric"].to(device),
                batch["prev_char_id"].to(device),
                batch["curr_char_id"].to(device),
                batch["next_char_id"].to(device),
            )
            targets = batch["target"].to(device)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            preds = logits.argmax(dim=-1)
            total += int(targets.numel())
            correct += int((preds == targets).sum().item())

        train_acc = safe_acc(correct, total)
        val_metrics = evaluate(model, val_loader, device)
        mean_loss = running_loss / max(total, 1)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_metrics = val_metrics
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "char_vocab": char_vocab,
                    "labels": list(DURATION_FUSION_LABELS),
                    "config": {
                        "num_numeric_features": len(train_examples[0]["numeric_features"]),
                        "char_embedding_dim": args.char_embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "split_mode": args.split_mode,
                    },
                },
                PROJECT_ROOT / "checkpoints" / args.checkpoint_name,
            )
            print(f"saved best checkpoint to {PROJECT_ROOT / 'checkpoints' / args.checkpoint_name}")

        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_madd_acc={val_metrics['class_accuracy']['madd']:.3f} "
            f"val_ghunnah_acc={val_metrics['class_accuracy']['ghunnah']:.3f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    print(f"best checkpoint: {PROJECT_ROOT / 'checkpoints' / args.checkpoint_name}")
    if best_val_metrics is not None:
        print(
            f"best_val_acc={best_val_metrics['accuracy']:.3f} "
            f"best_val_madd_acc={best_val_metrics['class_accuracy']['madd']:.3f} "
            f"best_val_ghunnah_acc={best_val_metrics['class_accuracy']['ghunnah']:.3f}"
        )
        if args.metrics_json:
            payload = {
                "split_mode": args.split_mode,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "baseline_metrics": baseline_metrics,
                "best_val_metrics": best_val_metrics,
                "checkpoint": str(PROJECT_ROOT / "checkpoints" / args.checkpoint_name),
            }
            out_path = PROJECT_ROOT / args.metrics_json
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved metrics JSON to {out_path}")


if __name__ == "__main__":
    main()

