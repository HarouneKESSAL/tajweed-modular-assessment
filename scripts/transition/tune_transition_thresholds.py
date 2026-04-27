from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import argparse
import json
import random

import torch

from tajweed_assessment.data.labels import TRANSITION_RULES, normalize_rule_name
from tajweed_assessment.features.mfcc import extract_mfcc_features
from tajweed_assessment.features.ssl import DummySSLFeatureExtractor
from tajweed_assessment.models.transition.idgham_ikhfa_module import TransitionRuleModule
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


def load_transition_module() -> TransitionRuleModule:
    model_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_transition.yaml")
    ckpt = load_checkpoint(PROJECT_ROOT / "checkpoints" / "transition_module.pt")
    model = TransitionRuleModule(
        mfcc_dim=model_cfg["mfcc_dim"],
        ssl_dim=model_cfg["ssl_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        num_rules=model_cfg["num_rules"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def gold_transition_label(row: dict) -> str:
    label = normalize_rule_name((row.get("canonical_rules") or ["none"])[0])
    return label if label in TRANSITION_RULES else "none"


def split_transition_rows(rows: list[dict], val_fraction: float = 0.2, seed: int = 7) -> list[int]:
    groups = {}
    for idx, row in enumerate(rows):
        group_key = row.get("reciter_id") or "Unknown"
        groups.setdefault(group_key, []).append(idx)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)

    def labels_for_indices(indices):
        labels = set()
        for i in indices:
            label = gold_transition_label(rows[i])
            if label != "none":
                labels.add(label)
        return labels

    target_val_rows = max(1, int(len(rows) * val_fraction))
    val_groups = set()
    val_count = 0
    covered_labels = set()
    required_labels = {gold_transition_label(row) for row in rows if gold_transition_label(row) != "none"}

    informative_groups = []
    background_groups = []
    for key, indices in group_items:
        item = (key, indices, labels_for_indices(indices))
        if item[2]:
            informative_groups.append(item)
        else:
            background_groups.append(item)

    informative_groups.sort(key=lambda item: (len(item[2]), len(item[1])), reverse=True)

    for key, indices, labels in informative_groups:
        if covered_labels >= required_labels:
            break
        if labels - covered_labels:
            val_groups.add(key)
            val_count += len(indices)
            covered_labels.update(labels)

    remaining_groups = [(k, idxs) for k, idxs, _ in informative_groups if k not in val_groups]
    remaining_groups.extend((k, idxs) for k, idxs, _ in background_groups)
    rng.shuffle(remaining_groups)

    for key, indices in remaining_groups:
        if val_count >= target_val_rows:
            break
        val_groups.add(key)
        val_count += len(indices)

    val_idx = []
    for key, indices in groups.items():
        if key in val_groups:
            val_idx.extend(indices)

    if not val_idx:
        split = max(1, int(0.2 * len(rows)))
        val_idx = list(range(split))
    return val_idx


def macro_accuracy_from_confusion(confusion: torch.Tensor) -> float:
    values = []
    for idx in range(confusion.size(0)):
        support = int(confusion[idx].sum().item())
        if support <= 0:
            continue
        values.append(float(confusion[idx, idx].item() / support))
    return sum(values) / len(values) if values else 0.0


def evaluate_thresholds(probs: torch.Tensor, gold: torch.Tensor, ikhfa_thr: float, idgham_thr: float) -> dict:
    pred = torch.zeros(len(gold), dtype=torch.long)
    non_none_best = probs[:, 1:].argmax(dim=1) + 1
    for i in range(len(gold)):
        best_idx = int(non_none_best[i].item())
        best_prob = float(probs[i, best_idx].item())
        if best_idx == 1 and best_prob >= ikhfa_thr:
            pred[i] = 1
        elif best_idx == 2 and best_prob >= idgham_thr:
            pred[i] = 2
        else:
            pred[i] = 0

    confusion = torch.zeros(3, 3, dtype=torch.long)
    for g, p in zip(gold.tolist(), pred.tolist()):
        confusion[g, p] += 1

    acc = float((pred == gold).float().mean().item())
    macro_acc = macro_accuracy_from_confusion(confusion)
    return {"accuracy": acc, "macro_accuracy": macro_acc, "confusion": confusion}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/retasy_transition_subset.jsonl")
    parser.add_argument("--output", default="checkpoints/transition_thresholds.json")
    parser.add_argument("--grid-start", type=float, default=0.4)
    parser.add_argument("--grid-stop", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    rows = load_jsonl(PROJECT_ROOT / args.manifest)
    val_idx = split_transition_rows(rows, val_fraction=0.2, seed=args.seed)
    model = load_transition_module().to("cpu").eval()
    ssl_extractor = DummySSLFeatureExtractor(output_dim=64)

    all_probs = []
    all_gold = []
    for idx in val_idx:
        row = rows[idx]
        mfcc = extract_mfcc_features(row["audio_path"])
        ssl = ssl_extractor.from_mfcc(mfcc)
        lengths = torch.tensor([mfcc.size(0)], dtype=torch.long)
        with torch.no_grad():
            logits = model(mfcc.unsqueeze(0), ssl.unsqueeze(0), lengths)
            probs = logits.softmax(dim=-1)[0]
        all_probs.append(probs)
        all_gold.append(TRANSITION_RULES.index(gold_transition_label(row)))

    probs = torch.stack(all_probs, dim=0)
    gold = torch.tensor(all_gold, dtype=torch.long)

    grid = []
    t = args.grid_start
    while t <= args.grid_stop + 1e-8:
        grid.append(round(t, 4))
        t += args.grid_step

    best = None
    for ikhfa_thr in grid:
        for idgham_thr in grid:
            metrics = evaluate_thresholds(probs, gold, ikhfa_thr, idgham_thr)
            item = {
                "ikhfa_threshold": ikhfa_thr,
                "idgham_threshold": idgham_thr,
                "accuracy": metrics["accuracy"],
                "macro_accuracy": metrics["macro_accuracy"],
                "confusion": metrics["confusion"].tolist(),
            }
            if best is None or item["macro_accuracy"] > best["macro_accuracy"] or (
                item["macro_accuracy"] == best["macro_accuracy"] and item["accuracy"] > best["accuracy"]
            ):
                best = item

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(PROJECT_ROOT / args.manifest),
        "thresholds": {
            "ikhfa": best["ikhfa_threshold"],
            "idgham": best["idgham_threshold"],
        },
        "metrics": {
            "accuracy": best["accuracy"],
            "macro_accuracy": best["macro_accuracy"],
            "confusion": best["confusion"],
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved thresholds to: {output_path}")
    print(f"ikhfa threshold : {best['ikhfa_threshold']:.2f}")
    print(f"idgham threshold: {best['idgham_threshold']:.2f}")
    print(f"accuracy        : {best['accuracy']:.3f}")
    print(f"macro_accuracy  : {best['macro_accuracy']:.3f}")


if __name__ == "__main__":
    main()

