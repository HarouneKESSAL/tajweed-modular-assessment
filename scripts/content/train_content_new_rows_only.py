from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.content.train_chunked_content import ChunkedContentDataset, collate_content_batch
from tajweed_assessment.models.content.wav2vec_ctc import ContentVerificationModule


NEW_CHARS = set("ءؤئثذزصضطظغفق")


def resolve_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def to_int_id_to_char(raw: dict[Any, str]) -> dict[int, str]:
    if isinstance(next(iter(raw.keys())), str):
        return {int(k): v for k, v in raw.items()}
    return raw


def compact(text: str) -> str:
    return str(text).replace(" ", "")


def edit_distance(a: str, b: str) -> int:
    a = compact(a)
    b = compact(b)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = old
    return dp[-1]


def char_acc(gold: str, pred: str) -> float:
    g = compact(gold)
    if not g:
        return 1.0 if not compact(pred) else 0.0
    return max(0.0, 1.0 - edit_distance(gold, pred) / len(g))


def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor, id_to_char: dict[int, str]) -> list[str]:
    ids = logits.argmax(dim=-1).detach().cpu()
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


def make_loader(
    manifest_path: Path,
    char_to_id: dict[str, int],
    feature_cache_dir: Path,
    split: str,
    batch_size: int,
) -> DataLoader:
    rows = load_jsonl(manifest_path)
    indices = [i for i, row in enumerate(rows) if row.get("split", "train") == split]

    ds = ChunkedContentDataset(
        manifest_path,
        sample_rate=16000,
        bundle_name="WAV2VEC2_BASE",
        feature_cache_dir=feature_cache_dir,
        indices=indices,
        char_to_id=char_to_id,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_content_batch,
    )


def evaluate(model, loader, id_to_char, device: str) -> dict[str, float]:
    model.eval()
    total = 0
    exact = 0
    acc_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            texts = batch["texts"]

            logits = model(x, input_lengths)
            preds = greedy_decode(logits, input_lengths, id_to_char)

            for gold, pred in zip(texts, preds):
                total += 1
                acc_sum += char_acc(gold, pred)
                exact += int(compact(gold) == compact(pred))

    return {
        "samples": total,
        "exact": exact / max(1, total),
        "char_accuracy": acc_sum / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--feature-cache-dir", default="data/interim/content_new_rows_only_ssl_cache")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_path(args.init_checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)

    char_to_id = ckpt["char_to_id"]
    id_to_char = to_int_id_to_char(ckpt["id_to_char"])

    hidden_dim = int(ckpt.get("config", {}).get("model", {}).get("hidden_dim", 96))
    model = ContentVerificationModule(
        hidden_dim=hidden_dim,
        num_phonemes=len(char_to_id) + 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    weight = model.ctc_head.proj.weight
    bias = model.ctc_head.proj.bias
    weight.requires_grad = True
    bias.requires_grad = True

    new_ids = sorted(char_to_id[ch] for ch in NEW_CHARS if ch in char_to_id)

    weight_mask = torch.zeros_like(weight, device=device)
    bias_mask = torch.zeros_like(bias, device=device)
    weight_mask[new_ids, :] = 1.0
    bias_mask[new_ids] = 1.0

    weight.register_hook(lambda grad: grad * weight_mask)
    bias.register_hook(lambda grad: grad * bias_mask)

    train_loader = make_loader(
        resolve_path(args.manifest),
        char_to_id,
        resolve_path(args.feature_cache_dir),
        "train",
        args.batch_size,
    )
    val_loader = make_loader(
        resolve_path(args.manifest),
        char_to_id,
        resolve_path(args.feature_cache_dir),
        "val",
        args.batch_size,
    )

    optimizer = torch.optim.AdamW([weight, bias], lr=args.learning_rate)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print("New-row-only content training")
    print("-----------------------------")
    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"manifest: {resolve_path(args.manifest)}")
    print(f"new_chars: {''.join(ch for ch in NEW_CHARS if ch in char_to_id)}")
    print(f"new_ids: {new_ids}")
    print(f"train_batches: {len(train_loader)}")
    print(f"val_batches: {len(val_loader)}")
    print()

    history = []
    best = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, input_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            steps += 1
            total_loss += float(loss.detach().cpu())

            if steps % 20 == 0:
                print(f"epoch={epoch} step={steps}/{len(train_loader)} loss={float(loss.detach().cpu()):.4f}")

        metrics = evaluate(model, val_loader, id_to_char, device)
        train_loss = total_loss / max(1, steps)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_exact={metrics['exact']:.3f} val_char={metrics['char_accuracy']:.3f}"
        )

        if best is None or metrics["char_accuracy"] > best["char_accuracy"]:
            best = dict(row)
            out = dict(ckpt)
            out["model_state_dict"] = model.state_dict()
            out["new_rows_only_training"] = {
                "init_checkpoint": str(ckpt_path),
                "manifest": str(resolve_path(args.manifest)),
                "new_chars": sorted(NEW_CHARS),
                "new_ids": new_ids,
                "best": best,
            }
            torch.save(out, resolve_path(args.output_checkpoint))
            print(f"saved best checkpoint: {resolve_path(args.output_checkpoint)}")

    result = {
        "init_checkpoint": str(ckpt_path),
        "manifest": str(resolve_path(args.manifest)),
        "output_checkpoint": str(resolve_path(args.output_checkpoint)),
        "new_ids": new_ids,
        "history": history,
        "best": best,
    }

    out_json = resolve_path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved json: {out_json}")


if __name__ == "__main__":
    main()
