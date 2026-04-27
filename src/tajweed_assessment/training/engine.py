from dataclasses import dataclass
import torch
from tajweed_assessment.models.common.losses import DurationLoss
from tajweed_assessment.training.metrics import (
    classification_accuracy,
    masked_classification_accuracy,
    phoneme_accuracy_from_log_probs,
)

@dataclass
class TrainResult:
    loss: float
    ctc_loss: float | None = None
    rule_loss: float | None = None
    accuracy: float | None = None
    phoneme_accuracy: float | None = None
    rule_accuracy: float | None = None

def train_duration_epoch(model, loader, optimizer, loss_fn: DurationLoss, device: str = "cpu") -> TrainResult:
    model.train()
    total_loss = total_ctc = total_rule = total_acc = total_rule_acc = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        phoneme_targets = batch["phoneme_targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        rule_targets = batch["rule_targets"].to(device)

        optimizer.zero_grad()
        log_probs, rule_logits = model(x, input_lengths)
        losses = loss_fn(
            log_probs,
            rule_logits,
            phoneme_targets,
            input_lengths,
            target_lengths,
            rule_targets,
        )
        losses.total.backward()
        optimizer.step()

        total_loss += float(losses.total.item())
        total_ctc += float(losses.ctc.item())
        total_rule += float(losses.rule.item())
        total_acc += phoneme_accuracy_from_log_probs(
            log_probs.detach().cpu(),
            batch["input_lengths"],
            batch["raw_phoneme_targets"],
        )
        total_rule_acc += masked_classification_accuracy(
            rule_logits.detach().cpu(),
            batch["rule_targets"],
        )

    n = len(loader)
    return TrainResult(
        loss=total_loss / n,
        ctc_loss=total_ctc / n,
        rule_loss=total_rule / n,
        phoneme_accuracy=total_acc / n,
        rule_accuracy=total_rule_acc / n,
    )

@torch.no_grad()
def evaluate_duration_epoch(model, loader, loss_fn: DurationLoss, device: str = "cpu") -> TrainResult:
    model.eval()
    total_loss = total_ctc = total_rule = total_acc = total_rule_acc = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        phoneme_targets = batch["phoneme_targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        rule_targets = batch["rule_targets"].to(device)

        log_probs, rule_logits = model(x, input_lengths)
        losses = loss_fn(
            log_probs,
            rule_logits,
            phoneme_targets,
            input_lengths,
            target_lengths,
            rule_targets,
        )

        total_loss += float(losses.total.item())
        total_ctc += float(losses.ctc.item())
        total_rule += float(losses.rule.item())
        total_acc += phoneme_accuracy_from_log_probs(
            log_probs.detach().cpu(),
            batch["input_lengths"],
            batch["raw_phoneme_targets"],
        )
        total_rule_acc += masked_classification_accuracy(
            rule_logits.detach().cpu(),
            batch["rule_targets"],
        )

    n = len(loader)
    return TrainResult(
        loss=total_loss / n,
        ctc_loss=total_ctc / n,
        rule_loss=total_rule / n,
        phoneme_accuracy=total_acc / n,
        rule_accuracy=total_rule_acc / n,
    )

def train_classifier_epoch(model, loader, optimizer, forward_fn, device: str = "cpu", loss_fn=None) -> TrainResult:
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = total_acc = 0.0

    for batch in loader:
        optimizer.zero_grad()
        logits = forward_fn(model, batch, device)
        targets = batch["label"].to(device)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_acc += classification_accuracy(logits.detach().cpu(), batch["label"])

    n = len(loader)
    return TrainResult(loss=total_loss / n, accuracy=total_acc / n)

@torch.no_grad()
def evaluate_classifier_epoch(model, loader, forward_fn, device: str = "cpu", loss_fn=None) -> TrainResult:
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = total_acc = 0.0

    for batch in loader:
        logits = forward_fn(model, batch, device)
        targets = batch["label"].to(device)
        loss = loss_fn(logits, targets)

        total_loss += float(loss.item())
        total_acc += classification_accuracy(logits.detach().cpu(), batch["label"])

    n = len(loader)
    return TrainResult(loss=total_loss / n, accuracy=total_acc / n)
