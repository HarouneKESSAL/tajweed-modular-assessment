from dataclasses import dataclass

from tajweed_assessment.text.normalization import normalize_arabic_text


@dataclass(frozen=True)
class ContentMetrics:
    exact_match: float
    normalized_exact_match: float
    char_accuracy: float
    normalized_char_accuracy: float
    mean_edit_distance: float
    normalized_mean_edit_distance: float
    samples: int


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
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def char_accuracy(pred: str, target: str) -> float:
    if not target:
        return 1.0 if not pred else 0.0
    dist = levenshtein(pred, target)
    return max(0.0, 1.0 - dist / len(target))


def compute_content_metrics(predictions: list[str], targets: list[str]) -> ContentMetrics:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")

    n = len(targets)
    if n == 0:
        return ContentMetrics(0, 0, 0, 0, 0, 0, 0)

    exact = 0
    norm_exact = 0
    char_scores = []
    norm_char_scores = []
    distances = []
    norm_distances = []

    for pred, target in zip(predictions, targets):
        exact += int(pred == target)
        distances.append(levenshtein(pred, target))
        char_scores.append(char_accuracy(pred, target))

        pred_n = normalize_arabic_text(pred)
        target_n = normalize_arabic_text(target)

        norm_exact += int(pred_n == target_n)
        norm_distances.append(levenshtein(pred_n, target_n))
        norm_char_scores.append(char_accuracy(pred_n, target_n))

    return ContentMetrics(
        exact_match=exact / n,
        normalized_exact_match=norm_exact / n,
        char_accuracy=sum(char_scores) / n,
        normalized_char_accuracy=sum(norm_char_scores) / n,
        mean_edit_distance=sum(distances) / n,
        normalized_mean_edit_distance=sum(norm_distances) / n,
        samples=n,
    )