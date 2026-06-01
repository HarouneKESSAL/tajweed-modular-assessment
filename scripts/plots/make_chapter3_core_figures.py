from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/experiments/chapter3_core")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def plot_content_gate_development() -> None:
    # Replace values if your Tables 3.11, 3.12, 3.13 use slightly different numbers.
    phases = [
        "Phase 1\nFull-ayah CTC",
        "Phase 2\nChunked CTC",
        "Phase 3\nWhisper-v2",
    ]

    exact_match = [1.6, 73.8, 73.96]
    char_accuracy = [53.6, 80.4, 98.17]

    x = np.arange(len(phases))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    bars1 = plt.bar(x - width / 2, exact_match, width, label="Exact match / acceptance")
    bars2 = plt.bar(x + width / 2, char_accuracy, width, label="Character accuracy")

    plt.ylabel("Score (%)")
    plt.title("Content gate development progression")
    plt.xticks(x, phases)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    savefig("content_gate_development_progression")


def plot_tajweed_per_class_accuracy() -> None:
    labels = [
        "Duration: Ghunnah",
        "Duration: Madd",
        "Transition: None",
        "Transition: Ikhfa'",
        "Transition: Idgham",
        "Burst: None",
        "Burst: Qalqalah",
    ]

    accuracy = [98.4, 99.5, 89.6, 92.1, 85.7, 91.4, 84.7]

    y = np.arange(len(labels))

    plt.figure(figsize=(9, 5.2))
    bars = plt.barh(y, accuracy)

    plt.yticks(y, labels)
    plt.xlabel("Accuracy (%)")
    plt.title("Per-class accuracy across Tajweed modules")
    plt.xlim(0, 105)
    plt.grid(axis="x", alpha=0.3)
    plt.gca().invert_yaxis()

    for bar, value in zip(bars, accuracy):
        plt.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8,
        )

    savefig("tajweed_per_class_accuracy")


def plot_whisper_v1_v2_ablation() -> None:
    metrics = ["Exact\nacceptance", "Char.\naccuracy", "CER", "Errors"]
    v1 = [73.46, 98.03, 2.05, 108]
    v2 = [73.96, 98.17, 1.89, 106]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    bars1 = plt.bar(x - width / 2, v1, width, label="Whisper-medium v1")
    bars2 = plt.bar(x + width / 2, v2, width, label="Whisper-medium v2")

    plt.ylabel("Value")
    plt.title("Whisper content gate ablation: v1 versus v2")
    plt.xticks(x, metrics)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            if value >= 20:
                label = f"{value:.0f}" if value > 100 else f"{value:.2f}"
            else:
                label = f"{value:.2f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    savefig("whisper_v1_v2_ablation")


def plot_duration_conservative_vs_fusion() -> None:
    rules = ["Ghunnah", "Madd"]

    conservative = [74.5, 98.9]
    learned_fusion = [87.2, 99.5]

    x = np.arange(len(rules))
    width = 0.35

    plt.figure(figsize=(7.5, 4.6))
    bars1 = plt.bar(x - width / 2, conservative, width, label="Conservative baseline")
    bars2 = plt.bar(x + width / 2, learned_fusion, width, label="Learned-fusion baseline")

    plt.ylabel("Accuracy (%)")
    plt.title("Duration module: conservative versus learned-fusion baseline")
    plt.xticks(x, rules)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    savefig("duration_conservative_vs_fusion")


def plot_localiser_support_rates() -> None:
    labels = ["Ghunnah", "Madd", "Ikhfa'", "Idgham", "None"]

    # Replace totals if your tables contain different support counts.
    support_rate = [90.11, 99.22, 98.72, 89.74, 0.0]
    unsupported_rate = [100 - x for x in support_rate]

    x = np.arange(len(labels))

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x, support_rate, label="Supported")
    plt.bar(x, unsupported_rate, bottom=support_rate, label="Unsupported / not localised")

    plt.ylabel("Share (%)")
    plt.title("Localiser support rates by rule/class")
    plt.xticks(x, labels)
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for i, value in enumerate(support_rate):
        plt.text(
            i,
            min(value + 2, 98),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    savefig("localiser_support_rates")


def plot_training_corpus_distribution() -> None:
    corpora = [
        "Retasy raw recitations",
        "Content verification",
        "Rule curriculum",
        "Duration subset",
        "Transition subset",
        "Burst subset",
    ]

    counts = [6828, 1944, 15536, 1435, 690, 1597]

    y = np.arange(len(corpora))

    plt.figure(figsize=(9, 5))
    bars = plt.barh(y, counts)

    plt.yticks(y, corpora)
    plt.xlabel("Number of samples")
    plt.title("Training corpus distribution")
    plt.grid(axis="x", alpha=0.3)
    plt.gca().invert_yaxis()

    for bar, value in zip(bars, counts):
        plt.text(
            value + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=8,
        )

    savefig("training_corpus_distribution")


def main() -> None:
    plot_content_gate_development()
    plot_tajweed_per_class_accuracy()
    plot_whisper_v1_v2_ablation()
    plot_duration_conservative_vs_fusion()
    plot_localiser_support_rates()
    plot_training_corpus_distribution()


if __name__ == "__main__":
    main()