from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


OUT_DIR = Path("figures/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> None:
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def plot_final_module_performance() -> None:
    data = pd.DataFrame(
        {
            "component": [
                "Content gate\nexact acceptance",
                "Duration\nmodule",
                "Transition\nmodule",
                "Burst/Qalqalah\nmodule",
            ],
            "score": [73.96, 99.3, 90.1, 87.4],
        }
    )

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(data["component"], data["score"])

    plt.ylabel("Performance (%)")
    plt.ylim(0, 105)
    plt.title("Final performance by module")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, data["score"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure("final_module_performance")


def plot_content_gate_ablation() -> None:
    metrics = ["Exact acceptance", "Character accuracy", "CER"]
    whisper_v1 = [73.46, 98.03, 2.05]
    whisper_v2 = [73.96, 98.17, 1.89]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, whisper_v1, width, label="Whisper-medium v1")
    plt.bar(x + width / 2, whisper_v2, width, label="Whisper-medium v2")

    plt.ylabel("Percentage (%)")
    plt.title("Content gate ablation: v1 versus v2")
    plt.xticks(x, metrics, rotation=15, ha="right")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_figure("content_gate_ablation_v1_v2")


def plot_burst_threshold_sweep() -> None:
    data = pd.DataFrame(
        {
            "threshold": [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50],
            "accuracy": [86.16, 86.41, 86.54, 86.79, 87.04, 87.54, 87.41, 87.41, 87.41],
            "precision": [79.69, 80.40, 81.36, 82.04, 83.03, 84.27, 84.65, 85.44, 86.38],
            "recall": [87.79, 87.32, 86.07, 85.76, 84.98, 84.66, 83.72, 82.63, 81.38],
            "f1": [83.54, 83.72, 83.65, 83.86, 83.99, 84.47, 84.19, 84.01, 83.80],
        }
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(data["threshold"], data["accuracy"], marker="o", label="Accuracy")
    plt.plot(data["threshold"], data["precision"], marker="o", label="Precision")
    plt.plot(data["threshold"], data["recall"], marker="o", label="Recall")
    plt.plot(data["threshold"], data["f1"], marker="o", label="F1")

    plt.axvline(0.47, linestyle="--", linewidth=1)
    plt.text(0.471, 86.8, "Selected threshold = 0.47", fontsize=9)

    plt.xlabel("Decision threshold")
    plt.ylabel("Score (%)")
    plt.title("Burst/Qalqalah threshold sweep")
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("burst_threshold_sweep")


def plot_content_cer_histogram(csv_path: str) -> None:
    path = Path(csv_path)

    if not path.exists():
        print(f"Skipped CER histogram: file not found: {path}")
        return

    df = pd.read_csv(path)

    if "cer" not in df.columns:
        print(f"Skipped CER histogram: no 'cer' column in {path}")
        return

    cer_percent = df["cer"].astype(float) * 100

    plt.figure(figsize=(8, 4.5))
    plt.hist(cer_percent, bins=20, edgecolor="black")

    plt.xlabel("Character error rate (%)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of content-gate CER")
    plt.grid(axis="y", alpha=0.3)

    save_figure("content_gate_cer_histogram")


def plot_confusion_matrix_from_csv(
    csv_path: str,
    labels: list[str],
    output_name: str,
    true_col: str = "gold",
    pred_col: str = "pred",
) -> None:
    path = Path(csv_path)

    if not path.exists():
        print(f"Skipped confusion matrix: file not found: {path}")
        return

    df = pd.read_csv(path)

    if true_col not in df.columns or pred_col not in df.columns:
        print(
            f"Skipped confusion matrix: expected columns "
            f"'{true_col}' and '{pred_col}' in {path}"
        )
        return

    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format=".2f", xticks_rotation=30)

    plt.title(output_name.replace("_", " ").title())

    save_figure(output_name)


def main() -> None:
    plot_final_module_performance()
    plot_content_gate_ablation()
    plot_burst_threshold_sweep()

    # Optional: use this if you have a CSV with a 'cer' column.
    plot_content_cer_histogram(
        "data/analysis/content_asr_labeled_model_comparison.csv"
    )

    # Optional examples.
    # These require CSV files with columns: gold,pred
    #
    # plot_confusion_matrix_from_csv(
    #     csv_path="data/analysis/transition_predictions.csv",
    #     labels=["none", "ikhfa", "idgham"],
    #     output_name="transition_confusion_matrix",
    # )
    #
    # plot_confusion_matrix_from_csv(
    #     csv_path="data/analysis/burst_predictions.csv",
    #     labels=["none", "qalqalah"],
    #     output_name="burst_confusion_matrix",
    # )


if __name__ == "__main__":
    main()