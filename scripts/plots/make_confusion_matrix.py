from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--true-col", default="gold")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Confusion matrix")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.true_col not in df.columns:
        raise SystemExit(f"Missing true column: {args.true_col}")

    if args.pred_col not in df.columns:
        raise SystemExit(f"Missing pred column: {args.pred_col}")

    y_true = df[args.true_col].astype(str)
    y_pred = df[args.pred_col].astype(str)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=args.labels,
        normalize="true" if args.normalize else None,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=args.labels,
    )

    disp.plot(
        ax=ax,
        values_format=".2f" if args.normalize else "d",
        xticks_rotation=30,
        colorbar=True,
    )

    ax.set_title(args.title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = args.output.with_suffix(".pdf")
    png_path = args.output.with_suffix(".png")

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()