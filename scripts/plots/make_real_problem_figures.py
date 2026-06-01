from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("figures/experiments/real_problems")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_CSV = Path("data/analysis/content_asr_labeled_model_comparison.csv")
UPLOADS_CSV = Path("data/analysis/asr_uploads_model_comparison.csv")


def savefig(name: str) -> None:
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"

    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def short_model_name(model: str) -> str:
    model = str(model)

    if "content_asr_whisper_medium" in model:
        return "Default\nWhisper-medium"
    if "tarteel-ai/whisper-base" in model:
        return "Tarteel\nWhisper-base"
    if "IJyad/whisper-large-v3" in model:
        return "IJyad\nWhisper-large-v3"

    return model.replace("/", "\n")


def load_labeled() -> pd.DataFrame:
    if not LABELED_CSV.exists():
        raise FileNotFoundError(f"Missing file: {LABELED_CSV}")

    df = pd.read_csv(LABELED_CSV)

    required = {
        "model",
        "audio_path",
        "cer",
        "char_accuracy",
        "exact",
        "accepted_segment",
        "transcription_time_sec",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{LABELED_CSV} is missing columns: {sorted(missing)}")

    df["model_short"] = df["model"].apply(short_model_name)
    df["cer_percent"] = df["cer"].astype(float) * 100
    df["char_accuracy_percent"] = df["char_accuracy"].astype(float) * 100
    df["accepted_segment"] = df["accepted_segment"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["exact"] = df["exact"].astype(str).str.lower().isin(["true", "1", "yes"])

    return df


def plot_model_acceptance_and_cer() -> None:
    df = load_labeled()

    summary = (
        df.groupby(["model", "model_short"], as_index=False)
        .agg(
            accepted_rate=("accepted_segment", "mean"),
            exact_rate=("exact", "mean"),
            avg_cer=("cer_percent", "mean"),
            avg_time=("transcription_time_sec", "mean"),
        )
    )

    summary["accepted_rate"] *= 100
    summary["exact_rate"] *= 100

    x = np.arange(len(summary))
    width = 0.28

    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width, summary["accepted_rate"], width, label="Accepted segments")
    plt.bar(x, summary["exact_rate"], width, label="Exact matches")
    plt.bar(x + width, summary["avg_cer"], width, label="Average CER")

    plt.xticks(x, summary["model_short"])
    plt.ylabel("Percentage (%)")
    plt.title("Labeled content-ASR comparison: acceptance, exactness, and CER")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for i, row in summary.iterrows():
        plt.text(i - width, row["accepted_rate"] + 1, f"{row['accepted_rate']:.1f}%", ha="center", fontsize=8)
        plt.text(i, row["exact_rate"] + 1, f"{row['exact_rate']:.1f}%", ha="center", fontsize=8)
        plt.text(i + width, row["avg_cer"] + 1, f"{row['avg_cer']:.1f}%", ha="center", fontsize=8)

    savefig("real_asr_acceptance_exact_cer")


def plot_speed_accuracy_tradeoff() -> None:
    df = load_labeled()

    summary = (
        df.groupby(["model", "model_short"], as_index=False)
        .agg(
            avg_cer=("cer_percent", "mean"),
            accepted_rate=("accepted_segment", "mean"),
            avg_time=("transcription_time_sec", "mean"),
        )
    )

    summary["accepted_rate"] *= 100

    plt.figure(figsize=(8, 5))

    for _, row in summary.iterrows():
        plt.scatter(row["avg_time"], row["avg_cer"], s=140)
        plt.text(
            row["avg_time"],
            row["avg_cer"] + 1.2,
            row["model_short"].replace("\n", " "),
            ha="center",
            fontsize=8,
        )

    plt.xlabel("Average transcription time per sample (s)")
    plt.ylabel("Average CER (%)")
    plt.title("ASR speed--error trade-off on labeled samples")
    plt.grid(alpha=0.3)

    savefig("real_asr_speed_error_tradeoff")


def plot_cer_distribution_by_model() -> None:
    df = load_labeled()

    models = list(df["model_short"].drop_duplicates())
    data = [df[df["model_short"] == m]["cer_percent"].values for m in models]

    plt.figure(figsize=(8.5, 4.8))
    plt.boxplot(data, tick_labels=models, showmeans=True)

    plt.ylabel("CER (%)")
    plt.title("Distribution of content recognition errors by ASR model")
    plt.grid(axis="y", alpha=0.3)

    savefig("real_asr_cer_distribution")


def plot_acceptance_breakdown() -> None:
    df = load_labeled()

    rows = []

    for (model, model_short), group in df.groupby(["model", "model_short"]):
        exact = int(group["exact"].sum())
        accepted_not_exact = int((group["accepted_segment"] & ~group["exact"]).sum())
        rejected = int((~group["accepted_segment"]).sum())

        rows.append(
            {
                "model": model,
                "model_short": model_short,
                "exact": exact,
                "accepted_not_exact": accepted_not_exact,
                "rejected": rejected,
            }
        )

    summary = pd.DataFrame(rows)
    x = np.arange(len(summary))

    plt.figure(figsize=(8.5, 4.8))

    plt.bar(x, summary["exact"], label="Exact accepted")
    plt.bar(
        x,
        summary["accepted_not_exact"],
        bottom=summary["exact"],
        label="Accepted by tolerance",
    )
    plt.bar(
        x,
        summary["rejected"],
        bottom=summary["exact"] + summary["accepted_not_exact"],
        label="Rejected",
    )

    plt.xticks(x, summary["model_short"])
    plt.ylabel("Number of labeled samples")
    plt.title("Content gate outcome breakdown by ASR model")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    savefig("real_asr_acceptance_breakdown")


def plot_hardest_samples_for_default_model(top_n: int = 10) -> None:
    df = load_labeled()

    default_df = df[df["model"].str.contains("content_asr_whisper_medium", regex=False)].copy()

    if default_df.empty:
        print("Skipped hardest samples: default model rows not found.")
        return

    hard = default_df.sort_values("cer_percent", ascending=False).head(top_n).copy()

    hard["sample"] = hard["audio_path"].apply(lambda p: Path(str(p)).name.replace(".wav", ""))

    plt.figure(figsize=(9, 5))
    plt.barh(hard["sample"], hard["cer_percent"])
    plt.gca().invert_yaxis()

    plt.xlabel("CER (%)")
    plt.ylabel("Audio sample")
    plt.title("Hardest labeled samples for the default content model")
    plt.grid(axis="x", alpha=0.3)

    savefig("real_default_model_hardest_samples")


def plot_uploads_speed_summary() -> None:
    if not UPLOADS_CSV.exists():
        print(f"Skipped uploads speed summary: missing {UPLOADS_CSV}")
        return

    df = pd.read_csv(UPLOADS_CSV)

    required = {"model", "transcription_time_sec", "speed_ratio", "status"}
    missing = required - set(df.columns)

    if missing:
        print(f"Skipped uploads speed summary: missing columns {sorted(missing)}")
        return

    df = df[df["status"] == "ok"].copy()
    df["model_short"] = df["model"].apply(short_model_name)

    summary = (
        df.groupby(["model", "model_short"], as_index=False)
        .agg(
            avg_time=("transcription_time_sec", "mean"),
            avg_speed_ratio=("speed_ratio", "mean"),
            files=("audio_path", "count"),
        )
    )

    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width / 2, summary["avg_time"], width, label="Avg. time per file")
    plt.bar(x + width / 2, summary["avg_speed_ratio"], width, label="Speed ratio")

    plt.xticks(x, summary["model_short"])
    plt.ylabel("Seconds / ratio")
    plt.title("Runtime behaviour on all uploaded audio files")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    savefig("real_uploads_runtime_summary")


def main() -> None:
    plot_model_acceptance_and_cer()
    plot_speed_accuracy_tradeoff()
    plot_cer_distribution_by_model()
    plot_acceptance_breakdown()
    plot_hardest_samples_for_default_model()
    plot_uploads_speed_summary()


if __name__ == "__main__":
    main()