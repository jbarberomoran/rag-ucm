"""Result summaries and plots for completed experiment observations."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def evaluate_results(df: pd.DataFrame, ff: str):
    """Print accuracy over valid observations for each retrieval method."""
    print("\n" + "=" * 30)
    print("ACCURACY OVER VALID RESPONSES (see summary.json for all attempts)")
    print("=" * 30)
    print(df.groupby("method")["correct"].mean() * 100)
    print(f"Final results: {ff}")


def load_data(dir_input: str):
    """Load valid observations and normalize the correctness column."""
    if not os.path.exists(dir_input):
        print(f"Results file not found: {dir_input}")
        print("Run main.py to generate experiment data first.")
        return None

    df = pd.read_csv(dir_input)
    if "error" in df:
        df = df[df["error"].fillna("") == ""].copy()

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(int)

    return df


def setup_plot_style():
    """Configure the shared plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.autolayout": True})


def clean_emojis(text):
    """Remove legacy status icons that can trigger font warnings."""
    if not isinstance(text, str):
        return text
    return (
        text.replace("✅", "")
        .replace("⚠️", "")
        .replace("📉", "")
        .replace("❌", "")
        .strip()
    )


def plot_accuracy(df, dir_output: str):
    """Plot answer accuracy by retrieval method."""
    if "correct" not in df.columns:
        print("Column 'correct' not found; skipping the accuracy plot.")
        return

    plt.figure(figsize=(10, 6))
    accuracy = (df.groupby("method")["correct"].mean() * 100).reset_index()
    barplot = sns.barplot(
        x="method",
        y="correct",
        hue="method",
        data=accuracy,
        palette="viridis",
        edgecolor="black",
        legend=False,
    )

    for patch in barplot.patches:
        barplot.annotate(
            f"{patch.get_height():.1f}%",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.title("Answer accuracy by method", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Retrieval method")
    plt.ylim(0, 115)

    save_path = os.path.join(dir_output, "1_accuracy.png")
    plt.savefig(save_path, dpi=300)
    print("Saved plot 1: accuracy")
    plt.close()


def plot_rag_quality(df, dir_output: str):
    """Plot answer correctness against detected reference overlap."""
    if "status" not in df.columns:
        return

    df = df.copy()
    df["status_clean"] = df["status"].apply(clean_emojis)
    plt.figure(figsize=(12, 7))

    counts = df.groupby(["method", "status_clean"]).size().reset_index(name="count")
    totals = df.groupby("method").size().reset_index(name="total")
    percentages = pd.merge(counts, totals, on="method")
    percentages["percentage"] = percentages["count"] / percentages["total"] * 100

    status_palette = {
        "Correct / reference overlap detected": "#2ecc71",
        "Correct / reference overlap not detected": "#f1c40f",
        "Incorrect / reference overlap detected": "#e67e22",
        "Incorrect / reference overlap not detected": "#e74c3c",
    }
    palette = {
        status: status_palette.get(status, "#95a5a6")
        for status in percentages["status_clean"].unique()
    }
    barplot = sns.barplot(
        data=percentages,
        x="method",
        y="percentage",
        hue="status_clean",
        palette=palette,
        edgecolor="black",
    )

    for patch in barplot.patches:
        height = patch.get_height()
        if height > 0:
            barplot.annotate(
                f"{height:.1f}%",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.title("Answer correctness and reference overlap", fontsize=14, fontweight="bold")
    plt.ylabel("Share of observations (%)")
    plt.xlabel("Retrieval method")
    plt.legend(title="Diagnostic", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.ylim(0, 110)
    plt.savefig(
        os.path.join(dir_output, "2_rag_quality_pct.png"), dpi=300, bbox_inches="tight"
    )
    print("Saved plot 2: answer correctness and reference overlap")
    plt.close()


def plot_latency(df, dir_output: str):
    """Plot end-to-end response-time distributions by method."""
    if "response_time" not in df.columns:
        print("Column 'response_time' not found; skipping the latency plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="method",
        y="response_time",
        hue="method",
        palette="pastel",
        showfliers=True,
        legend=False,
    )

    plt.title("End-to-end response time by method", fontsize=14)
    plt.ylabel("Seconds")
    plt.xlabel("Retrieval method")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    save_path = os.path.join(dir_output, "3_latency.png")
    plt.savefig(save_path, dpi=300)
    print("Saved plot 3: latency")
    plt.close()


def plot_retrieval_score(df, dir_output: str):
    """Plot lexical reference-overlap scores for retrieval methods."""
    if "retrieval_score" not in df.columns:
        print("Column 'retrieval_score' not found; skipping the overlap plot.")
        return
    df = df[df["method"] != "baseline"]
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="method",
        y="retrieval_score",
        hue="method",
        palette="Set3",
        inner="quartile",
        legend=False,
    )

    plt.title("Reference text overlap (not faithfulness)", fontsize=14)
    plt.ylabel("Longest matching span / reference length (0–1)")
    plt.xlabel("Retrieval method")
    plt.ylim(-0.1, 1.1)

    save_path = os.path.join(dir_output, "4_retrieval_fidelity.png")
    plt.savefig(save_path, dpi=300)
    print("Saved plot 4: reference overlap")
    plt.close()


def generate_dashboard(dir_input, dir_output: str):
    """Generate all plots for a completed experiment file."""
    print(f"\nGenerating plots from: {dir_input}")
    if not os.path.exists(dir_output):
        print(f"Creating plot directory: {dir_output}")
        os.makedirs(dir_output)

    df = load_data(dir_input)
    if df is None:
        return

    setup_plot_style()
    try:
        plot_accuracy(df, dir_output)
        plot_rag_quality(df, dir_output)
        plot_latency(df, dir_output)
        plot_retrieval_score(df, dir_output)
        print(f"\nPlots generated in: {os.path.abspath(dir_output)}")
    except Exception as error:
        print(f"Unable to generate plots: {error}")
