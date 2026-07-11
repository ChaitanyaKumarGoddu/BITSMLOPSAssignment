"""Task 1 — Exploratory Data Analysis figure generator.

Produces the professional visualisations the assignment asks for and saves
them to ``reports/figures/``. The same analysis is presented narratively in
``notebooks/01_eda.ipynb``; this script exists so the figures can be
regenerated headlessly (e.g. in CI or from the Makefile) without a notebook.

Run:  python -m src.data.eda
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    FIGURES_DIR,
    NUMERIC_FEATURES,
    PROCESSED_CSV,
    TARGET,
    ensure_dirs,
)

sns.set_theme(style="whitegrid", palette="deep")


def _load() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        from src.data.download import download
        from src.data.preprocess import build_processed_csv

        download()
        build_processed_csv()
    return pd.read_csv(PROCESSED_CSV)


def plot_class_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    sns.barplot(
        x=["No disease", "Disease"],
        y=counts.values,
        ax=ax,
        hue=["No disease", "Disease"],
        legend=False,
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Balance (target)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_balance.png", dpi=120)
    plt.close(fig)


def plot_histograms(df: pd.DataFrame) -> None:
    axes = df[NUMERIC_FEATURES].hist(
        figsize=(12, 8), bins=20, color="#4c72b0", edgecolor="white"
    )
    fig = axes.flatten()[0].get_figure()
    fig.suptitle("Distributions of Numeric Features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_histograms.png", dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=120)
    plt.close(fig)


def plot_feature_relationships(df: pd.DataFrame) -> None:
    """Box/violin plots of key numeric features split by target."""
    key_feats = ["age", "thalach", "oldpeak", "chol"]
    fig, axes = plt.subplots(1, len(key_feats), figsize=(16, 4))
    label = df[TARGET].map({0: "No disease", 1: "Disease"})
    for ax, feat in zip(axes, key_feats):
        sns.boxplot(x=label, y=df[feat], ax=ax, hue=label, legend=False)
        ax.set_title(f"{feat} by target")
        ax.set_xlabel("")
    fig.suptitle("Feature Relationships with Target", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_feature_relationships.png", dpi=120)
    plt.close(fig)


def missing_value_report(df_raw: pd.DataFrame) -> pd.Series:
    """Return per-column missing counts from the *raw* frame."""
    return df_raw.isna().sum()


def main() -> None:
    ensure_dirs()
    df = _load()

    # Missing-value analysis is most meaningful on the raw frame.
    from src.config import RAW_CSV

    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV).replace("?", pd.NA)
        missing = missing_value_report(raw)
        print("[eda] Missing values per raw column:\n", missing[missing > 0])

    plot_class_balance(df)
    plot_histograms(df)
    plot_correlation_heatmap(df)
    plot_feature_relationships(df)
    print(f"[eda] Saved 4 EDA figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
