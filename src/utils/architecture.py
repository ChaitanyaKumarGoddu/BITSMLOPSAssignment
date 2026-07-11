"""Generate the architecture diagram used in the report (Task 9).

Draws a labelled block diagram of the pipeline with matplotlib (no external
tools like Graphviz needed) and saves it to reports/figures/architecture.png.

Run:  python -m src.utils.architecture
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.config import FIGURES_DIR, ensure_dirs  # noqa: E402


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#333",
            facecolor=color,
            alpha=0.95,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
        fontweight="bold",
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#444",
        )
    )


def main() -> None:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    blue, green, orange, purple, red = (
        "#cfe2f3",
        "#d9ead3",
        "#fce5cd",
        "#e6d0f0",
        "#f4cccc",
    )

    # Row 1 — development / training
    _box(ax, 0.3, 7.3, 2.2, 1.1, "UCI Heart Disease\n(download + clean)", blue)
    _box(ax, 2.9, 7.3, 2.2, 1.1, "EDA\n(histograms, heatmap,\nclass balance)", blue)
    _box(ax, 5.5, 7.3, 2.6, 1.1, "Train LR / RF / XGB\nGridSearchCV + 5-fold CV", green)
    _box(
        ax,
        8.6,
        7.3,
        3.0,
        1.1,
        "MLflow tracking\nparams, metrics, ROC,\nconfusion matrix, model",
        purple,
    )

    _arrow(ax, 2.5, 7.85, 2.9, 7.85)
    _arrow(ax, 5.1, 7.85, 5.5, 7.85)
    _arrow(ax, 8.1, 7.85, 8.6, 7.85)

    # Down to model artifact
    _box(ax, 5.5, 5.6, 2.6, 1.0, "model.pkl\n(sklearn Pipeline)", green)
    _arrow(ax, 6.8, 7.3, 6.8, 6.6)

    # Row 2 — CI/CD
    _box(
        ax,
        0.3,
        5.5,
        4.6,
        1.2,
        "GitHub Actions CI/CD\nlint → pytest → train → docker build → smoke test",
        orange,
    )
    _arrow(ax, 4.9, 6.1, 5.5, 6.1)

    # Docker image
    _box(
        ax,
        4.6,
        3.7,
        4.0,
        1.1,
        "Docker image — FastAPI\n/predict  /health  /metrics  /docs",
        red,
    )
    _arrow(ax, 6.8, 5.6, 6.8, 4.8)

    # Kubernetes
    _box(
        ax,
        4.6,
        1.8,
        4.0,
        1.1,
        "Kubernetes (Docker Desktop)\nDeployment ×2 + LoadBalancer",
        blue,
    )
    _arrow(ax, 6.6, 3.7, 6.6, 2.9)

    # Monitoring
    _box(ax, 9.2, 3.5, 2.5, 1.1, "Prometheus\nscrape /metrics", orange)
    _box(ax, 9.2, 1.8, 2.5, 1.1, "Grafana\ndashboards", green)
    _arrow(ax, 8.6, 4.25, 9.2, 4.05)
    _arrow(ax, 10.45, 3.5, 10.45, 2.9)
    _arrow(ax, 8.6, 2.35, 9.2, 2.35)

    # Client
    _box(ax, 0.3, 1.8, 3.2, 1.1, "Client\n(curl / Postman / Swagger)", purple)
    _arrow(ax, 3.5, 2.35, 4.6, 2.35)

    ax.set_title(
        "Heart Disease MLOps — System Architecture",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    out = FIGURES_DIR / "architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[architecture] Saved -> {out}")


if __name__ == "__main__":
    main()
