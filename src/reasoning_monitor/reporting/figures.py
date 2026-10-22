"""Figure generation.

Publication defaults: vector PDF alongside raster PNG, no seaborn dependency,
a colour-blind-safe palette, and no reliance on colour alone to distinguish
series (line style varies too). Every figure is written by a function that
takes a DataFrame and a path, so the pipeline can regenerate all of them from
saved results without re-running any model calls.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in CI or on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
PALETTE = {
    "CQ": "#0072B2", "CS": "#E69F00", "RS": "#009E73",
    "LS": "#CC79A7", "ES": "#D55E00", "SS": "#56B4E9",
}
LINESTYLES = {
    "CQ": "-", "CS": "--", "RS": "-.", "LS": ":", "ES": (0, (3, 1, 1, 1)), "SS": (0, (5, 1)),
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path.with_suffix(".png")


def plot_metric_traces(
    df: pd.DataFrame,
    change_point: int,
    out: str | Path,
    title: str = "Reasoning quality over the query stream",
) -> Path:
    """All six metric traces with the ground-truth change point marked."""
    metrics = [m for m in PALETTE if m in df.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in metrics:
        ax.plot(df["t"], df[m], label=m, color=PALETTE[m],
                linestyle=LINESTYLES[m], linewidth=1.6)
    if change_point > 0:
        ax.axvline(change_point, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.annotate("drift onset", xy=(change_point, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=9)
    ax.set_xlabel("Query index $t$")
    ax.set_ylabel("Metric value (window mean)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title)
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0, -0.28))
    return _save(fig, out)


def plot_composite_with_alarms(
    df: pd.DataFrame,
    alarms: pd.DataFrame,
    change_point: int,
    out: str | Path,
) -> Path:
    """Composite Q(t) with detector alarms overlaid as markers."""
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(df["t"], df["composite"], color="#333333", linewidth=1.8, label="Composite $Q(t)$")
    if change_point > 0:
        ax.axvline(change_point, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    if not alarms.empty:
        for i, (metric, grp) in enumerate(alarms.groupby("metric")):
            y = df["composite"].max() * (0.95 - 0.04 * i)
            ax.scatter(grp["t"], [y] * len(grp), s=36,
                       color=PALETTE.get(metric, "#666666"),
                       marker="v", label=f"{metric} alarm", zorder=5)
    ax.set_xlabel("Query index $t$")
    ax.set_ylabel("Composite quality")
    ax.set_title("Composite quality and drift alarms")
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.32))
    return _save(fig, out)


def plot_detection_delay(delays: pd.DataFrame, out: str | Path) -> Path:
    """Mean detection delay per metric — the early-warning ranking."""
    d = delays.dropna(subset=["mean_delay"]).sort_values("mean_delay")
    fig, ax = plt.subplots(figsize=(7, 4.0))
    colors = [PALETTE.get(m, "#666666") for m in d["metric"]]
    bars = ax.barh(d["metric"], d["mean_delay"], color=colors, alpha=0.85)
    for bar, rate in zip(bars, d["detection_rate"], strict=True):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"det. {rate:.0%}", va="center", fontsize=8)
    ax.set_xlabel("Mean detection delay (queries)")
    ax.set_title("Which dimension warns first?")
    ax.invert_yaxis()
    return _save(fig, out)


def plot_scenario_heatmap(
    matrix: pd.DataFrame, out: str | Path, title: str, cbar_label: str
) -> Path:
    """Metric x scenario heatmap (delay or effect size)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < np.nanmean(data) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    return _save(fig, out)


def plot_window_ablation(df: pd.DataFrame, out: str | Path) -> Path:
    """Effect of the sliding-window size W on detection delay."""
    fig, ax = plt.subplots(figsize=(7, 4.0))
    for metric, grp in df.groupby("metric"):
        grp = grp.sort_values("window_size")
        ax.plot(grp["window_size"], grp["mean_delay"], marker="o",
                color=PALETTE.get(metric, "#666666"),
                linestyle=LINESTYLES.get(metric, "-"), label=metric)
    ax.set_xlabel("Sliding-window size $W$")
    ax.set_ylabel("Mean detection delay (queries)")
    ax.set_title("Sensitivity to window size")
    ax.legend(ncol=3)
    return _save(fig, out)


def plot_model_comparison(df: pd.DataFrame, out: str | Path) -> Path:
    """Composite quality before vs after drift, by model."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    models = df["model"].unique()
    x = np.arange(len(models))
    w = 0.38
    before = [df[df["model"] == m]["q_pre"].mean() for m in models]
    after = [df[df["model"] == m]["q_post"].mean() for m in models]
    ax.bar(x - w / 2, before, w, label="Pre-drift", color="#0072B2", alpha=0.9)
    ax.bar(x + w / 2, after, w, label="Post-drift", color="#D55E00", alpha=0.9)
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.set_ylabel("Composite quality")
    ax.set_title("Quality degradation by model")
    ax.legend()
    return _save(fig, out)
