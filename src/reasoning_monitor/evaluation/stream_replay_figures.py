"""PNG/PDF figures for the LIC-RQ-v2-SR session-stream replay."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reasoning_monitor.evaluation import stream_replay as sr  # noqa: E402

# Okabe-Ito (colour-blind safe), fixed order: identity follows the metric.
METRIC_COLORS = {"CQ": "#000000", "CS": "#E69F00", "RS": "#009E73",
                 "LS": "#CC79A7", "SS": "#0072B2"}
STRATEGY_COLORS = {"distribution_based": "#0072B2", "proposal_mapping": "#D55E00",
                   "adwin_all": "#7F7F7F"}
UNOBSERVABLE_NOTE = "ES and RQ6_mean are unobservable in LIC-RQ-v2 and are not monitored."
LABEL = "Post-hoc replay of 288 observed sessions (method demonstration, not confirmatory)"


def _save(fig, target: Path, stem: str) -> list[str]:
    names = []
    for extension in ("png", "pdf"):
        path = target / f"{stem}.{extension}"
        fig.savefig(path, dpi=160)
        names.append(path.name)
    plt.close(fig)
    return names


def _change_lines(axis, scenario: str) -> None:
    base = sr.SCENARIOS[scenario]
    axis.axvline(base.t_s, color="#555555", linestyle="--", linewidth=1)
    if base.t_star != base.t_s:
        axis.axvline(base.t_star, color="#555555", linestyle=":", linewidth=1)


def _trajectories(bands: pd.DataFrame, tables, models, scenario, window, target):
    fig, axes = plt.subplots(len(sr.METRICS), len(models), figsize=(13, 13),
                             sharex=True, constrained_layout=True)
    for col, model in enumerate(models):
        table = tables[model]
        n = table.n_tasks
        for row, metric in enumerate(sr.METRICS):
            axis = axes[row, col]
            part = bands[(bands["model"] == model) & (bands["scenario"] == scenario)
                         & (bands["window"] == window) & (bands["metric"] == metric)]
            part = part.sort_values("t")
            color = METRIC_COLORS[metric]
            axis.fill_between(part["t"], part["p10"].astype(float), part["p90"].astype(float),
                              color=color, alpha=0.18, linewidth=0)
            axis.plot(part["t"], part["median"].astype(float), color=color, linewidth=2)
            _change_lines(axis, scenario)
            values = table.values[metric]
            fresh_valid = int(np.isfinite(values[:n]).sum())
            dialogue_valid = int(np.isfinite(values[n:]).sum())
            axis.text(0.01, 0.03, f"valid sessions: fresh {fresh_valid}/{n}, "
                      f"dialogue {dialogue_valid}/{n}", transform=axis.transAxes,
                      fontsize=7, color="#333333")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(model)
            if col == 0:
                axis.set_ylabel(f"{metric} (W={window} mean)")
            if row == len(sr.METRICS) - 1:
                axis.set_xlabel("Session index in stream")
    title = {"sudden": "LIC-S sudden", "gradual": "LIC-G gradual"}[scenario]
    fig.suptitle(f"{title} context-accumulation drift, W={window}: median and 10–90% band "
                 f"over {sr.N_ORDERED} orderings\n{LABEL}. {UNOBSERVABLE_NOTE}", fontsize=10)
    return _save(fig, target, f"trajectories_{scenario}_W{window}")


def _first_alarms(rows: pd.DataFrame, detection: pd.DataFrame, models, scenario, window,
                  target):
    strategies = sr.strategy_detectors()
    fig, axes = plt.subplots(len(sr.STRATEGIES), len(models), figsize=(14, 9),
                             sharex=True, constrained_layout=True)
    for r, strategy in enumerate(sr.STRATEGIES):
        for c, model in enumerate(models):
            axis = axes[r, c]
            for position, metric in enumerate(sr.METRICS):
                detector = strategies[strategy][metric]
                part = rows[(rows["model"] == model) & (rows["scenario"] == scenario)
                            & (rows["window"] == window) & (rows["metric"] == metric)
                            & (rows["detector"] == detector)]
                y = len(sr.METRICS) - position
                if part.empty:
                    axis.text(2, y, f"{metric}/{detector}: no validated detector",
                              va="center", fontsize=7, color="#555555")
                    continue
                times = part["first_post_ts"].dropna().astype(float)
                if len(times):
                    axis.boxplot(times, positions=[y], vert=False, widths=0.5,
                                 patch_artist=True, showfliers=False,
                                 boxprops={"facecolor": METRIC_COLORS[metric], "alpha": 0.35},
                                 medianprops={"color": METRIC_COLORS[metric]})
                rate = detection[(detection["model"] == model)
                                 & (detection["scenario"] == scenario)
                                 & (detection["strategy"] == strategy)
                                 & (detection["window"] == window)
                                 & (detection["metric"] == metric)]["detection_rate"]
                text = f"det={float(rate.iloc[0]):.2f}" if len(rate) else ""
                axis.text(97, y, text, va="center", fontsize=7)
            _change_lines(axis, scenario)
            axis.set_yticks(range(len(sr.METRICS), 0, -1),
                            [f"{m}/{strategies[strategy][m]}" for m in sr.METRICS], fontsize=7)
            axis.set_xlim(0, 110)
            axis.set_ylim(0.4, len(sr.METRICS) + 0.6)
            axis.grid(axis="x", alpha=0.2)
            if r == 0:
                axis.set_title(model)
            if c == 0:
                axis.set_ylabel(strategy)
            if r == len(sr.STRATEGIES) - 1:
                axis.set_xlabel("First alarm at or after t_s (session index)")
    fig.suptitle(f"First post-change alarm times, {scenario}, W={window}. det = detection "
                 f"rate (first alarm at or after t_s)\n{LABEL}", fontsize=10)
    return _save(fig, target, f"first_alarms_{scenario}_W{window}")


def _h2(detection: pd.DataFrame, models, scenario, window, target):
    fig, axes = plt.subplots(1, len(models), figsize=(14, 4.2), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(sr.METRICS))
    width = 0.26
    for c, model in enumerate(models):
        axis = axes[c]
        for k, strategy in enumerate(sr.STRATEGIES):
            heights = []
            for metric in sr.METRICS:
                rate = detection[(detection["model"] == model)
                                 & (detection["scenario"] == scenario)
                                 & (detection["strategy"] == strategy)
                                 & (detection["window"] == window)
                                 & (detection["metric"] == metric)]["detection_rate"]
                heights.append(float(rate.iloc[0]) if len(rate) else np.nan)
            axis.bar(x + (k - 1) * width, np.nan_to_num(heights), width * 0.92,
                     color=STRATEGY_COLORS[strategy], label=strategy)
            for xi, h in zip(x + (k - 1) * width, heights, strict=True):
                if np.isnan(h):
                    axis.text(xi, 0.02, "n/v", ha="center", fontsize=6, rotation=90)
        axis.set_xticks(x, sr.METRICS)
        axis.set_ylim(0, 1.05)
        axis.set_title(model)
        axis.grid(axis="y", alpha=0.2)
        if c == 0:
            axis.set_ylabel("Detection rate")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"H2 strategy comparison, {scenario}, W={window} (n/v = no validated "
                 f"detector)\n{LABEL}", fontsize=10)
    return _save(fig, target, f"h2_detection_{scenario}_W{window}")


def _ablation(detection: pd.DataFrame, models, scenario, strategy, target):
    fig, axes = plt.subplots(2, len(models), figsize=(14, 6.5), sharex=True,
                             constrained_layout=True)
    for c, model in enumerate(models):
        for metric in sr.METRICS:
            part = detection[(detection["model"] == model)
                             & (detection["scenario"] == scenario)
                             & (detection["strategy"] == strategy)
                             & (detection["metric"] == metric)].sort_values("window")
            if part.empty:
                continue
            style = {"color": METRIC_COLORS[metric], "marker": "o", "linewidth": 2,
                     "markersize": 6, "label": metric}
            axes[0, c].plot(part["window"], part["detection_rate"], **style)
            axes[1, c].plot(part["window"], part["median_signed_delay"].astype(float), **style)
        axes[0, c].set_title(model)
        axes[0, c].set_ylim(-0.03, 1.03)
        axes[1, c].axhline(0, color="#555555", linewidth=1)
        axes[1, c].set_xlabel("Window W")
        axes[1, c].set_xticks(list(sr.WINDOWS))
        for axis in axes[:, c]:
            axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Detection rate")
    axes[1, 0].set_ylabel("Median signed delay D* = t_alarm − t*")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"W ablation, {scenario}, strategy {strategy}. Metrics without a validated "
                 f"detector at a W are absent.\n{LABEL}", fontsize=10)
    return _save(fig, target, f"w_ablation_{scenario}_{strategy}")


def write_figures(result: dict, summary: dict, tables, target: Path) -> list[str]:
    target = Path(target) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    models = result["models"]
    detection = summary.get("detection", pd.DataFrame())
    names: list[str] = []
    for scenario in ("sudden", "gradual"):
        for window in sr.WINDOWS:
            names += _trajectories(result["bands"], tables, models, scenario, window, target)
        if detection.empty:
            continue
        names += _first_alarms(result["rows"], detection, models, scenario,
                               sr.PRIMARY_WINDOW, target)
        names += _h2(detection, models, scenario, sr.PRIMARY_WINDOW, target)
        for strategy in sr.STRATEGIES:
            names += _ablation(detection, models, scenario, strategy, target)
    return names
