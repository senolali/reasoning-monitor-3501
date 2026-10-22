"""Aggregated (6 dialogues per turn) monitoring of the GPT control arm (EXPLORATORY).

Rules: CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md, Part 2. Call-free.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor import cdr_stream_analysis as ca  # noqa: E402
from reasoning_monitor.evaluation.detector_development import (  # noqa: E402
    build_window_signal,
    first_alarms_for_grid,
)
from reasoning_monitor.metrics.rqeval_adapters import _ESMetric  # noqa: E402

BASE = ROOT / "results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini"
OUTPUT = BASE / "aggregated"
METRICS = ("CQ", "LS", "ES")
DETECTORS = ("cusum", "page_hinkley", "ewma", "adwin", "ddm")
WINDOWS = (10, 20, 30)
CHANGES = {"S1": (34, 68), "S2": (51,), "S3": (34,)}
N_NULL, SEED, ALPHA, SPAN, T = 2000, 20261010, 0.05, ca.DETECTION_SPAN, 100
COLORS = {"CQ": "#0072B2", "LS": "#CC79A7", "ES": "#56B4E9"}


def es_series(cq: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    out = np.full(len(cq), np.nan)
    for t in range(ca.ES_WINDOW - 1, len(cq)):
        window = slice(t - ca.ES_WINDOW + 1, t + 1)
        out[t] = _ESMetric().compute_per_instance(list(cq[window]), list(tokens[window]))[-1]
    return out


def aggregate(frame: pd.DataFrame, scenario: str) -> dict[str, np.ndarray]:
    part = frame[frame["scenario"] == scenario]
    return {m: part.groupby("turn")[m].mean().reindex(range(1, T + 1)).to_numpy()
            for m in METRICS}


def null_streams(frame: pd.DataFrame, rng) -> list[dict[str, np.ndarray]]:
    dialogues = [p.sort_values("turn") for _, p in frame[frame["scenario"] == "S0-E"]
                 .groupby("dialogue_id")]
    streams = []
    for _ in range(N_NULL):
        stacks = {m: [] for m in METRICS}
        for p in dialogues:
            order = rng.permutation(len(p))
            cq = p["CQ"].to_numpy(float)[order]
            stacks["CQ"].append(cq)
            stacks["LS"].append(p["LS"].to_numpy(float)[order])
            stacks["ES"].append(es_series(cq, p["primary_tokens"].to_numpy(float)[order]))
        # Dialogues missing final turns (not acquired at the pause) are NaN-padded.
        streams.append({m: np.nanmean(np.vstack([np.pad(x, (0, T - len(x)),
                                                        constant_values=np.nan) for x in v]),
                                      axis=0) for m, v in stacks.items()})
    return streams


def grid(values: np.ndarray, window: int, detector: str) -> dict[float, int | None]:
    signal = build_window_signal("CQ", values, [], [], list(range(T)), window)
    return first_alarms_for_grid(signal, ca.GRIDS[detector])


def classify(first, changes):
    if first is None:
        return "no_alarm", None, None
    if first < changes[0]:
        return "false_alarm", None, None
    for c in reversed(changes):
        if first >= c:
            return ("detection" if first <= c + SPAN else "late"), c, first - c
    return "late", None, None


def main() -> int:
    global OUTPUT, DETECTORS, BASE
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--model", default="gpt4o_mini")
    parser.add_argument("--source", default="gpt_screen")
    args = parser.parse_args()
    DETECTORS = tuple(args.detectors)
    BASE = ROOT / "results/cdr_stream_v1_control_monitoring" / args.source / args.model
    OUTPUT = BASE / "aggregated"
    if args.tag:
        OUTPUT = OUTPUT / args.tag
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(BASE / "control_turn_metrics.csv")
    started = time.time()
    nulls = null_streams(frame, np.random.default_rng(SEED))
    print(f"built {N_NULL} null streams in {time.time() - started:.0f}s", flush=True)
    streams = {s: aggregate(frame, s) for s in ("S1", "S2", "S3", "S0-E", "S0-H")}
    rows, calib = [], {}
    combos = [(w, m, d) for w in WINDOWS for m in METRICS for d in DETECTORS]
    for k, (window, metric, detector) in enumerate(combos, start=1):
        null_firsts = [grid(n[metric], window, detector) for n in nulls]
        chosen = None
        for value in ca.GRIDS[detector].values:
            if np.mean([g[value] is not None for g in null_firsts]) <= ALPHA:
                chosen = value
                break
        far = float(np.mean([g[chosen] is not None for g in null_firsts])) if chosen else None
        calib[f"W{window}|{metric}|{detector}"] = {"value": chosen, "null_far": far}
        for scenario, data in streams.items():
            first = grid(data[metric], window, detector)[chosen] if chosen is not None else None
            changes = CHANGES.get(scenario)
            category, change, delay = classify(first, changes) if changes else (
                "null_alarm" if first is not None else "null_quiet", None, None)
            rows.append({"window": window, "metric": metric, "detector": detector,
                         "parameter": chosen, "null_far": far, "scenario": scenario,
                         "first_alarm": first, "category": category,
                         "relative_to_change": change, "delay": delay})
        print(f"{k}/{len(combos)} W={window} {metric} {detector} "
              f"({(time.time() - started) / 60:.1f} min)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "aggregated_alarms.csv", index=False)
    plot(streams, table)
    summary = {"exploratory": True, "n_null": N_NULL, "calibration": calib,
               "alarms": table.to_dict(orient="records")}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                         encoding="utf-8")
    print(table[table["window"] == 20].to_string(index=False))
    return 0


def plot(streams, table) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for window in WINDOWS:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True, constrained_layout=True)
        for ax, (scenario, data) in zip(axes, streams.items(), strict=True):
            for metric in METRICS:
                signal = build_window_signal("CQ", data[metric], [], [], list(range(T)), window)
                ax.plot([t for t, _ in signal], [v for _, v in signal], color=COLORS[metric],
                        label=metric, lw=1.6)
                sub = table[(table["window"] == window) & (table["metric"] == metric)
                            & (table["scenario"] == scenario) & table["first_alarm"].notna()]
                for r in sub.itertuples():
                    ax.plot(r.first_alarm, 0.03 + 0.04 * METRICS.index(metric), "v",
                            color=COLORS[metric], ms=7)
            for c in CHANGES.get(scenario, ()):
                ax.axvline(c, color="black", ls="--", lw=1)
            ax.set_title(f"{scenario} (mean of 6 dialogues)")
            ax.set_xlabel("turn")
            ax.set_ylim(0, 1.02)
        axes[0].set_ylabel(f"window mean, W={window}")
        axes[0].legend(loc="lower left")
        fig.suptitle(f"GPT control arm, 6 dialogues aggregated per turn, W={window} "
                     "(EXPLORATORY). Triangles: first alarms of the five detectors "
                     "(threshold: null FAR <= 0.05 on 2,000 permuted S0-E streams).")
        for ext in ("png", "pdf"):
            fig.savefig(OUTPUT / f"aggregated_W{window}.{ext}", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
