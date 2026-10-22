"""Call-free power analysis of the unchanged detector layer on CDR-Stream-like streams.

Question: on a 100-turn stream of per-turn correctness, how large a drop in
accuracy can the existing window + detector machinery detect within 33 turns,
at a controlled false-alarm rate, and how does that depend on the window W,
the number of null streams used for calibration, and the number m of parallel
dialogues aggregated per turn?

Design (fixed before running; simulation only, no model call, no real data):
- Pre-change accuracy p0 = 0.80 (the order of the GPT control arm in the easy
  regime). After the change point c, p1 = p0 - delta.
- delta in {0.1, ..., 0.6}; c in {34, 51}; W in {10, 20, 30};
  m in {1, 3, 6, 18}: the per-turn value is the mean of m independent
  Bernoulli draws (m = 1 is one dialogue, as in the GPT screen).
- Detectors: the unchanged reference detectors and grids (cusum, page_hinkley,
  ewma, adwin, ddm), fed by build_window_signal exactly as in the analysis.
- Nulls: 2,000 streams with no change per (W, m, detector). Calibration picks
  the most sensitive grid value with null first-alarm rate <= 0.05 using
  (a) all 2,000 nulls ("oracle"), (b) 200 random subsets of 36 nulls, or
  (c) 200 random subsets of 12 nulls (the GPT screen). The true FAR of each
  pick is measured on the nulls outside the subset.
- Drift: 300 streams per (delta, c, W, m, detector). Detection = first alarm
  in [c, c + 33]; an alarm before c is a false alarm; delay = alarm - c.
Seeds: nulls 20261007, drift 20261008, subsets 20261009.
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

OUTPUT = ROOT / "results/cdr_stream_v1_power"
P0 = 0.80
DELTAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
CHANGES = (34, 51)
WINDOWS = (10, 20, 30)
MS = (1, 3, 6, 18)
DETECTORS = ("cusum", "page_hinkley", "ewma", "adwin", "ddm")
N_NULL, N_DRIFT, N_SUBSETS = 2000, 300, 200
SUBSET_SIZES = {"oracle": None, "n36": 36, "n12": 12}
ALPHA, SPAN, T = 0.05, ca.DETECTION_SPAN, 100


def first_alarm_matrix(rng, n, m, window, detector, p_of_turn) -> np.ndarray:
    """Rows = streams, columns = grid values; entry = first alarm turn or 0 (none)."""
    grid = ca.GRIDS[detector].values
    out = np.zeros((n, len(grid)), dtype=np.int16)
    p = np.asarray(p_of_turn)
    for i in range(n):
        values = (rng.random((T, m)) < p[:, None]).mean(axis=1)
        signal = build_window_signal("CQ", values, [], [], list(range(T)), window)
        firsts = first_alarms_for_grid(signal, ca.GRIDS[detector])
        out[i] = [firsts[g] or 0 for g in grid]
    return out


def pick(null: np.ndarray) -> int | None:
    rates = (null > 0).mean(axis=0)
    ok = np.flatnonzero(rates <= ALPHA)
    return int(ok[0]) if len(ok) else None


def main() -> int:
    global OUTPUT, DETECTORS
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    DETECTORS = tuple(args.detectors)
    if args.tag:
        OUTPUT = OUTPUT / args.tag
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng_null, rng_drift = np.random.default_rng(20261007), np.random.default_rng(20261008)
    rng_sub = np.random.default_rng(20261009)
    rows, started = [], time.time()
    combos = [(w, m, d) for w in WINDOWS for m in MS for d in DETECTORS]
    for k, (window, m, detector) in enumerate(combos, start=1):
        nulls = first_alarm_matrix(rng_null, N_NULL, m, window, detector, [P0] * T)
        subsets = {"oracle": [np.arange(N_NULL)]}
        for name, size in SUBSET_SIZES.items():
            if size:
                subsets[name] = [rng_sub.choice(N_NULL, size, replace=False)
                                 for _ in range(N_SUBSETS)]
        picks = {}
        for name, draws in subsets.items():
            chosen = []
            for idx in draws:
                j = pick(nulls[idx])
                rest = np.setdiff1d(np.arange(N_NULL), idx) if name != "oracle" else idx
                far = float((nulls[rest, j] > 0).mean()) if j is not None else None
                chosen.append((j, far))
            picks[name] = chosen
        for delta in DELTAS:
            for change in CHANGES:
                p = [P0 if t < change else P0 - delta for t in range(1, T + 1)]
                drift = first_alarm_matrix(rng_drift, N_DRIFT, m, window, detector, p)
                for name, chosen in picks.items():
                    det, ever, false, delays, fars, missing = [], [], [], [], [], 0
                    for j, far in chosen:
                        if j is None:
                            missing += 1
                            continue
                        first = drift[:, j].astype(int)
                        hit = (first >= change) & (first <= change + SPAN)
                        det.append(hit.mean())
                        ever.append((first >= change).mean())
                        false.append(((first > 0) & (first < change)).mean())
                        delays.extend((first[hit] - change).tolist())
                        fars.append(far)
                    rows.append({
                        "window": window, "m": m, "detector": detector, "delta": delta,
                        "change": change, "calibration": name,
                        "uncalibratable_share": missing / len(chosen),
                        "detection_33": float(np.mean(det)) if det else None,
                        "detection_by_100": float(np.mean(ever)) if ever else None,
                        "false_alarm_before_change": float(np.mean(false)) if false else None,
                        "median_delay": float(np.median(delays)) if delays else None,
                        "true_far": float(np.mean(fars)) if fars else None})
        elapsed = time.time() - started
        print(f"{k}/{len(combos)} ({100 * k / len(combos):.0f}%) W={window} m={m} {detector} "
              f"| {elapsed / 60:.1f} min, ETA {(elapsed / k) * (len(combos) - k) / 60:.1f} min",
              flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "power_grid.csv", index=False)
    plot(frame)
    summary = summarize(frame)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def best_over_detectors(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.sort_values("detection_33", ascending=False)
            .groupby(["window", "m", "delta", "change", "calibration"], as_index=False).first())


def summarize(frame: pd.DataFrame) -> dict:
    best = best_over_detectors(frame)
    out = {}
    for calibration in SUBSET_SIZES:
        part = best[(best["calibration"] == calibration)]
        table = {}
        for (window, m, change), sub in part.groupby(["window", "m", "change"]):
            reach = sub[sub["detection_33"] >= 0.8]["delta"]
            table[f"W{window}|m{m}|c{change}"] = {
                "min_delta_for_80pct_detection_within_33": float(reach.min())
                if len(reach) else None,
                "detection_at_delta_0.3": float(sub.loc[sub["delta"] == 0.3,
                                                        "detection_33"].iloc[0]),
                "best_detector_at_delta_0.3": sub.loc[sub["delta"] == 0.3, "detector"].iloc[0]}
        out[calibration] = table
    gpt = frame[(frame["m"] == 1) & (frame["window"] == 20) & (frame["calibration"] == "n12")
                & (frame["delta"] == 0.3)]
    out["gpt_screen_analogue_m1_W20_n12_delta0.3"] = gpt[
        ["detector", "change", "detection_33", "detection_by_100", "median_delay", "true_far",
         "uncalibratable_share"]].to_dict(orient="records")
    return out


def plot(frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = best_over_detectors(frame)
    colors = {1: "#D55E00", 3: "#E69F00", 6: "#009E73", 18: "#0072B2"}
    for calibration in SUBSET_SIZES:
        fig, axes = plt.subplots(len(CHANGES), len(WINDOWS), figsize=(13, 7), sharey=True,
                                 constrained_layout=True, squeeze=False)
        for r, change in enumerate(CHANGES):
            for c, window in enumerate(WINDOWS):
                ax = axes[r, c]
                for m in MS:
                    sub = best[(best["calibration"] == calibration) & (best["window"] == window)
                               & (best["m"] == m) & (best["change"] == change)]
                    ax.plot(sub["delta"], sub["detection_33"], "o-", color=colors[m],
                            label=f"m={m}")
                ax.axhline(0.8, color="#777777", ls=":", lw=0.8)
                ax.set_title(f"W={window}, change at turn {change}")
                ax.set_xlabel("accuracy drop delta (from 0.80)")
                if c == 0:
                    ax.set_ylabel("detection within 33 turns\n(best detector)")
                ax.set_ylim(-0.02, 1.02)
        axes[0, 0].legend(title="dialogues per turn")
        fig.suptitle(f"Simulated power, calibration={calibration} (FAR target 0.05); "
                     "m=1 is a single dialogue as in the GPT screen")
        for ext in ("png", "pdf"):
            fig.savefig(OUTPUT / f"power_{calibration}.{ext}", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
