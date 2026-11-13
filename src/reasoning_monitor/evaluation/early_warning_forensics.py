"""Call-free forensic checks for CDR-Stream early-warning analyses.

This module does not replace any frozen analysis.  It separates three questions:

1. Can the exact monitoring pipeline react to an injected precursor in time?
2. Do raw, baseline-standardised metric deviations lead CQ on saved streams?
3. How much delay is introduced by windowing followed by detector warm-up?

All routines operate on already-scored turn tables or synthetic scalar streams.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from reasoning_monitor.detectors.detectors import CUSUM
from reasoning_monitor.evaluation.detector_development import build_window_signal

BASELINE_END = 30
PERSISTENCE = 3
CHANGE_POINTS = {"S1": 34, "S2": 51, "S3": 34}
NULL_SCENARIOS = ("S0-E", "S0-H")
RESIDUAL_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
LEGACY_THRESHOLDS = (0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                     5.0, 7.5, 10.0)


def first_possible_legacy_turn(window: int = 20, warmup: int = 20) -> int:
    """Earliest source turn on which a windowed detector may emit an alarm."""
    return window + warmup


def _baseline(values: Sequence[float], baseline_end: int = BASELINE_END,
              scale_floor: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)[:baseline_end]
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(np.mean(x)), max(float(np.std(x, ddof=1)), scale_floor)


def fixed_baseline_cusum(
    values: Sequence[float],
    threshold: float,
    *,
    direction: str = "down",
    baseline_end: int = BASELINE_END,
    allowance: float = 0.5,
    scale_floor: float = 0.05,
) -> int | None:
    """CUSUM of standardised deviations from a fixed pre-monitoring baseline.

    Unlike the historical path, this consumes raw turn values, has no second
    warm-up, and never lets post-baseline observations move the reference mean.
    ``direction='both'`` spends the same threshold in two directions and is
    therefore calibrated as a combined rule on null streams.
    """
    mu, scale = _baseline(values, baseline_end, scale_floor)
    if not (math.isfinite(mu) and math.isfinite(scale)):
        return None
    down = up = 0.0
    for index, value in enumerate(values, start=1):
        if index <= baseline_end or not math.isfinite(float(value)):
            continue
        z = (float(value) - mu) / scale
        down = max(0.0, down - z - allowance)
        up = max(0.0, up + z - allowance)
        fired = down > threshold if direction == "down" else up > threshold
        if direction == "both":
            fired = down > threshold or up > threshold
        if fired:
            return index
    return None


def legacy_window_cusum(values: Sequence[float], threshold: float, *,
                        window: int = 20, warmup: int = 20) -> int | None:
    """Exact historical rolling-mean then running-reference CUSUM path."""
    signal = build_window_signal("metric", values, [], [], range(len(values)), window)
    detector = CUSUM(threshold=threshold, warmup=warmup)
    for turn, value in signal:
        alarm = detector.update(value, turn, "synthetic")
        if alarm is not None:
            return int(alarm.t)
    return None


def select_threshold(null_streams: Iterable[Sequence[float]], *, direction: str,
                     grid: Sequence[float] = RESIDUAL_THRESHOLDS,
                     max_alarm_rate: float = 0.05) -> tuple[float | None, float]:
    """Choose the first (most sensitive) threshold satisfying the null FAR."""
    streams = list(null_streams)
    if not streams:
        return None, float("nan")
    for threshold in grid:
        alarms = sum(fixed_baseline_cusum(x, threshold, direction=direction) is not None
                     for x in streams)
        rate = alarms / len(streams)
        if rate <= max_alarm_rate:
            return float(threshold), float(rate)
    return None, 1.0


def select_legacy_threshold(null_streams: Iterable[Sequence[float]], *,
                            grid: Sequence[float] = LEGACY_THRESHOLDS,
                            max_alarm_rate: float = 0.05) -> tuple[float | None, float]:
    streams = list(null_streams)
    if not streams:
        return None, float("nan")
    for threshold in grid:
        alarms = sum(legacy_window_cusum(x, threshold) is not None for x in streams)
        rate = alarms / len(streams)
        if rate <= max_alarm_rate:
            return float(threshold), float(rate)
    return None, 1.0


def stream_arrays(frame: pd.DataFrame, metric: str, scenarios: Sequence[str] | None = None
                  ) -> dict[str, np.ndarray]:
    selected = frame if scenarios is None else frame[frame["scenario"].isin(scenarios)]
    return {
        str(dialogue): part.sort_values("turn")[metric].to_numpy(float)
        for dialogue, part in selected.groupby("dialogue_id")
    }


def alarm_table(frame: pd.DataFrame, metric: str, threshold: float, *,
                direction: str) -> pd.DataFrame:
    rows = []
    for (dialogue, scenario), part in frame.groupby(["dialogue_id", "scenario"]):
        part = part.sort_values("turn")
        alarm = fixed_baseline_cusum(part[metric].to_numpy(float), threshold,
                                     direction=direction)
        change = CHANGE_POINTS.get(str(scenario))
        if change is None:
            category = "null_alarm" if alarm is not None else "null_quiet"
        elif alarm is None:
            category = "miss"
        elif alarm < change:
            category = "false_alarm"
        elif alarm <= change + 33:
            category = "detection"
        else:
            category = "late"
        rows.append({"dialogue_id": dialogue, "scenario": scenario, "metric": metric,
                     "direction": direction, "threshold": threshold,
                     "alarm": alarm, "change": change, "category": category})
    return pd.DataFrame(rows)


def summarize_against_cq(metric_table: pd.DataFrame, cq_table: pd.DataFrame) -> dict:
    null = metric_table[metric_table["scenario"].isin(NULL_SCENARIOS)]
    drift = metric_table[metric_table["scenario"].isin(CHANGE_POINTS)]
    joined = drift.merge(
        cq_table[["dialogue_id", "alarm", "category"]], on="dialogue_id",
        suffixes=("_metric", "_cq"), validate="one_to_one",
    )
    valid = joined[(joined["category_metric"] == "detection")
                   & (joined["category_cq"] == "detection")].copy()
    valid["lead"] = valid["alarm_metric"] - valid["alarm_cq"]
    return {
        "n_null": int(len(null)),
        "null_alarm_rate": float((null["category"] == "null_alarm").mean()),
        "n_drift": int(len(drift)),
        "detection_rate": float((drift["category"] == "detection").mean()),
        "prechange_false_alarm_rate": float((drift["category"] == "false_alarm").mean()),
        "both_detected": int(len(valid)),
        "leads_cq": int((valid["lead"] < 0).sum()),
        "same_as_cq": int((valid["lead"] == 0).sum()),
        "lags_cq": int((valid["lead"] > 0).sum()),
        "median_lead": float(valid["lead"].median()) if len(valid) else None,
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-aware AUROC via the Mann-Whitney rank identity."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank_sum = float(_ranks(s)[y == 1].sum())
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def lead_lag_auc(frame: pd.DataFrame, metric: str, *, direction: str,
                 lags: Sequence[int] = tuple(range(1, 21))) -> pd.DataFrame:
    """Descriptive AUROC for a metric anomaly at t predicting CQ failure at t+lag.

    Rows where CQ is already wrong at t are excluded, so this is a warning rather
    than contemporaneous discrimination calculation. Dependence within dialogues
    is intentionally not converted into a p-value.
    """
    labels = {lag: [] for lag in lags}
    scores = {lag: [] for lag in lags}
    for _, part in frame[frame["scenario"].isin(CHANGE_POINTS)].groupby("dialogue_id"):
        part = part.sort_values("turn").reset_index(drop=True)
        values = part[metric].to_numpy(float)
        cq = part["CQ"].to_numpy(float)
        mu, scale = _baseline(values)
        if not (math.isfinite(mu) and math.isfinite(scale)):
            continue
        z = (values - mu) / scale
        anomaly = -z if direction == "down" else z
        if direction == "both":
            anomaly = np.abs(z)
        for lag in lags:
            for i in range(BASELINE_END, len(part) - lag):
                if not (math.isfinite(cq[i]) and cq[i] == 1.0
                        and math.isfinite(cq[i + lag]) and math.isfinite(anomaly[i])):
                    continue
                labels[lag].append(int(cq[i + lag] == 0.0))
                scores[lag].append(float(anomaly[i]))
    return pd.DataFrame([
        {"metric": metric, "direction": direction, "lag": lag,
         "auc": auc_binary(labels[lag], scores[lag]), "n": len(labels[lag]),
         "failures": int(sum(labels[lag]))}
        for lag in lags
    ])


def synthetic_injection(seed: int = 20261105, n_null: int = 2000,
                        n_shift: int = 1000) -> pd.DataFrame:
    """Measure whether an injected turn-31 precursor can be seen before turn 34."""
    rng = np.random.default_rng(seed)

    def draw(delta=0.0):
        values = np.clip(rng.normal(0.75, 0.10, 100), 0.0, 1.0)
        if delta:
            values[30:] = np.clip(values[30:] - delta, 0.0, 1.0)
        return values

    nulls = [draw() for _ in range(n_null)]
    residual_h, residual_far = select_threshold(nulls, direction="down")
    legacy_h, legacy_far = select_legacy_threshold(nulls)
    rows = []
    for delta in (0.0, 0.1, 0.2, 0.3, 0.5):
        streams = [draw(delta) for _ in range(n_shift)]
        for method, threshold, far, alarm_fn in (
            ("fixed_baseline_raw", residual_h, residual_far,
             lambda x, h: fixed_baseline_cusum(x, h, direction="down")),
            ("legacy_W20_plus_warmup20", legacy_h, legacy_far,
             lambda x, h: legacy_window_cusum(x, h)),
        ):
            alarms = [alarm_fn(x, threshold) if threshold is not None else None for x in streams]
            finite = [x for x in alarms if x is not None]
            rows.append({
                "method": method,
                "delta": delta,
                "threshold": threshold,
                "calibration_null_far": far,
                "alarm_rate": len(finite) / len(alarms),
                "warning_by_turn_33": sum(x is not None and x <= 33 for x in alarms)
                / len(alarms),
                "detection_by_turn_67": sum(x is not None and x <= 67 for x in alarms)
                / len(alarms),
                "median_alarm": float(np.median(finite)) if finite else None,
            })
    return pd.DataFrame(rows)
