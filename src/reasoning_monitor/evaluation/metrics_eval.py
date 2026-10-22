"""Evaluation of monitoring performance.

v0.1.3-final changes
--------------------
Three alarm categories (P0 fix from ChatGPT review):
  false_alarm      : ta < ts  (before drift starts — definitively wrong)
  early_detection  : ts ≤ ta < t*  (drift started, threshold not yet crossed)
  post_threshold   : ta ≥ t*  (after operational quality threshold)

For sudden scenarios ts == t*, so early_detection zone is empty and every
alarm at ta ≥ ts is post_threshold.  The fix matters only for gradual
scenarios (S1, S4, S5) where ts < t* by transition_width // 2 queries.

Signed delay D* = ta - t* is stored (negative = early warning).

Event-based Precision / Recall / F1 (H2 requirement from proposal):
  Per (metric, detector, strategy):
    TP  = drift scenarios where alarm fired in drift window [ts, end]
    FP  = S0 scenarios where alarm fired  (false alarm events)
    FN  = drift scenarios where no alarm fired
  F1 = 2·TP / (2·TP + FP + FN)

This is scenario-level F1, appropriate for single-event scenarios.
Multi-event scenarios (S3, S6) are treated as single events here;
multi-event ground truth is documented as future work in DEVIATIONS §17.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from reasoning_monitor.types import Alarm, MetricSnapshot


class AlarmCategory(str, Enum):
    FALSE_ALARM     = "false_alarm"      # ta < ts
    EARLY_DETECTION = "early_detection"  # ts ≤ ta < t* (gradual only)
    POST_THRESHOLD  = "post_threshold"   # ta ≥ t*


@dataclass(frozen=True)
class ChangeTest:
    metric: str
    mean_before: float
    mean_after: float
    delta: float
    pct_change: float
    u_statistic: float
    p_value: float
    rank_biserial: float
    n_before: int
    n_after: int


@dataclass(frozen=True)
class DetectionResult:
    metric: str
    detector: str
    level: str
    detected: bool
    """True if any alarm fired in [ts, end] (drift window)."""
    early_detected: bool
    """True if the FIRST detected alarm is in [ts, t*) — an early warning."""
    delay: float | None
    """Signed D* = ta_first - t*. Negative = early warning. None if not detected."""
    delay_from_transition_start: float | None
    """ta_first - ts. Always ≥ 0 when detected. Identical to delay for sudden."""
    first_alarm_category: str | None
    """Category of first alarm: 'early_detection' or 'post_threshold'. None if missed."""
    n_false_alarms: int
    false_alarm_rate: float


@dataclass(frozen=True)
class F1Result:
    metric: str
    detector: str
    level: str
    strategy: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_drift_scenarios: int
    n_null_scenarios: int


def _rank_biserial(a: Sequence[float], b: Sequence[float], u: float) -> float:
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    return 1.0 - (2.0 * u) / (n1 * n2)


def test_change(
    snapshots: Sequence[MetricSnapshot],
    change_point: int,
    metric: str,
    guard: int = 0,
    evaluation_start: int | None = None,
    evaluation_end: int | None = None,
) -> ChangeTest | None:
    """H1 test: Mann-Whitney U on pre- vs post-drift quality windows.

    `change_point` should be effective_t_star, not transition_start.
    """
    start = evaluation_start if evaluation_start is not None else -float("inf")
    end = evaluation_end if evaluation_end is not None else float("inf")
    before = [s.values[metric] for s in snapshots
              if start <= s.t < change_point - guard and metric in s.values]
    after  = [s.values[metric] for s in snapshots
              if change_point + guard < s.t < end and metric in s.values]
    if len(before) < 3 or len(after) < 3:
        return None

    u, p = stats.mannwhitneyu(before, after, alternative="two-sided")
    mb, ma = float(np.mean(before)), float(np.mean(after))
    return ChangeTest(
        metric=metric, mean_before=mb, mean_after=ma,
        delta=ma - mb,
        pct_change=((ma - mb) / mb * 100.0) if mb else float("nan"),
        u_statistic=float(u), p_value=float(p),
        rank_biserial=_rank_biserial(before, after, float(u)),
        n_before=len(before), n_after=len(after),
    )


def evaluate_detection(
    alarms: Sequence[Alarm],
    effective_cp: int,
    metric: str,
    detector: str,
    n_pre_observations: int,
    level: str = "drift",
    transition_start: int | None = None,
    evaluation_start: int | None = None,
    evaluation_end: int | None = None,
) -> DetectionResult:
    """Classify alarms into three categories and compute detection metrics.

    For sudden scenarios (ts == t*): early_detection zone is empty.
    For gradual scenarios (ts < t*): alarms in [ts, t*) are early detections —
      valid true detections with negative signed delay.

    Parameters
    ----------
    effective_cp      : t* = midpoint for gradual, change_point for sudden
    transition_start  : ts = declared change_point in the spec
    level             : "drift" or "drift_raw" (CQ two-level separation)
    """
    ts = transition_start if transition_start is not None else effective_cp
    start = evaluation_start if evaluation_start is not None else -float("inf")
    end = evaluation_end if evaluation_end is not None else float("inf")
    mine = [a for a in alarms
            if a.metric == metric and a.detector == detector and a.level == level
            and start <= a.t < end]

    # Null scenario (S0): all alarms are false
    if effective_cp < 0:
        return DetectionResult(
            metric=metric, detector=detector, level=level,
            detected=False, early_detected=False,
            delay=None, delay_from_transition_start=None,
            first_alarm_category=None,
            n_false_alarms=len(mine),
            false_alarm_rate=100.0 * len(mine) / max(n_pre_observations, 1),
        )

    # Three-category classification (P0 fix)
    false_alarms   = [a for a in mine if a.t < ts]
    early_det      = [a for a in mine if ts <= a.t < effective_cp]
    post_thresh    = [a for a in mine if a.t >= effective_cp]

    # All alarms at/after ts are "detected" (drift has begun)
    true_alarms = sorted(early_det + post_thresh, key=lambda a: a.t)
    detected    = bool(true_alarms)

    if detected:
        first = true_alarms[0]
        delay              = float(first.t - effective_cp)   # signed D*
        delay_from_start   = float(first.t - ts)             # ≥ 0
        early_detected     = first.t < effective_cp
        first_category     = (AlarmCategory.EARLY_DETECTION.value
                               if early_detected
                               else AlarmCategory.POST_THRESHOLD.value)
    else:
        delay = delay_from_start = None
        early_detected = False
        first_category = None

    return DetectionResult(
        metric=metric, detector=detector, level=level,
        detected=detected, early_detected=early_detected,
        delay=delay, delay_from_transition_start=delay_from_start,
        first_alarm_category=first_category,
        n_false_alarms=len(false_alarms),
        false_alarm_rate=100.0 * len(false_alarms) / max(n_pre_observations, 1),
    )


def compute_f1(
    detection_rows: list[dict],
    strategy: str,
    level: str = "drift",
) -> list[dict]:
    """H2 metric: scenario-level Precision, Recall, F1 per (metric, detector).

    Classification:
      TP  = drift scenarios (change_point > 0) where detected=True
      FP  = null scenarios  (change_point < 0) where detected would fire
            Approximated here as: scenarios where n_false_alarms > 0
      FN  = drift scenarios where detected=False

    For single-event scenarios this is exact.  S3 (recurring) and S6 (mixed)
    are treated as single events; multi-event ground truth is future work.
    """
    rows = [r for r in detection_rows if r.get("level", "drift") == level
            and r.get("strategy", strategy) == strategy]

    results = []
    for (metric, detector), grp in _group_by(rows, ["metric", "detector"]):
        drift_rows = [r for r in grp if r.get("effective_t_star", 1) > 0]
        null_rows  = [r for r in grp if r.get("effective_t_star", 1) <= 0]

        tp = sum(1 for r in drift_rows if r["detected"])
        fn = sum(1 for r in drift_rows if not r["detected"])
        fp = sum(1 for r in null_rows  if r.get("n_false_alarms", 0) > 0)

        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall    = tp / denom_r if denom_r > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results.append({
            "metric": metric, "detector": detector,
            "level": level, "strategy": strategy,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "n_drift_scenarios": len(drift_rows),
            "n_null_scenarios": len(null_rows),
        })
    return results


def _group_by(rows: list[dict], keys: list[str]):
    """Simple groupby helper (no pandas dependency in this module)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = tuple(r.get(key) for key in keys)
        groups.setdefault(k, []).append(r)
    return groups.items()


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction for H1 multiple testing."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            rejected[idx] = True
        else:
            break
    return rejected


def change_tests_frame(tests) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in tests if t is not None])


def detection_frame(results) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def early_warning_ranking(detections) -> pd.DataFrame:
    """Which metric-level pair gives the earliest warning?"""
    rows: list[dict] = []
    by_key: dict[tuple, list] = {}
    for d in detections:
        k = (d.metric, d.detector, d.level)
        by_key.setdefault(k, []).append(d)

    for (metric, detector, level), items in by_key.items():
        delays = [d.delay for d in items if d.detected and d.delay is not None]
        early  = [d for d in items if d.early_detected]
        rows.append({
            "metric": metric, "detector": detector, "level": level,
            "detection_rate": sum(d.detected for d in items) / len(items),
            "early_detection_rate": len(early) / len(items),
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "median_delay": float(np.median(delays)) if delays else float("nan"),
            "mean_false_alarm_rate": float(np.mean([d.false_alarm_rate for d in items])),
            "n_scenarios": len(items),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("mean_delay", na_position="last").reset_index(drop=True)


def _kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance for seed-by-metric ranks."""
    n_judges, n_items = rank_matrix.shape
    if n_judges < 2 or n_items < 2:
        return float("nan")
    rank_sums = rank_matrix.sum(axis=0)
    numerator = 12.0 * np.square(rank_sums - rank_sums.mean()).sum()
    denominator = n_judges**2 * (n_items**3 - n_items)
    return float(numerator / denominator) if denominator else float("nan")


def cross_seed_alarm_order(
    detections: pd.DataFrame,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise first-alarm order and its reproducibility across seeds.

    Missed detections are tied after all observed alarms. The permutation null
    independently shuffles metric labels within each seed while preserving the
    observed alarm/miss pattern.
    """
    required = {"seed", "scenario", "model", "transition", "metric", "detected", "delay"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"cross-seed analysis missing columns: {sorted(missing)}")
    data = detections[
        (detections.get("level", "drift") == "drift")
        & (detections["effective_t_star"] > 0)
    ].copy()
    group_cols = ["scenario", "model", "transition", "window_size", "strategy"]
    summaries: list[dict] = []
    order_rows: list[dict] = []
    rng = np.random.default_rng(random_seed)

    for key, group in data.groupby(group_cols, dropna=False):
        seeds = sorted(group["seed"].unique())
        metrics = [metric for metric in ("CQ", "CS", "RS", "LS", "ES", "SS")
                   if metric in set(group["metric"])]
        if len(seeds) < 2 or len(metrics) < 2:
            continue
        matrix = []
        first_times: dict[int, dict[str, float]] = {}
        for seed in seeds:
            seed_rows = group[group["seed"] == seed].set_index("metric")
            observed = {
                metric: (
                    float(seed_rows.loc[metric, "delay"])
                    if metric in seed_rows.index
                    and bool(seed_rows.loc[metric, "detected"])
                    and pd.notna(seed_rows.loc[metric, "delay"])
                    else float("inf")
                )
                for metric in metrics
            }
            first_times[seed] = observed
            finite = [value for value in observed.values() if np.isfinite(value)]
            miss_rank = len(finite) + 1.0
            ranks = pd.Series(observed).rank(method="average").replace(np.inf, miss_rank)
            matrix.append([float(ranks[metric]) for metric in metrics])
            order_rows.extend({
                **dict(zip(group_cols, key, strict=True)), "seed": seed,
                "metric": metric, "first_alarm_delay": observed[metric],
                "rank": float(ranks[metric]),
            } for metric in metrics)

        rank_matrix = np.asarray(matrix, dtype=float)
        observed_w = _kendalls_w(rank_matrix)
        null = np.empty(n_permutations, dtype=float)
        for index in range(n_permutations):
            permuted = np.vstack([rng.permutation(row) for row in rank_matrix])
            null[index] = _kendalls_w(permuted)
        p_value = float((1 + np.sum(null >= observed_w)) / (n_permutations + 1))
        base = dict(zip(group_cols, key, strict=True))
        summaries.append({
            **base, "n_seeds": len(seeds), "n_metrics": len(metrics),
            "kendalls_w": observed_w, "permutation_p": p_value,
            "n_permutations": n_permutations,
            "ss_before_cq_rate": float(np.mean([
                first_times[seed].get("SS", np.inf) < first_times[seed].get("CQ", np.inf)
                for seed in seeds
            ])),
            "cs_before_cq_rate": float(np.mean([
                first_times[seed].get("CS", np.inf) < first_times[seed].get("CQ", np.inf)
                for seed in seeds
            ])),
        })
    return pd.DataFrame(summaries), pd.DataFrame(order_rows)
