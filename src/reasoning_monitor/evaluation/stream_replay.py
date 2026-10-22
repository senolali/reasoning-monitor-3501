"""Call-free LIC-RQ-v2-SR session-stream detector replay.

See LIC_RQ_V2_STREAM_REPLAY_PROTOCOL.md. Each completed LIC-RQ-v2 session
(model, task, condition) is one stream item. Streams switch from
``fresh_prefix`` to ``dialogue_history`` over the same tasks, so the only
change seen by the monitor is accumulated assistant history.

The window signal and the grid fast path are reused from
``evaluation.detector_development``; the detectors themselves are the
unchanged reference classes. No model, gateway or network call is made.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from reasoning_monitor.detectors.assignment import DETECTOR_STRATEGIES, assign_detector
from reasoning_monitor.detectors.detectors import build_detector
from reasoning_monitor.evaluation.cusum_calibration import wilson_interval
from reasoning_monitor.evaluation.detector_development import (
    DevelopmentSpec,
    _adwin_first_alarms,
    build_window_signal,
    first_alarms_for_grid,
)
from reasoning_monitor.evaluation.permutation_controls import (
    generate_matched_permutations,
)
from reasoning_monitor.metrics.dimensions import build_metric

METRICS: tuple[str, ...] = ("CQ", "CS", "RS", "LS", "SS")
STRATEGIES: tuple[str, ...] = tuple(DETECTOR_STRATEGIES)
WINDOWS: tuple[int, ...] = (10, 20, 30)
PRIMARY_WINDOW = 20
FAMILY_ALPHA = 0.05
CELL_ALPHA = FAMILY_ALPHA / (len(METRICS) * 3)
RAW_CELL_ALPHA = FAMILY_ALPHA / 3
RAW_CHANNEL = "CQ_raw"

N_CALIBRATION = 2000
N_VALIDATION = 2000
N_ORDERED = 1000
N_STATIONARY = 1000
N_BOOTSTRAP_STREAMS = 1000
N_H2_BOOTSTRAP = 2000

SEEDS = {
    "calibration": 71042,
    "validation": 81042,
    "sudden": 91042,
    "gradual": 91043,
    "stationary": 91044,
    "task_bootstrap": 101042,
    "h2_bootstrap": 121042,
}

# Dialogue-history sessions per 16-session block in LIC-G (sums to 48).
GRADUAL_DIALOGUE_COUNTS: tuple[int, ...] = (0, 3, 6, 10, 13, 16)
BLOCK_SIZE = 16

# Frozen grids (Sections 36/39), most to least sensitive.
GRIDS: dict[str, DevelopmentSpec] = {
    "cusum": DevelopmentSpec(
        "cusum", "threshold",
        (0.15, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00),
    ),
    "adwin": DevelopmentSpec(
        "adwin", "delta", (0.05, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001),
    ),
    "ddm": DevelopmentSpec(
        "ddm", "drift_level", (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0),
    ),
    "page_hinkley": DevelopmentSpec(
        "page_hinkley", "threshold",
        (0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
    ),
}

PREDICTIONS = {"gradual": ("CS", "LS"), "sudden": ("CQ", "RS")}


@dataclass(frozen=True)
class Scenario:
    name: str
    t_s: int | None
    t_star: int | None


SCENARIOS = {
    "sudden": Scenario("sudden", 49, 49),
    "gradual": Scenario("gradual", 17, 49),
}


# ── session values ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionTable:
    """Per-model session values. Index i < n_tasks is fresh, i + n_tasks dialogue."""

    tasks: tuple[str, ...]
    values: dict[str, np.ndarray]

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)


def session_values(metrics: pd.DataFrame, value_mode: str = "session_mean"
                   ) -> dict[str, SessionTable]:
    """Build the frozen Section 4 session values for every model."""
    if value_mode not in {"session_mean", "final_turn"}:
        raise ValueError(f"unknown value_mode {value_mode!r}")
    tables = {}
    for model, frame in metrics.groupby("model", sort=True):
        tasks = tuple(sorted(frame["task_id"].unique()))
        values = {metric: np.full(2 * len(tasks), np.nan) for metric in METRICS}
        for offset, condition in enumerate(("fresh_prefix", "dialogue_history")):
            arm = frame[frame["condition"] == condition]
            for index, task in enumerate(tasks):
                rows = arm[arm["task_id"] == task].sort_values("turn")
                if rows.empty:
                    raise ValueError(f"missing session {model}/{task}/{condition}")
                final = rows[rows["turn"] == rows["total_turns"]]
                if len(final) != 1:
                    raise ValueError(f"no unique final turn for {model}/{task}/{condition}")
                final = final.iloc[0]
                slot = offset * len(tasks) + index
                cq = final["CQ"]
                values["CQ"][slot] = 0.0 if pd.isna(cq) else float(cq)
                values["RS"][slot] = np.nan if pd.isna(final["RS"]) else float(final["RS"])
                for metric in ("CS", "LS", "SS"):
                    if value_mode == "final_turn":
                        value = final[metric]
                        values[metric][slot] = np.nan if pd.isna(value) else float(value)
                    else:
                        measured = rows[metric].dropna()
                        values[metric][slot] = (
                            float(measured.mean()) if len(measured) else np.nan
                        )
        tables[str(model)] = SessionTable(tasks, values)
    return tables


def missingness(tables: Mapping[str, SessionTable]) -> list[dict]:
    rows = []
    for model, table in tables.items():
        n = table.n_tasks
        for metric in METRICS:
            for label, part in (("fresh_prefix", slice(0, n)),
                                ("dialogue_history", slice(n, 2 * n))):
                values = table.values[metric][part]
                valid = int(np.isfinite(values).sum())
                rows.append({"model": model, "metric": metric, "condition": label,
                             "sessions": n, "valid": valid,
                             "missing_rate": 1 - valid / n,
                             "mean": float(np.nanmean(values)) if valid else None})
    return rows


# ── stream construction ─────────────────────────────────────────────────────

def null_streams(n_items: int, count: int, seed: int,
                 exclude: Sequence[tuple[int, ...]] = ()) -> list[tuple[int, ...]]:
    """Unique non-identity permutations, disjoint from ``exclude``."""
    excluded = set(exclude)
    candidates = generate_matched_permutations(n_items, count + len(excluded), seed)
    streams = [candidate for candidate in candidates if candidate not in excluded]
    if len(streams) < count:
        raise RuntimeError("not enough disjoint permutations")
    return streams[:count]


def sudden_stream(fresh: Sequence[int], dialogue: Sequence[int],
                  rng: random.Random) -> tuple[int, ...]:
    first, second = list(fresh), list(dialogue)
    rng.shuffle(first)
    rng.shuffle(second)
    return tuple(first + second)


def gradual_stream(fresh: Sequence[int], dialogue: Sequence[int],
                   rng: random.Random) -> tuple[int, ...]:
    fresh_pool, dialogue_pool = list(fresh), list(dialogue)
    if len(fresh_pool) != 48 or len(dialogue_pool) != 48:
        raise ValueError("LIC-G requires 48 sessions per condition")
    rng.shuffle(fresh_pool)
    rng.shuffle(dialogue_pool)
    stream: list[int] = []
    for n_dialogue in GRADUAL_DIALOGUE_COUNTS:
        block = [fresh_pool.pop() for _ in range(BLOCK_SIZE - n_dialogue)]
        block += [dialogue_pool.pop() for _ in range(n_dialogue)]
        rng.shuffle(block)
        stream.extend(block)
    return tuple(stream)


def ordered_streams(kind: str, n_tasks: int, count: int, seed: int,
                    bootstrap: bool = False) -> list[tuple[int, ...]]:
    builder = {"sudden": sudden_stream, "gradual": gradual_stream}[kind]
    rng = random.Random(seed)
    streams = []
    for _ in range(count):
        if bootstrap:
            tasks = [rng.randrange(n_tasks) for _ in range(n_tasks)]
        else:
            tasks = list(range(n_tasks))
        fresh = tasks
        dialogue = [task + n_tasks for task in tasks]
        streams.append(builder(fresh, dialogue, rng))
    return streams


def stationary_streams(n_tasks: int, count: int, seed: int
                       ) -> dict[str, list[tuple[int, ...]]]:
    rng = random.Random(seed)
    result = {}
    for label, offset in (("fresh_only", 0), ("dialogue_only", n_tasks)):
        streams = []
        for _ in range(count):
            items = [offset + task for task in range(n_tasks)]
            rng.shuffle(items)
            streams.append(tuple(items))
        result[label] = streams
    return result


# ── detectors ───────────────────────────────────────────────────────────────

def strategy_detectors() -> dict[str, dict[str, str]]:
    """Strategy -> metric -> detector name, read from the assignment layer."""
    table = {}
    for strategy in STRATEGIES:
        table[strategy] = {
            metric: assign_detector(build_metric(metric, window_size=10),
                                    strategy=strategy).name
            for metric in METRICS
        }
    return table


def calibration_cells() -> list[tuple[str, str]]:
    """Unique (metric, detector) pairs used by any strategy."""
    cells = set()
    for mapping in strategy_detectors().values():
        cells.update(mapping.items())
    return sorted(cells)


def window_signal(values: np.ndarray, stream: Sequence[int], window: int
                  ) -> list[tuple[int, float]]:
    return build_window_signal("replay", values, [], [], stream, window_size=window)


def raw_signal(values: np.ndarray, stream: Sequence[int]) -> list[tuple[int, float]]:
    return [(t, float(values[index])) for t, index in enumerate(stream, start=1)
            if math.isfinite(float(values[index]))]


def grid_first_alarms(signal: Sequence[tuple[int, float]], detector: str
                      ) -> dict[float, int | None]:
    return first_alarms_for_grid(signal, GRIDS[detector])


def alarm_times(signal: Sequence[tuple[int, float]], detector: str, value: float,
                stop_at: int | None = None) -> list[int]:
    """All alarm times with reference reset semantics.

    Stops after the first alarm at or after ``stop_at`` when it is given.
    ADWIN uses the exact shared-split fast path, restarted after each alarm,
    which is what the reference ``reset()`` implies.
    """
    spec = GRIDS[detector]
    times: list[int] = []
    if detector == "adwin":
        remaining = list(signal)
        while remaining:
            first = _adwin_first_alarms(remaining, [value])[float(value)]
            if first is None:
                break
            times.append(first)
            if stop_at is not None and first >= stop_at:
                break
            remaining = [(t, v) for t, v in remaining if t > first]
        return times
    instance = build_detector(detector, **{spec.parameter: value})
    for t, item in signal:
        alarm = instance.update(item, t, "replay")
        if alarm is not None:
            times.append(int(alarm.t))
            if stop_at is not None and alarm.t >= stop_at:
                break
    return times


# ── calibration ─────────────────────────────────────────────────────────────

def select_parameter(alarm_counts: Mapping[str, Mapping[float, int]], n_streams: int,
                     values: Sequence[float], alpha: float) -> float | None:
    """First grid value whose Wilson upper bound is <= alpha in every model."""
    for value in values:
        if all(wilson_interval(counts[value], n_streams)[1] <= alpha
               for counts in alarm_counts.values()):
            return value
    return None


def passes_bound(count: int, n_streams: int, alpha: float) -> bool:
    return wilson_interval(count, n_streams)[1] <= alpha


# ── outcome summaries ───────────────────────────────────────────────────────

def categorize(first_alarm: int | None, scenario: Scenario) -> str:
    if first_alarm is None:
        return "no_alarm"
    if first_alarm < scenario.t_s:
        return "false_alarm"
    if first_alarm < scenario.t_star:
        return "early_detection"
    return "post_threshold"


def first_at_or_after(times: Sequence[int], t_s: int) -> int | None:
    return next((t for t in times if t >= t_s), None)


def cliffs_delta(post: Sequence[float], pre: Sequence[float]) -> float:
    """P(post > pre) - P(post < pre); negative means post is lower."""
    a, b = np.asarray(post, float), np.asarray(pre, float)
    if not len(a) or not len(b):
        return float("nan")
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (len(a) * len(b)))


def kendalls_w(ranks: np.ndarray) -> float | None:
    """Kendall's W with tie correction; rows are raters, columns items."""
    m, n = ranks.shape
    if m < 2 or n < 2:
        return None
    totals = ranks.sum(axis=0)
    s = float(((totals - totals.mean()) ** 2).sum())
    ties = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        ties += float((counts ** 3 - counts).sum())
    denominator = m * m * (n ** 3 - n) - m * ties
    if denominator <= 0:
        return None
    return 12.0 * s / denominator


def rank_rows(times: np.ndarray) -> np.ndarray:
    """Average ranks per row; no-alarm (inf) ties at the maximum rank."""
    from scipy.stats import rankdata

    return np.vstack([rankdata(row, method="average") for row in times])


def first_to_alarm_share(times: np.ndarray) -> np.ndarray:
    """Per column: fraction of rows where it is strictly earliest (finite)."""
    shares = np.zeros(times.shape[1])
    for row in times:
        finite = np.isfinite(row)
        if not finite.any():
            continue
        minimum = row.min()
        winners = np.flatnonzero(row == minimum)
        if len(winners) == 1:
            shares[winners[0]] += 1
    return shares / max(len(times), 1)


def prediction_holds(shares: Mapping[str, float], predicted: Sequence[str]) -> bool:
    if not all(metric in shares for metric in predicted):
        return False
    if any(shares[metric] <= 0 for metric in predicted):
        return False
    lowest_predicted = min(shares[metric] for metric in predicted)
    return all(share < lowest_predicted
               for metric, share in shares.items() if metric not in predicted)


def paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, n_boot: int,
                                seed: int) -> tuple[float, float, float]:
    """Mean(a - b) with a percentile 95% interval over resampled rows."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, float) - np.asarray(b, float)
    if not len(diff):
        return float("nan"), float("nan"), float("nan")
    draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[draws].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975))
