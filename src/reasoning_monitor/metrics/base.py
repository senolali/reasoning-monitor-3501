"""Incremental metric interface.

Design note: all six metrics are O(1) per update.

  WindowMeanMetric    — CQ, CS, RS, LS, ES
    Maintains a running sum over a deque; subtract the departing value, add
    the arriving one.  True O(1).

  WindowDispersionMetric — SS
    v0.1.0/v0.1.1 computed `sum(self._means)` and `sum((m-mean)**2 ...)` on
    every update — O(W), not O(1).
    v0.1.2 fix: sliding-window sum and sum-of-squares allow O(1) variance via
      Var = E[X²] - E[X]²
    applied to the deque of consecutive window means.

raw_detection flag (new in v0.1.2)
-----------------------------------
CQ is Bernoulli per query (0 or 1).  Feeding the window *mean* (∈ [0,1/W,
2/W, …, 1]) to DDM violates DDM's Binomial assumptions.  MetricSpec carries
a `raw_detection` flag; when True, the monitor also runs a secondary detector
on the per-query value rather than the window value.  Only CQ uses this.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from reasoning_monitor.types import Distribution, MetricName, QueryObservation


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: MetricName
    description: str
    distribution: Distribution
    """Per-query distribution — used for the RAW secondary detector (e.g. DDM on Bernoulli CQ)."""
    higher_is_better: bool = True
    raw_detection: bool = False
    windowed_distribution: Distribution | None = None
    """Distribution of the W-step WINDOWED value.
    When set, the PRIMARY (windowed) detector is assigned from this distribution
    instead of `distribution`.

    Rationale for CQ:
      - Per-query: Bernoulli(p) → raw DDM is appropriate.
      - W=20 window mean: bounded [0,1], Beta-like by CLT → CUSUM is appropriate.
    Setting windowed_distribution=BETA on CQ gives:
      primary  CQ_windowed → CUSUM  (correct for bounded continuous mean)
      secondary CQ_raw     → DDM    (correct for Bernoulli stream)
    Without this field both levels would use DDM (v0.1.2 bug flagged by reviewers)."""


class IncrementalMetric(ABC):
    """A metric computed incrementally over a sliding window."""

    spec: MetricSpec

    def __init__(self, window_size: int) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size

    @abstractmethod
    def per_query_value(self, obs: QueryObservation) -> float:
        """Scalar contribution of a single query, in [0, 1] (or NaN for RS C-subset)."""

    @abstractmethod
    def update(self, obs: QueryObservation) -> float | None:
        """Push one observation. Returns the window value, or None while warming up."""

    @abstractmethod
    def reset(self) -> None: ...


class WindowMeanMetric(IncrementalMetric):
    """value(t) = mean over the current window of per_query_value.  O(1) update."""

    def __init__(self, window_size: int) -> None:
        super().__init__(window_size)
        self._buf: deque[float] = deque()
        self._sum = 0.0
        self._count = 0   # valid (non-NaN) entries in the buffer

    def update(self, obs: QueryObservation) -> float | None:
        v = float(self.per_query_value(obs))
        # NaN values (RS C-subset exclusions) are skipped: they do not enter
        # the window and do not move the window mean.
        if math.isnan(v):
            return None
        self._buf.append(v)
        self._sum += v
        self._count += 1
        if len(self._buf) > self.window_size:
            old = self._buf.popleft()
            self._sum -= old
            self._count -= 1
        if self._count < self.window_size:
            return None   # warm-up
        return self._sum / self._count

    def reset(self) -> None:
        self._buf.clear()
        self._sum = 0.0
        self._count = 0

    @property
    def filled(self) -> bool:
        return self._count >= self.window_size


class WindowDispersionMetric(IncrementalMetric):
    """SS: dispersion of consecutive window means over time.

    Computes the standard deviation of the last W window means and maps it to
    [0, 1] via SS = 1 − std / max_std.

    v0.1.2 — true O(1) implementation
    -----------------------------------
    Previous versions recomputed mean and variance with full passes over the
    deque on every update: O(W).  This version maintains:
      _means_sum    — sum of window means in the buffer
      _means_sum_sq — sum of squared window means
    so that variance is computed in O(1) as Var = E[X²] − E[X]².

    Numerical note: E[X²] − E[X]² is susceptible to cancellation when values
    are large and variance is small.  Here values are means of correctness
    scores ∈ [0, 1], so cancellation is not a practical problem.
    """

    def __init__(self, window_size: int, max_std: float = 0.25) -> None:
        super().__init__(window_size)
        self._inner: deque[float] = deque()
        self._inner_sum = 0.0
        self._means: deque[float] = deque()
        self._means_sum = 0.0
        self._means_sum_sq = 0.0
        self.max_std = max_std

    def update(self, obs: QueryObservation) -> float | None:
        v = float(self.per_query_value(obs))
        # Build inner correctness window.
        self._inner.append(v)
        self._inner_sum += v
        if len(self._inner) > self.window_size:
            self._inner_sum -= self._inner.popleft()
        if len(self._inner) < self.window_size:
            return None  # inner window not yet full

        new_mean = self._inner_sum / self.window_size

        # Sliding window of means: O(1) sum/sum_sq update.
        if len(self._means) == self.window_size:
            old = self._means.popleft()
            self._means_sum -= old
            self._means_sum_sq -= old * old

        self._means.append(new_mean)
        self._means_sum += new_mean
        self._means_sum_sq += new_mean * new_mean

        n = len(self._means)
        if n < 3:
            return 1.0  # too few window means to estimate dispersion

        ex  = self._means_sum / n
        ex2 = self._means_sum_sq / n
        var = max(ex2 - ex * ex, 0.0)   # clamp floating-point negatives to 0
        std = math.sqrt(var)
        return max(0.0, min(1.0, 1.0 - std / self.max_std))

    def reset(self) -> None:
        self._inner.clear()
        self._inner_sum = 0.0
        self._means.clear()
        self._means_sum = 0.0
        self._means_sum_sq = 0.0
