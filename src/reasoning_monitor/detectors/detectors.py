"""Drift detectors.

All five operate on a stream of scalar metric values and emit an Alarm when the
underlying mean has shifted. They share one interface so that the assignment
layer can pair any detector with any metric, and so that the ablation can swap
them without touching the monitor.

Convention: metrics here are quality scores where HIGHER IS BETTER, so drift of
interest is a DOWNWARD shift. Detectors that were designed for error rates
(DDM) receive 1 - value; this is done inside the detector, not by the caller,
so that the polarity convention lives in exactly one place.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from reasoning_monitor.types import Alarm


class DriftDetector(ABC):
    name: str = "base"

    def __init__(self, warmup: int = 20) -> None:
        self.warmup = warmup
        self.n = 0

    @abstractmethod
    def _update(self, value: float) -> tuple[bool, float]:
        """Returns (alarm_fired, statistic)."""

    def update(self, value: float, t: int, metric: str) -> Alarm | None:
        self.n += 1
        fired, stat = self._update(float(value))
        if fired and self.n > self.warmup:
            alarm = Alarm(t=t, metric=metric, detector=self.name, statistic=stat)
            self.reset()
            return alarm
        return None

    @abstractmethod
    def reset(self) -> None:
        self.n = 0


class CUSUM(DriftDetector):
    """Cumulative sum test for a persistent shift in the mean.

    Tracks the running mean and accumulates negative deviations; `drift` is the
    allowance (slack) below which fluctuation is ignored.
    """

    name = "cusum"

    def __init__(self, threshold: float = 0.15, drift: float = 0.01, warmup: int = 20):
        super().__init__(warmup)
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def _update(self, value: float) -> tuple[bool, float]:
        self._count += 1
        self._mean += (value - self._mean) / self._count
        dev = self._mean - value  # positive when quality drops
        self._g = max(0.0, self._g + dev - self.drift)
        return self._g > self.threshold, self._g

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._count = 0
        self._g = 0.0


class PageHinkley(DriftDetector):
    """Page-Hinkley test — sequential change detection on the running mean.

    Implementation for detecting DOWNWARD quality shifts.

    v0.1.2 corrected formulation:
        cum_t = max(0, cum_{t-1} + (μ̄_{t-1} − x_t − δ))
        alarm when cum_t > λ

    Key design notes:
    - Uses the PREVIOUS step's running mean (not current) as the reference to
      avoid feedback where the current value immediately affects its own detection.
    - max(0, ...) reset: when quality is fine (x_t ≥ μ̄), the accumulator
      naturally resets, making detection causal (no window of past data needed).
    - δ (delta) is the minimum detectable shift; fluctuations smaller than δ
      below the running mean are ignored.

    v0.1.0/v0.1.1 had two bugs:
      1. Alarm condition was `self._cum > threshold` ← correct direction but
         the running-mean-with-alpha made it a non-standard variant.
      2. `_max_cum` and `stat = max_cum - cum` were computed but not used in
         the alarm; the standard two-sided PH statistic (max − cum) fires when
         the accumulated sum FALLS after having risen, i.e. it detects recovery,
         not the onset of degradation — the opposite of what we want.
    """

    name = "page_hinkley"

    def __init__(
        self,
        threshold: float = 0.25,
        delta: float = 0.005,
        warmup: int = 20,
    ):
        super().__init__(warmup)
        self.threshold = threshold
        self.delta = delta
        self.reset()

    def _update(self, value: float) -> tuple[bool, float]:
        prev_mean = self._mean
        self._count += 1
        self._mean += (value - self._mean) / self._count
        # Accumulate downward deviations using previous mean as reference.
        # max(0, ...) resets the accumulator when quality is fine.
        self._cum = max(0.0, self._cum + (prev_mean - value - self.delta))
        return self._cum > self.threshold, self._cum

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._count = 0
        self._cum = 0.0


class EWMA(DriftDetector):
    """Exponentially weighted moving average control chart.

    Good at small persistent shifts, which is the gradual-drift regime.
    """

    name = "ewma"

    def __init__(self, lam: float = 0.2, control_limit: float = 3.0, warmup: int = 20):
        super().__init__(warmup)
        if not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        self.lam = lam
        self.control_limit = control_limit
        self.reset()

    def _update(self, value: float) -> tuple[bool, float]:
        self._count += 1
        prev_mean = self._mean
        self._mean += (value - self._mean) / self._count
        self._m2 += (value - prev_mean) * (value - self._mean)
        if self._z is None:
            self._z = value
            return False, 0.0
        self._z = self.lam * value + (1 - self.lam) * self._z
        if self._count < 3:
            return False, 0.0
        var = self._m2 / max(self._count - 1, 1)
        std = math.sqrt(max(var, 1e-12))
        sigma_z = std * math.sqrt(self.lam / (2 - self.lam))
        lower = self._mean - self.control_limit * sigma_z
        return self._z < lower, self._z

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0
        self._z = None


class DDM(DriftDetector):
    """Drift Detection Method (Gama et al.).

    Built for binary error streams, so it consumes 1 - value: a quality score of
    0.9 becomes an error rate of 0.1. Emits a warning level before drift.
    """

    name = "ddm"

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0,
                 warmup: int = 30, min_count: int = 30):
        super().__init__(warmup)
        self.warning_level = warning_level
        self.drift_level = drift_level
        # min_count: DDM only tracks p_min/s_min and fires after this many
        # samples. On short binary streams the running std shrinks fast, so a
        # premature p_min lock-in causes false alarms on stationary data.
        # Requiring a minimum count before arming the detector fixes this.
        self.min_count = min_count
        self.reset()

    def _update(self, value: float) -> tuple[bool, float]:
        error = 1.0 - value
        self._count += 1
        self._p += (error - self._p) / self._count
        s = math.sqrt(max(self._p * (1.0 - self._p) / self._count, 0.0))
        stat = self._p + s
        # Only start tracking the minimum after min_count samples, so the
        # baseline p_min/s_min reflect a stable estimate rather than the first
        # few noisy observations.
        if self._count < self.min_count:
            return False, stat
        if stat < self._p_min + self._s_min:
            self._p_min, self._s_min = self._p, s
        # Guard against a degenerate s_min that makes the threshold trivial.
        if self._s_min < 1e-3:
            return False, stat
        return stat > self._p_min + self.drift_level * self._s_min, stat

    def reset(self) -> None:
        self.n = 0
        self._p = 0.0
        self._count = 0
        self._p_min = float("inf")
        self._s_min = float("inf")


class ADWIN(DriftDetector):
    """Adaptive windowing (Bifet & Gavaldà), bucketed approximation.

    Keeps a window of recent values and looks for a split point whose two sides
    differ by more than a Hoeffding-style bound. This is the practical bucketed
    variant rather than the full exponential-histogram version: it checks a
    bounded number of split points, which caps the per-step cost.
    """

    name = "adwin"

    def __init__(self, delta: float = 0.002, max_window: int = 200, warmup: int = 20):
        super().__init__(warmup)
        self.delta = delta
        self.max_window = max_window
        self.reset()

    def _update(self, value: float) -> tuple[bool, float]:
        self._window.append(value)
        if len(self._window) > self.max_window:
            self._window.pop(0)
        n = len(self._window)
        if n < 20:
            return False, 0.0

        total = sum(self._window)
        best = 0.0
        # Check a limited set of split points rather than all n-1 of them.
        step = max(1, n // 20)
        left_sum = 0.0
        for i in range(step, n - step + 1, step):
            left_sum = sum(self._window[:i])
            n0, n1 = i, n - i
            if n0 < 5 or n1 < 5:
                continue
            m0 = left_sum / n0
            m1 = (total - left_sum) / n1
            diff = m0 - m1  # positive when the recent half is worse
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt((2.0 / m) * math.log(2.0 * n / self.delta))
            if diff > eps:
                best = max(best, diff)
        return best > 0.0, best

    def reset(self) -> None:
        self.n = 0
        self._window: list[float] = []


class ADWINVariance(ADWIN):
    """ADWIN with the variance-based cut bound of Bifet & Gavaldà (2007).

    The historical ``ADWIN`` above uses eps = sqrt((2/m) ln(2n/delta)), which
    omits the window variance. For n <= ~100 values in [0, 1], that eps
    exceeds every attainable mean difference, so it can never alarm
    (CDR_STREAM_STAGE_A_REPORT.md section 4). This class uses the paper's
    practical bound

        eps = sqrt((2/m) * var_W * ln(2/delta')) + (2/(3m)) * ln(2/delta'),
        m = 1 / (1/n0 + 1/n1),  delta' = delta / n,

    with var_W the variance of the whole window. Split points, minimum side
    length, one-sided (decrease) direction, warm-up and reset are unchanged.
    The historical class is kept as is so that frozen results reproduce; this
    detector is opt-in under the name ``adwin_v``.
    """

    name = "adwin_v"

    def _update(self, value: float) -> tuple[bool, float]:
        self._window.append(value)
        if len(self._window) > self.max_window:
            self._window.pop(0)
        n = len(self._window)
        if n < 20:
            return False, 0.0
        total = sum(self._window)
        mean = total / n
        variance = sum((x - mean) ** 2 for x in self._window) / n
        log_term = math.log(2.0 * n / self.delta)
        best = 0.0
        step = max(1, n // 20)
        for i in range(step, n - step + 1, step):
            n0, n1 = i, n - i
            if n0 < 5 or n1 < 5:
                continue
            left_sum = sum(self._window[:i])
            diff = left_sum / n0 - (total - left_sum) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt((2.0 / m) * variance * log_term) + (2.0 / (3.0 * m)) * log_term
            if diff > eps:
                best = max(best, diff)
        return best > 0.0, best


DETECTOR_CLASSES = {
    "cusum": CUSUM,
    "page_hinkley": PageHinkley,
    "ewma": EWMA,
    "ddm": DDM,
    "adwin": ADWIN,
    "adwin_v": ADWINVariance,
}


def build_detector(name: str, **kwargs) -> DriftDetector:
    if name not in DETECTOR_CLASSES:
        raise ValueError(f"unknown detector: {name!r}; expected one of {list(DETECTOR_CLASSES)}")
    return DETECTOR_CLASSES[name](**kwargs)
