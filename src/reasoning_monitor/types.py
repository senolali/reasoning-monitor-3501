"""Immutable data contracts shared by every module.

Everything else in the package depends on this file and on nothing else in the
package, which keeps the dependency graph acyclic and makes the metric and
detector layers independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

MetricName = str
METRIC_ORDER: tuple[MetricName, ...] = ("CQ", "CS", "RS", "LS", "ES", "SS")

METRIC_LONG_NAME: Mapping[MetricName, str] = {
    "CQ": "Correctness Quality",
    "CS": "Consistency Score",
    "RS": "Robustness Score",
    "LS": "Logical Coherence Score",
    "ES": "Efficiency Score",
    "SS": "Stability Score",
}


class DriftType(str, Enum):
    """Drift shapes used in the stream scenarios."""

    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    RECURRING = "recurring"
    COMPOSITION = "composition"   # v0.2: staged difficulty-composition shift (ST2)
    REVERSE = "reverse"           # v0.2: reverse-composition / recovery test (ST6)


class Distribution(str, Enum):
    """Statistical character of a metric's per-query value.

    The detector assignment is derived from this rather than chosen ad hoc:
    a Bernoulli metric gets a detector built for binary error streams (DDM),
    a bounded continuous one gets a mean-shift test (CUSUM/EWMA), and so on.
    """

    BERNOULLI = "bernoulli"
    BETA = "beta"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"


@dataclass(frozen=True, slots=True)
class Query:
    """One timestamped item in the query stream."""

    qid: str
    t: int
    dataset: str
    domain: str
    prompt: str
    gold_answer: str
    difficulty: str = "base"
    paraphrases: Sequence[str] = field(default_factory=tuple)
    solution_steps: int | None = None
    """Reference-solution operation count when supplied by the dataset."""
    difficulty_rating: float | None = None
    """External continuous difficulty estimate, when supplied by a benchmark."""
    difficulty_rating_std: float | None = None
    difficulty_quantile: float | None = None
    difficulty_model_avg_acc: float | None = None
    family_id: str | None = None
    """Problem-family identifier for grouped trajectory validation."""
    trajectory_level: int | None = None
    """Ordered structural level inside a controlled problem family."""
    trajectory_split: str | None = None
    """Family-level calibration, development or holdout assignment."""


@dataclass(frozen=True, slots=True)
class Sample:
    """Result of a single model call.

    `raw_text` is the completion exactly as the backend returned it, before any
    answer extraction or step splitting.  It exists so a campaign can be
    re-scored offline without re-issuing (and re-paying for) the calls: every
    parsing or metric bug found after the fact is then fixable from the record
    rather than from a new run.  Backends that cannot supply it leave it empty.
    """

    answer: str
    reasoning_steps: Sequence[str]
    latency_s: float
    n_tokens: int
    raw_text: str = ""
    ok: bool = True
    """False when the call failed and `answer` is a placeholder rather than a
    model output.  Failed calls must not be scored as model inconsistency."""
    finish_reason: str | None = None
    """Provider stop reason when available (for example ``stop`` or ``length``)."""


@dataclass(frozen=True, slots=True)
class QueryObservation:
    """The unit the metric layer consumes: one query plus its repeats.

    `runs` holds K repeated calls on the original prompt (CS needs K >= 2);
    `paraphrase_runs` holds L calls on semantically equivalent rewrites (RS).
    """

    query: Query
    runs: Sequence[Sample]
    paraphrase_runs: Sequence[Sample]

    @property
    def primary(self) -> Sample:
        return self.runs[0]


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Window-level metric values at stream position t."""

    t: int
    values: Mapping[MetricName, float]
    composite: float


@dataclass(frozen=True, slots=True)
class Alarm:
    """A drift detection event."""

    t: int
    metric: MetricName
    detector: str
    statistic: float
    level: str = "drift"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Definition of one stream scenario.

    v0.2 (E2H-GSM8K controlled-composition design):
      `regimes` is an ordered list of (difficulty, n_queries) segments.
      Within each regime, queries are drawn by RANDOM PERMUTATION of that
      difficulty pool — never sorted — so within-regime order carries no
      difficulty signal.  `change_points` (derived) marks each regime boundary.

    The legacy fields (drift_type, change_point, n_queries, transition_width)
    are kept for backward compatibility with the S0–S6 scenarios and the
    detectors that read a single change_point.
    """

    name: str
    drift_type: DriftType
    change_point: int
    n_queries: int
    transition_width: int = 0
    description: str = ""
    # v0.2 staged-regime fields:
    regimes: Sequence[tuple[str, int]] = field(default_factory=tuple)
    """Ordered (difficulty, count) segments. Empty for legacy S0–S6 scenarios."""

    @property
    def change_points(self) -> list[int]:
        """Regime boundaries (1-based stream positions where difficulty changes).

        For a regimes list [(easy,50),(medium,50),(hard,50)] this returns
        [50, 100] — the two transition points.  A single-regime (stationary)
        stream returns [] (no change point).
        """
        if not self.regimes:
            return [self.change_point] if self.change_point > 0 else []
        bounds, acc = [], 0
        for _diff, n in self.regimes[:-1]:
            acc += n
            bounds.append(acc)
        return bounds
