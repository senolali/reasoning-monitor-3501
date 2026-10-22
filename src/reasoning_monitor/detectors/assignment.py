"""Metric-to-detector assignment — three named strategies.

v0.1.3 design
-------------
The proposal submission contained two inconsistent detector tables.  Rather
than picking one and discarding the other, v0.1.3 treats the choice of
assignment strategy as an **experimental variable** to be tested under H2
("do metric-specific detectors outperform a generic baseline?").

Three named strategies are provided:

  "distribution_based"  — primary strategy; derived from each metric's
    statistical distribution.  This is a testable hypothesis, not an
    established result: Beta→CUSUM is motivated by SPRT theory but
    requires empirical validation against proposal_mapping and adwin_all.

  "proposal_mapping"    — the mapping stated in the project proposal
    (best-guess single consistent interpretation of the two tables):
      CQ→CUSUM, CS→ADWIN, RS→DDM, LS→PageHinkley, ES→EWMA, SS→ADWIN

  "adwin_all"           — distribution-agnostic Hoeffding baseline.
    Applied to every metric; the anchor for H2 comparisons.

The ablation in the paper should compare all three strategies on detection
delay and false-alarm rate, reporting which performs best per scenario type.

CQ two-level detection (v0.1.3 fix)
--------------------------------------
MetricSpec.windowed_distribution overrides the distribution used for the
PRIMARY (windowed) detector, independently of the per-query distribution used
for the secondary raw detector.

  CQ: distribution=BERNOULLI → raw DDM  (per-query 0/1 stream)
      windowed_distribution=BETA → primary CUSUM  (W-step window mean)

v0.1.2 bug: both levels mapped to DDM because THEORETICAL_ASSIGNMENT
[Bernoulli]→DDM was applied to the windowed level too.  This is now fixed.
"""

from __future__ import annotations

from collections.abc import Mapping

from reasoning_monitor.detectors.detectors import DriftDetector, build_detector
from reasoning_monitor.metrics.base import IncrementalMetric
from reasoning_monitor.types import Distribution

# ── distribution-based assignment (primary strategy) ─────────────────────────

THEORETICAL_ASSIGNMENT: Mapping[Distribution, str] = {
    Distribution.BERNOULLI: "ddm",     # raw stream only; windowed CQ uses BETA→CUSUM
    Distribution.BETA:      "cusum",
    Distribution.NORMAL:    "ewma",
    Distribution.LOGNORMAL: "page_hinkley",
}

ASSIGNMENT_RATIONALE: Mapping[Distribution, str] = {
    Distribution.BERNOULLI: "DDM is defined on binary error streams; "
        "used for the raw per-query CQ/RS stream only.",
    Distribution.BETA: "CUSUM is the SPRT-derived test for a shift in the "
        "mean of a bounded [0,1] signal.  'Beta' here means bounded continuous "
        "with support [0,1], not that the variable is proven Beta-distributed.  "
        "This mapping is a working approximation evaluated empirically under H2.",
    Distribution.NORMAL: "EWMA control charts for small persistent shifts "
        "in an approximately normal signal (window-mean sequence of SS).",
    Distribution.LOGNORMAL: "Page-Hinkley is one-sided; kept for ablation "
        "comparison with CUSUM on log-compressed cost measures.",
}

# ── proposal mapping (as-stated in the project proposal) ─────────────────────

PROPOSAL_ASSIGNMENT: Mapping[str, str] = {
    "CQ": "cusum",
    "CS": "adwin",
    "RS": "ddm",
    "LS": "page_hinkley",
    "ES": "ewma",
    "SS": "adwin",
}

# ── three named strategies ────────────────────────────────────────────────────

DETECTOR_STRATEGIES = {
    "distribution_based": None,          # uses THEORETICAL_ASSIGNMENT via distribution
    "proposal_mapping":   PROPOSAL_ASSIGNMENT,
    "adwin_all":          {m: "adwin" for m in ("CQ", "CS", "RS", "LS", "ES", "SS")},
}

# Opt-in strategies with the corrected ADWIN (``adwin_v``; see detectors.ADWINVariance).
# Kept separate so that every frozen analysis iterating DETECTOR_STRATEGIES is
# unchanged; new protocols must name these strategies explicitly.
CORRECTED_STRATEGIES = {
    "proposal_mapping_v": {m: ("adwin_v" if d == "adwin" else d)
                           for m, d in PROPOSAL_ASSIGNMENT.items()},
    "adwin_all_v":        {m: "adwin_v" for m in ("CQ", "CS", "RS", "LS", "ES", "SS")},
}


def assign_detector(
    metric: IncrementalMetric,
    overrides: Mapping[str, dict] | None = None,
    strategy: str = "distribution_based",
) -> DriftDetector:
    """Build the windowed detector for a metric.

    Priority (highest to lowest):
      1. overrides[metric.name]  — explicit per-metric override from config
      2. strategy                — one of DETECTOR_STRATEGIES
      3. distribution-based      — fallback when strategy=="distribution_based"

    For the windowed detector, MetricSpec.windowed_distribution is used in
    preference to MetricSpec.distribution when available (CQ two-level fix).
    """
    overrides = overrides or {}
    name = metric.spec.name

    # Explicit override wins unconditionally.
    if name in overrides:
        cfg = dict(overrides[name])
        det_name = cfg.pop("name")
        return build_detector(det_name, **cfg)

    # Named strategy lookup.
    strategy_map = {**DETECTOR_STRATEGIES, **CORRECTED_STRATEGIES}.get(strategy)
    if strategy_map is not None and name in strategy_map:
        return build_detector(strategy_map[name])

    # Distribution-based fallback (also the default for "distribution_based").
    # Use windowed_distribution when set (CQ: BETA→CUSUM instead of BERNOULLI→DDM).
    dist = metric.spec.windowed_distribution or metric.spec.distribution
    det_name = THEORETICAL_ASSIGNMENT[dist]
    return build_detector(det_name)


def assign_raw_detector(metric: IncrementalMetric) -> DriftDetector:
    """Build the raw (per-query) detector for metrics with raw_detection=True.

    Always uses the per-query `distribution` (not windowed_distribution),
    so CQ always gets DDM regardless of strategy.  The raw detector is not
    affected by detector_strategy — it tests a different hypothesis (event-level
    vs aggregate-level drift) and should always use DDM for Bernoulli metrics.
    """
    det_name = THEORETICAL_ASSIGNMENT[metric.spec.distribution]
    return build_detector(det_name)


def assignment_table(strategy: str = "distribution_based") -> list[dict]:
    """Rows for the LaTeX table documenting a strategy's metric→detector pairing."""
    rows = []
    from reasoning_monitor.metrics.dimensions import build_metric
    for mname in ("CQ", "CS", "RS", "LS", "ES", "SS"):
        m = build_metric(mname, window_size=5)
        det = assign_detector(m, strategy=strategy)
        rows.append({"metric": mname, "detector": det.name, "strategy": strategy})
    return rows
