"""Adapter layer: wraps rqeval metric classes into the sliding-window interface.

This module is the sole point of contact between reasoning-monitor and the
published rqeval package (pip install rqeval).  All six metric scoring
functions are delegated to rqeval; reasoning-monitor contributes only the
online streaming aggregation and drift-detection layer.

Paper claim:
    "We use the six reasoning-quality metrics implemented in RQEval without
    redefining their scoring functions.  Our contribution is the online
    temporal monitoring and drift-detection layer built on top of these
    per-query reasoning-quality signals."

Design
------
Five metrics (CQ, CS, RS, LS, SS) produce a scalar per query and are wrapped
as WindowMeanMetric subclasses.  Efficiency (ES) is the exception: rqeval's
EfficiencyMetric normalizes token counts via batch min-max, which is undefined
for a single observation (T_min == T_max → T' = 0 → ES collapses to CQ).
EfficiencyWindowAdapter therefore passes the *entire current window* to
rqeval's compute_per_instance() and returns the current item's efficiency
computed under the window's token range.  This matches rqeval's intent while
respecting the streaming constraint.

Deviations from rqeval
-----------------------
See DEVIATIONS_FROM_PROPOSAL.md §20 for the full list.  The main ones:
  * RS: rqeval returns 0 for non-C items; the adapter returns NaN so the
    WindowMeanMetric's NaN-skipping logic excludes them from the window mean.
    This preserves the C-subset semantics (average only over correct items).
  * ES: rqeval normalizes over a batch; the adapter normalizes over the window.
    The effective range shrinks in the early warm-up period.
  * E2H-AMC: rqeval formulas are unchanged, but answer canonicalization uses the
    dataset-specific symbolic matcher because rqeval treats non-numeric LaTeX
    answers as free text.
  * Optional semantic answer policy: CS/RS retain rqeval's pairwise and
    C-subset formulas but replace surface answer identity with a prospectively
    frozen dataset-aware equivalence relation. The default is disabled.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

from reasoning_monitor.metrics.base import MetricSpec, WindowMeanMetric
from reasoning_monitor.metrics.dimensions import answers_equal
from reasoning_monitor.types import Distribution, QueryObservation

# ── rqeval imports (requires: pip install rqeval) ────────────────────────────
try:
    from metrics.accuracy import AccuracyMetric as _CQMetric
    from metrics.consistency import ConsistencyMetric as _CSMetric
    from metrics.efficiency import EfficiencyMetric as _ESMetric
    from metrics.explainability import ExplainabilityMetric as _SSMetric
    from metrics.logical_consistency import LogicalConsistencyMetric as _LSMetric
    from metrics.robustness import RobustnessMetric as _RSMetric
    _RQEVAL_AVAILABLE = True
except ImportError as _e:
    _RQEVAL_AVAILABLE = False
    _RQEVAL_IMPORT_ERROR = str(_e)


def _require_rqeval() -> None:
    if not _RQEVAL_AVAILABLE:
        raise ImportError(
            "rqeval is required for metric scoring. "
            "Install it with: pip install rqeval\n"
            f"Original error: {_RQEVAL_IMPORT_ERROR}"
        )


# ── CQ — Correctness ─────────────────────────────────────────────────────────

class CQAdapter(WindowMeanMetric):
    """CQ via rqeval AccuracyMetric: type-aware canonical answer matching."""

    spec = MetricSpec(
        "CQ", "Correctness (rqeval AccuracyMetric)",
        Distribution.BERNOULLI,
        raw_detection=True,
        windowed_distribution=Distribution.BETA,
    )

    def __init__(self, window_size: int) -> None:
        super().__init__(window_size)
        _require_rqeval()
        self._metric = _CQMetric()

    def per_query_value(self, obs: QueryObservation) -> float:
        if obs.query.dataset == "e2h_amc":
            return float(answers_equal(
                obs.primary.answer, obs.query.gold_answer, obs.query.dataset
            ))
        scores = self._metric.compute_per_instance(
            [obs.primary.answer], [obs.query.gold_answer]
        )
        return float(scores[0])


# ── CS — Consistency ─────────────────────────────────────────────────────────

class CSAdapter(WindowMeanMetric):
    """CS via rqeval ConsistencyMetric: pairwise answer agreement across K runs."""

    spec = MetricSpec("CS", "Consistency (rqeval ConsistencyMetric)", Distribution.BETA)

    def __init__(
        self,
        window_size: int,
        k_runs: int = 5,
        semantic_answer_policy: str | None = None,
    ) -> None:
        super().__init__(window_size)
        _require_rqeval()
        self.k_runs = k_runs
        self.semantic_answer_policy = semantic_answer_policy
        self._metric = _CSMetric({"consistency_runs": k_runs})

    def per_query_value(self, obs: QueryObservation) -> float:
        answers = [r.answer for r in obs.runs[:self.k_runs]]
        if self.semantic_answer_policy:
            from reasoning_monitor.metrics.semantic_answers import (
                semantic_answers_equal,
            )

            if len(answers) < 2:
                return 1.0
            pairs = list(combinations(answers, 2))
            matches = sum(
                semantic_answers_equal(
                    left,
                    right,
                    obs.query.dataset,
                    self.semantic_answer_policy,
                )
                for left, right in pairs
            )
            return matches / len(pairs)
        if obs.query.dataset == "e2h_amc":
            if len(answers) < 2:
                return 1.0
            pairs = list(combinations(answers, 2))
            matches = sum(
                answers_equal(left, right, obs.query.dataset) for left, right in pairs
            )
            return matches / len(pairs)
        return float(self._metric.compute_instance(answers, gold=obs.query.gold_answer))


# ── RS — Robustness ──────────────────────────────────────────────────────────

class RSAdapter(WindowMeanMetric):
    """RS via rqeval RobustnessMetric: paraphrase stability (C-subset).

    Uses the public `metrics.answer_extraction.is_correct` function directly
    for the C-subset check, avoiding the private `AccuracyMetric._is_correct`
    method which could break across rqeval versions.
    """

    spec = MetricSpec("RS", "Robustness (rqeval RobustnessMetric, C-subset)", Distribution.BETA)

    def __init__(
        self,
        window_size: int,
        l_paraphrases: int = 3,
        semantic_answer_policy: str | None = None,
    ) -> None:
        super().__init__(window_size)
        _require_rqeval()
        self.l_paraphrases = l_paraphrases
        self.semantic_answer_policy = semantic_answer_policy
        self._metric = _RSMetric({"robustness_perturbations": l_paraphrases})

    def per_query_value(self, obs: QueryObservation) -> float:
        from metrics.answer_extraction import is_correct  # rqeval-provided utility
        gold = obs.query.gold_answer
        # C-subset: return NaN for wrong primary answers so WindowMeanMetric
        # skips them, matching rqeval's C-subset average semantics.
        # is_correct() is the same function rqeval's own AccuracyMetric calls;
        # using it directly avoids depending on the private _is_correct method.
        primary_correct = (
            answers_equal(obs.primary.answer, gold, obs.query.dataset)
            if obs.query.dataset == "e2h_amc"
            else is_correct(obs.primary.answer, gold)
        )
        if not primary_correct:
            return float("nan")
        perturb_answers = [r.answer for r in obs.paraphrase_runs[:self.l_paraphrases]]
        if self.semantic_answer_policy:
            from reasoning_monitor.metrics.semantic_answers import (
                semantic_rs_answer_correct,
            )

            if not perturb_answers:
                return 1.0
            return sum(
                semantic_rs_answer_correct(
                    answer,
                    gold,
                    obs.primary.answer,
                    obs.query.dataset,
                    self.semantic_answer_policy,
                )
                for answer in perturb_answers
            ) / len(perturb_answers)
        if obs.query.dataset == "e2h_amc":
            if not perturb_answers:
                return 1.0
            return sum(
                answers_equal(answer, gold, obs.query.dataset)
                for answer in perturb_answers
            ) / len(perturb_answers)
        scores = self._metric.compute_per_instance(
            [obs.primary.answer], [perturb_answers], [gold]
        )
        return float(scores[0])


# ── LS — Logical Coherence ───────────────────────────────────────────────────

class LSAdapter(WindowMeanMetric):
    """LS via rqeval LogicalConsistencyMetric: NLI contradiction detection.

    rqeval loads the NLI pipeline lazily and swallows every loading error
    (`except Exception: self._nli_pipeline = None`), after which every step pair
    is scored `neutral` and LS is a constant 1.0 — with no warning.  A run can
    therefore look healthy while the metric is dead.  `scorer_status()` forces
    the load and reports what actually happened, so `ReasoningMonitor.preflight()`
    can refuse the run instead of discovering it in the results.
    """

    spec = MetricSpec("LS", "Logical Coherence (rqeval LogicalConsistencyMetric)", Distribution.BETA)

    def __init__(self, window_size: int, nli_model: str | None = None) -> None:
        super().__init__(window_size)
        _require_rqeval()
        self.nli_model = nli_model
        self._metric = _LSMetric(nli_model=nli_model)

    def scorer_status(self) -> dict:
        """Force the NLI pipeline to load and report whether it is real.

        Returns {"ready": bool, "scorer": str, "model": str}.  `ready` is False
        when rqeval fell back to its no-contradiction heuristic.
        """
        try:
            self._metric._ensure_nli()
            ready = self._metric._nli_pipeline is not None
        except Exception:            # pragma: no cover - defensive
            ready = False
        return {
            "ready": ready,
            "scorer": "nli" if ready else "heuristic-fallback",
            "model": getattr(self._metric, "nli_model_name", "unknown"),
        }

    def per_query_value(self, obs: QueryObservation) -> float:
        steps = list(obs.primary.reasoning_steps)
        return float(self._metric.compute_trace(steps))


# ── ES — Efficiency ───────────────────────────────────────────────────────────

class ESAdapter(WindowMeanMetric):
    """ES via rqeval EfficiencyMetric: harmonic mean of correctness and conciseness.

    rqeval normalizes token counts via batch min-max.  For streaming, we pass
    the full current window to rqeval's compute_per_instance() and return
    the last item's efficiency computed under the window token range.
    During warm-up (buffer not full), we fall back to a fixed reference.
    """

    spec = MetricSpec("ES", "Efficiency (rqeval EfficiencyMetric, window-range norm)", Distribution.BETA)

    def __init__(self, window_size: int, reference_tokens: float = 256.0) -> None:
        # Don't call super().__init__ — we manage the buffer ourselves
        # because we need to pass the full window to rqeval.
        from metrics.accuracy import AccuracyMetric

        from reasoning_monitor.metrics.base import IncrementalMetric
        IncrementalMetric.__init__(self, window_size)
        _require_rqeval()
        self.reference_tokens = reference_tokens
        self._metric = _ESMetric()
        self._cq_metric = AccuracyMetric()
        self._cq_buf: deque[float] = deque()
        self._tok_buf: deque[int] = deque()

    def _item_cq(self, obs: QueryObservation) -> float:
        if obs.query.dataset == "e2h_amc":
            return float(answers_equal(
                obs.primary.answer, obs.query.gold_answer, obs.query.dataset
            ))
        return float(self._cq_metric.compute_per_instance(
            [obs.primary.answer], [obs.query.gold_answer]
        )[0])

    def per_query_value(self, obs: QueryObservation) -> float:
        """ES for this item alone, normalized over the window it arrives in.

        Before this fix the method called a private rqeval helper, discarded the
        result and returned NaN, which made the `ES_raw` column of every
        per-query record entirely NaN — the one column `pilot_check.py` reads to
        judge whether ES is alive.  It now returns the item's own efficiency
        under the current window's token range (the same range `update()` uses),
        leaving `update()` responsible for the window mean.
        """
        cq  = self._item_cq(obs)
        tok = int(obs.primary.n_tokens)
        cq_list  = [*self._cq_buf, cq]
        tok_list = [*self._tok_buf, tok]
        return float(self._metric.compute_per_instance(cq_list, tok_list)[-1])

    def update(self, obs: QueryObservation) -> float | None:
        """Compute ES for the new item using the full window for min-max range."""
        cq  = self._item_cq(obs)
        tok = int(obs.primary.n_tokens)

        self._cq_buf.append(cq)
        self._tok_buf.append(tok)
        if len(self._cq_buf) > self.window_size:
            self._cq_buf.popleft()
            self._tok_buf.popleft()

        if len(self._cq_buf) < self.window_size:
            return None  # warm-up

        # Pass entire window to rqeval for min-max normalization.
        cq_list  = list(self._cq_buf)
        tok_list = list(self._tok_buf)
        per_inst = self._metric.compute_per_instance(cq_list, tok_list)
        # Return the window mean (same as rqeval's compute()).
        return float(sum(per_inst) / len(per_inst))

    def reset(self) -> None:
        self.n = 0  # required by IncrementalMetric
        self._cq_buf.clear()
        self._tok_buf.clear()


# ── SS — Stability ────────────────────────────────────────────────────────────

class SSAdapter(WindowMeanMetric):
    """SS via rqeval ExplainabilityMetric: pairwise BERTScore across K traces."""

    spec = MetricSpec("SS", "Stability (rqeval ExplainabilityMetric, BERTScore)", Distribution.BETA)

    def __init__(self, window_size: int, k_runs: int = 5, bertscore_model: str | None = None) -> None:
        super().__init__(window_size)
        _require_rqeval()
        self.k_runs = k_runs
        self.bertscore_model = bertscore_model
        self._metric = _SSMetric(
            config={"stability_runs": k_runs},
            bertscore_model=bertscore_model,
        )

    def scorer_status(self) -> dict:
        """Report whether BERTScore is available or rqeval falls back to Jaccard.

        Same failure mode as LS: the fallback is silent, and a Jaccard SS over
        templated traces is near-constant, which reads as "no drift" rather than
        as "no metric".
        """
        try:
            ready = bool(self._metric._check_bertscore())
        except Exception:            # pragma: no cover - defensive
            ready = False
        return {
            "ready": ready,
            "scorer": "bertscore" if ready else "jaccard-fallback",
            "model": self.bertscore_model or "rqeval-default",
        }

    def per_query_value(self, obs: QueryObservation) -> float:
        traces = [
            " ".join(r.reasoning_steps) if r.reasoning_steps else r.answer
            for r in obs.runs[:self.k_runs]
        ]
        return max(0.0, min(1.0, float(self._metric.compute_instance(traces))))


# ── Registry ──────────────────────────────────────────────────────────────────

RQEVAL_ADAPTER_CLASSES = {
    "CQ": CQAdapter,
    "CS": CSAdapter,
    "RS": RSAdapter,
    "LS": LSAdapter,
    "ES": ESAdapter,
    "SS": SSAdapter,
}


def build_rqeval_metric(name: str, window_size: int, **kwargs):
    """Build a rqeval-backed metric adapter."""
    if name not in RQEVAL_ADAPTER_CLASSES:
        raise ValueError(f"unknown metric: {name!r}; expected one of {list(RQEVAL_ADAPTER_CLASSES)}")
    return RQEVAL_ADAPTER_CLASSES[name](window_size, **kwargs)
