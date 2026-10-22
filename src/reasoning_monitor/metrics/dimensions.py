"""The six reasoning-quality dimensions.

v0.1.2 changes
--------------
* Dataset-aware answer normalization: GSM8K, MATH, ARC-Challenge, StrategyQA
  each need different extraction logic.  Generic first-number regex was too
  fragile (e.g. "first 3 steps gives 7" → 3 instead of 7).
* K and L are now constructor arguments (default 3) so MonitorConfig.k_repeats
  and l_paraphrases actually propagate to the metrics.  v0.1.0/v0.1.1 had a
  bug where the module-level K_RUNS/L_PARAPHRASES constants shadowed the config
  values, meaning config changes were silently ignored.
* CQ carries raw_detection=True so the monitor can run a secondary DDM on the
  raw per-query Bernoulli stream alongside the windowed CUSUM on the mean.
  Feeding a windowed mean to DDM violates DDM's Binomial assumptions; the
  two-level design fixes this without changing the window metric pipeline.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations

from reasoning_monitor.metrics.base import (
    MetricSpec,
    WindowMeanMetric,
)
from reasoning_monitor.types import Distribution, QueryObservation

# Defaults; overridden via constructor arguments (see K/L bug fix note above).
_DEFAULT_K = 3
_DEFAULT_L = 3

# ── answer normalisation ──────────────────────────────────────────────────────

_NORM = re.compile(r"[^a-z0-9.\-]+")
_NUM = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+\-]?\d+)?")
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})+)\}")
_MCHOICE = re.compile(r"\b([a-d])\b")
# Fraction patterns imported from GPT v0.2.0 review: enables cross-format
# matching (gold=\frac{1}{2} matches prediction=0.5).
_LATEX_FRAC = re.compile(r"\\?frac\{(-?\d+)\}\{(-?\d+)\}")
_PLAIN_FRAC = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
# Explicit answer markers (####, "Answer:", "final answer") — faster than
# scanning a full reasoning trace for the last number.
_MARKERS = re.compile(r"(?:final answer|answer is|answer:|####)\s*([^\n]+)", re.IGNORECASE)


def _last_balanced_boxed(text: str) -> str | None:
    """Return the final balanced ``\boxed{...}`` expression, if present."""
    marker = r"\boxed{"
    starts = [index for index in range(len(text)) if text.startswith(marker, index)]
    for start in reversed(starts):
        depth = 1
        content_start = start + len(marker)
        for position in range(content_start, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:position].strip()
    return None


def _amc_final_candidate(text: str) -> str:
    boxed = _last_balanced_boxed(text)
    if boxed is not None:
        return boxed
    markers = _MARKERS.findall(text)
    candidate = markers[-1] if markers else text
    # Providers sometimes emit nested labels such as
    # ``Final answer: Answer: 36``. Peel all leading answer labels so the raw
    # completion and the client's extracted answer reach the same candidate.
    for _ in range(3):
        nested = _MARKERS.findall(candidate)
        if not nested:
            break
        candidate = nested[-1]
    candidate = candidate.strip().strip("$").strip()
    if candidate.startswith(r"\(") and candidate.endswith(r"\)"):
        candidate = candidate[2:-2]
    if candidate.startswith(r"\[") and candidate.endswith(r"\]"):
        candidate = candidate[2:-2]
    candidate = candidate.strip()
    candidate = re.sub(r"^\}+\s*", "", candidate)
    candidate = re.sub(r"\s*\\\]\s*$", "", candidate)
    return candidate.strip().rstrip(".;")


def _amc_surface(text: str) -> str:
    """Conservative surface form used when symbolic parsing is inapplicable."""
    value = _amc_final_candidate(text).lower()
    for old, new in ((r"\dfrac", r"\frac"), (r"\tfrac", r"\frac")):
        value = value.replace(old, new)
    for token in (r"\left", r"\right", r"\,", r"\;", r"\!", r"\quad"):
        value = value.replace(token, "")
    value = value.replace("−", "-").replace("×", r"\cdot")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<![\w}])(-?\d+)\\frac", r"\1+\\frac", value)
    return re.sub(r"\^\{(-?\d+)\}", r"^\1", value)


def _split_amc_sequence(surface: str) -> list[str] | None:
    """Split ordered tuples/lists while leaving scalar commas untouched."""

    def split_at_depth_zero(value: str) -> list[str]:
        parts: list[str] = []
        start = 0
        depth = 0
        for index, char in enumerate(value):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(value[start:index])
                start = index + 1
        parts.append(value[start:])
        return parts

    direct = split_at_depth_zero(surface)
    if len(direct) > 1:
        return direct
    if surface.startswith("(") and surface.endswith(")"):
        inner = split_at_depth_zero(surface[1:-1])
        if len(inner) > 1:
            return inner
    return None


def _amc_scalar_equal(surface_a: str, surface_b: str) -> bool:
    if surface_a == surface_b:
        return True
    expression_a = _parse_amc_expression(surface_a)
    expression_b = _parse_amc_expression(surface_b)
    if expression_a is None or expression_b is None:
        return False
    try:
        import sympy

        return bool(sympy.simplify(expression_a - expression_b) == 0)
    except Exception:
        return False


def symbolic_answers_equal(a: str, b: str) -> bool:
    """Compare scalar or ordered symbolic answers without dataset extraction."""
    surface_a, surface_b = _amc_surface(a), _amc_surface(b)
    sequence_a = _split_amc_sequence(surface_a)
    sequence_b = _split_amc_sequence(surface_b)
    if sequence_a is not None or sequence_b is not None:
        if sequence_a is None or sequence_b is None or len(sequence_a) != len(sequence_b):
            return False
        return all(
            _amc_scalar_equal(left, right)
            for left, right in zip(sequence_a, sequence_b, strict=True)
        )
    return _amc_scalar_equal(surface_a, surface_b)


@lru_cache(maxsize=4096)
def _parse_amc_expression(surface: str):
    """Parse one scalar AMC expression; non-scalars fall back to surface equality."""
    if not surface or any(token in surface for token in (",", r"\pm")):
        return None
    powers = [int(value) for value in re.findall(r"\^\{?(\d+)\}?", surface)]
    factorials = [int(value) for value in re.findall(r"(\d+)\s*!", surface)]
    if any(value > 100 for value in powers + factorials):
        return None
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        return sympy.simplify(parse_latex(surface))
    except Exception:
        return None


def _normalize_amc_answer(text: str) -> str:
    surface = _amc_surface(text)
    expression = _parse_amc_expression(surface)
    return str(expression) if expression is not None else surface


def require_amc_symbolic_matcher() -> None:
    """Fail before paid calls when the frozen AMC scorer is unavailable."""
    try:
        import antlr4.error.ErrorListener  # noqa: F401
        from sympy.parsing.latex import parse_latex

        parsed = parse_latex(r"\frac{1}{2}")
    except Exception as exc:
        raise RuntimeError(
            "E2H-AMC requires SymPy and antlr4-python3-runtime 4.11.x for the "
            "frozen symbolic answer matcher; reinstall the project dependencies "
            "before running the paid campaign"
        ) from exc
    if str(parsed) != "1/2":
        raise RuntimeError("E2H-AMC symbolic answer matcher failed its preflight")


def _parse_number(s: str) -> str | None:
    """Try to parse s as a canonical numeric string, else return None.

    Handles: integers, decimals, scientific notation, LaTeX fractions,
    plain fractions.  Fractions are converted to decimals so that
    gold=\\frac{1}{2} and prediction=0.5 compare as equal.
    """
    s = s.strip()
    fraction_surface = s.replace(",", "")
    # LaTeX fraction \frac{a}{b} or plain a/b
    m = _LATEX_FRAC.fullmatch(fraction_surface)
    if m and int(m.group(2)) != 0:
        return f"{int(m.group(1)) / int(m.group(2)):.12g}"
    m = _PLAIN_FRAC.fullmatch(fraction_surface)
    if m and int(m.group(2)) != 0:
        return f"{int(m.group(1)) / int(m.group(2)):.12g}"
    # Plain number (decimal, scientific notation)
    nums = _NUM.findall(s)
    if nums:
        try:
            val = float(nums[-1].replace(",", ""))
            return str(int(val)) if float(val).is_integer() else f"{val:.12g}"
        except (ValueError, OverflowError):
            pass
    return None


def _strip_outer_math_delimiters(text: str) -> str:
    """Remove one complete pair of display or inline LaTeX delimiters."""
    value = text.strip()
    pairs = (("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"))
    for opening, closing in pairs:
        if (
            value.startswith(opening)
            and value.endswith(closing)
            and len(value) >= len(opening) + len(closing)
        ):
            return value[len(opening) : -len(closing)].strip()
    return value


def normalize_answer(text: str, dataset: str = "") -> str:
    """Dataset-aware canonical form for answer comparison.

    Dispatch order:
      StrategyQA    → yes/no keyword detection
      ARC-Challenge → single letter A-D
      MATH          → \\boxed{} (nested braces), fraction→decimal, last number
      GSM8K/generic → answer-marker extraction, fraction→decimal, last number

    v0.1.3 additions (incorporating GPT v0.2.0 improvements):
      - \\frac{a}{b} and a/b converted to decimal for cross-format matching
      - Scientific notation support (e.g. 2.5e-3)
      - Explicit answer-marker shortcuts (####, "Answer:", "final answer")
    """
    t = text.strip().lower()

    if dataset.startswith("supergpqa"):
        marked = re.findall(
            r"(?:final_choice|final choice|final answer|answer)\s*:\s*[\[(]?([a-j])\b",
            t,
        )
        if marked:
            return marked[-1]
        standalone = re.fullmatch(r"\s*[\[(]?([a-j])[\])?.!\s]*", t)
        return standalone.group(1) if standalone else t

    # E2H-AMC includes exact fractions, radicals, powers and pi expressions.
    # Generic "last number" extraction would turn, for example,
    # ``2^{2011}-2012`` into the incorrect canonical answer ``2012``.
    if dataset == "e2h_amc":
        return _normalize_amc_answer(text)

    # ── StrategyQA: binary yes/no ─────────────────────────────────────────
    if dataset == "strategyqa":
        if re.search(r"\byes\b", t):
            return "yes"
        if re.search(r"\bno\b", t):
            return "no"
        return t

    # ── ARC-Challenge: multiple-choice letter ─────────────────────────────
    if dataset == "arc_challenge":
        m = _MCHOICE.search(t)
        if m:
            return m.group(1)
        return t

    # ── MATH: \\boxed{} first (handles nested braces), then fallback ──────
    if dataset == "math":
        text = _strip_outer_math_delimiters(text)
        t = text.lower()
        m = _BOXED.search(text)  # original text for LaTeX case sensitivity
        if m:
            expr = m.group(1).strip()
            num = _parse_number(expr)
            if num is not None:
                return num
            # Non-numeric boxed expression (polynomial, set, …): preserve verbatim.
            # Both gold and prediction normalize the same way, so equality holds.
            return re.sub(r"\s+", " ", expr).strip()
        # No \\boxed{}: fall through to generic extraction below.

    # ── GSM8K / generic: marker → fraction → last number ─────────────────
    # Check explicit answer markers first to avoid scanning intermediate steps.
    markers = _MARKERS.findall(t)
    if markers:
        t = markers[-1].strip()

    num = _parse_number(t)
    if num is not None:
        return num

    return _NORM.sub(" ", t).strip()


def answers_equal(a: str, b: str, dataset: str = "") -> bool:
    if dataset == "e2h_amc":
        return symbolic_answers_equal(a, b)
    return normalize_answer(a, dataset) == normalize_answer(b, dataset)


# ── metric classes ────────────────────────────────────────────────────────────


class CorrectnessQuality(WindowMeanMetric):
    """CQ: is the final answer right?  Bernoulli per query.

    Two-level detection (v0.1.3):
      raw_detection=True  → secondary DDM on per-query 0/1 stream (Bernoulli).
      windowed_distribution=BETA → primary CUSUM on the W-step window mean
        (which is Beta-distributed, not Bernoulli — CLT applies at W=20).

    v0.1.2 bug: both levels mapped to DDM because THEORETICAL_ASSIGNMENT
    [Bernoulli]→DDM was used for the windowed level too.  This fix makes
    the windowed detector CUSUM and keeps the raw detector DDM.
    """

    spec = MetricSpec(
        "CQ",
        "Final-answer correctness",
        Distribution.BERNOULLI,
        raw_detection=True,
        windowed_distribution=Distribution.BETA,  # v0.1.3 fix
    )

    def per_query_value(self, obs: QueryObservation) -> float:
        return float(
            answers_equal(obs.primary.answer, obs.query.gold_answer, obs.query.dataset)
        )


class ConsistencyScore(WindowMeanMetric):
    """CS: do K repeated calls agree with each other?  Pairwise agreement.

    K is now a constructor argument so MonitorConfig.k_repeats propagates
    correctly.  Previously K_RUNS=3 was a module constant that shadowed config.
    """

    spec = MetricSpec("CS", "Self-consistency across repeats", Distribution.BETA)

    def __init__(
        self,
        window_size: int,
        k_runs: int = _DEFAULT_K,
        semantic_answer_policy: str | None = None,
    ) -> None:
        super().__init__(window_size)
        self.k_runs = k_runs
        self.semantic_answer_policy = semantic_answer_policy

    def per_query_value(self, obs: QueryObservation) -> float:
        runs = list(obs.runs[: self.k_runs])
        if len(runs) < 2:
            return 1.0
        ds = obs.query.dataset
        pairs = list(combinations(runs, 2))
        if self.semantic_answer_policy:
            from reasoning_monitor.metrics.semantic_answers import (
                semantic_answers_equal,
            )

            agree = sum(
                semantic_answers_equal(
                    a.answer, b.answer, ds, self.semantic_answer_policy
                )
                for a, b in pairs
            )
        else:
            agree = sum(answers_equal(a.answer, b.answer, ds) for a, b in pairs)
        return agree / len(pairs)


class RobustnessScore(WindowMeanMetric):
    """RS: does the correct answer survive meaning-preserving rewrites?

    C-subset: undefined (NaN) when the primary call is wrong.  WindowMeanMetric
    skips NaN values, so incorrect queries do not enter the window mean.

    L is now a constructor argument (same K/L bug fix as ConsistencyScore).
    """

    spec = MetricSpec("RS", "Robustness to paraphrase (C-subset)", Distribution.BETA)

    def __init__(
        self,
        window_size: int,
        l_paraphrases: int = _DEFAULT_L,
        semantic_answer_policy: str | None = None,
    ) -> None:
        super().__init__(window_size)
        self.l_paraphrases = l_paraphrases
        self.semantic_answer_policy = semantic_answer_policy

    def per_query_value(self, obs: QueryObservation) -> float:
        gold = obs.query.gold_answer
        ds = obs.query.dataset
        if not answers_equal(obs.primary.answer, gold, ds):
            return float("nan")  # C-subset exclusion
        pruns = list(obs.paraphrase_runs[: self.l_paraphrases])
        if not pruns:
            return 1.0
        if self.semantic_answer_policy:
            from reasoning_monitor.metrics.semantic_answers import (
                semantic_rs_answer_correct,
            )

            return sum(
                semantic_rs_answer_correct(
                    p.answer,
                    gold,
                    obs.primary.answer,
                    ds,
                    self.semantic_answer_policy,
                )
                for p in pruns
            ) / len(pruns)
        return sum(answers_equal(p.answer, gold, ds) for p in pruns) / len(pruns)


class LogicalCoherenceScore(WindowMeanMetric):
    """LS: is the reasoning chain internally coherent?

    Consecutive-step NLI contradiction detection via DeBERTa-v3-small.
    Falls back to LexicalCoherence if torch/transformers not installed.
    In publication_mode the lexical fallback raises an error (see MonitorConfig).
    """

    spec = MetricSpec("LS", "Logical coherence of the trace", Distribution.BETA)

    def __init__(self, window_size: int, scorer=None) -> None:
        super().__init__(window_size)
        if scorer is None:
            from reasoning_monitor.metrics.coherence import LexicalCoherence

            scorer = LexicalCoherence()
        self.scorer = scorer

    def per_query_value(self, obs: QueryObservation) -> float:
        steps = list(obs.primary.reasoning_steps)
        if len(steps) < 2:
            return 1.0
        incoherence = self.scorer.score_pairs(steps[:-1], steps[1:])
        return max(0.0, 1.0 - sum(incoherence) / len(incoherence))


class EfficiencyScore(WindowMeanMetric):
    """ES: harmonic mean of correctness and token efficiency.

    ES_i = 2·CQ_i·(1−T_i) / (CQ_i + (1−T_i)),   T_i = n_tokens / t_max

    When CQ_i = 0 (wrong answer), ES_i = 0 regardless of token cost.
    Distribution is Beta (bounded [0,1]) → CUSUM is the assigned detector.
    """

    spec = MetricSpec("ES", "Token efficiency (harmonic mean)", Distribution.BETA)

    def __init__(self, window_size: int, reference_tokens: float = 256.0) -> None:
        super().__init__(window_size)
        self.reference_tokens = reference_tokens

    def per_query_value(self, obs: QueryObservation) -> float:
        ds = obs.query.dataset
        cq_i = float(answers_equal(obs.primary.answer, obs.query.gold_answer, ds))
        t_i = min(obs.primary.n_tokens / max(self.reference_tokens, 1.0), 1.0)
        eff = 1.0 - t_i
        denom = cq_i + eff
        if denom < 1e-9:
            return 0.0
        return (2.0 * cq_i * eff) / denom


class StabilityScore(WindowMeanMetric):
    """SS: pairwise trace similarity across K reasoning runs (matches rqeval / arXiv:2605.24661).

    Formula (per query):
        SS_i = (2 / K(K-1)) * Σ_{k<l} BERTScore(r_i^(k), r_i^(l))

    Measures whether the *reasoning process* is stable across K independent
    calls — not just whether the final answer matches (that is CS).

    Distribution: BETA (bounded [0,1] similarity) → CUSUM for windowed detection.

    v0.1.3: changed from WindowDispersionMetric (temporal variance of window
    means) to WindowMeanMetric (per-query pairwise trace similarity), aligning
    with the published rqeval definition.  See DEVIATIONS §6.

    Falls back to Jaccard token similarity when bert-score is not installed.
    """

    spec = MetricSpec(
        "SS", "Trace stability across runs (BERTScore / Jaccard)", Distribution.BETA
    )

    def __init__(self, window_size: int, scorer=None, k_runs: int = _DEFAULT_K) -> None:
        super().__init__(window_size)
        if scorer is None:
            from reasoning_monitor.metrics.coherence import build_trace_scorer

            scorer = build_trace_scorer("auto")
        self.scorer = scorer
        self.k_runs = k_runs

    def per_query_value(self, obs: QueryObservation) -> float:
        runs = list(obs.runs[: self.k_runs])
        if len(runs) < 2:
            return 1.0  # single run: trivially stable

        # Concatenate reasoning steps into one trace string per run.
        traces = [" ".join(r.reasoning_steps) if r.reasoning_steps else r.answer for r in runs]

        # All pairwise combinations (k < l)
        pairs_a, pairs_b = [], []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                pairs_a.append(traces[i])
                pairs_b.append(traces[j])

        if not pairs_a:
            return 1.0

        scores = self.scorer.score_pairs(pairs_a, pairs_b)
        return max(0.0, min(1.0, float(sum(scores) / len(scores))))


METRIC_CLASSES = {
    "CQ": CorrectnessQuality,
    "CS": ConsistencyScore,
    "RS": RobustnessScore,
    "LS": LogicalCoherenceScore,
    "ES": EfficiencyScore,
    "SS": StabilityScore,
}


def build_metric(name: str, window_size: int, **kwargs):
    if name not in METRIC_CLASSES:
        raise ValueError(f"unknown metric: {name!r}; expected one of {list(METRIC_CLASSES)}")
    return METRIC_CLASSES[name](window_size, **kwargs)
