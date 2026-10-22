"""Unit tests for v0.1.2.

New in v0.1.2
-------------
* test_normalize_answer_dataset_aware  — dataset-specific extraction
* test_rs_correct_primary / test_rs_nan_for_wrong_primary — C-subset
* test_es_harmonic_mean — CQ=0 → ES=0
* test_cs_k_propagates / test_rs_l_propagates — K/L config bug fix
* test_publication_mode_blocks_lexical — coherence guard
* test_page_hinkley_statistic — correct alarm condition
* test_ss_o1_matches_naive — O(1) variance equals explicit computation
* test_raw_detector_registered — CQ has a secondary DDM
* test_effective_change_point — gradual t* = midpoint
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from reasoning_monitor.cli import apply_quick_preset
from reasoning_monitor.data.scenarios import SCENARIOS, build_stream
from reasoning_monitor.detectors.assignment import THEORETICAL_ASSIGNMENT, assign_detector
from reasoning_monitor.detectors.detectors import ADWIN, CUSUM, DDM, EWMA, PageHinkley
from reasoning_monitor.experiment import (
    ExperimentConfig,
    _make_monitor,
    _validate_scenarios,
    run_experiment,
)
from reasoning_monitor.metrics.dimensions import (
    ConsistencyScore,
    CorrectnessQuality,
    EfficiencyScore,
    RobustnessScore,
    StabilityScore,
    answers_equal,
    build_metric,
    normalize_answer,
)
from reasoning_monitor.models.clients import SimulatedClient, reasoning_steps_from_text
from reasoning_monitor.monitor import MonitorConfig, MonitorResult, ReasoningMonitor
from reasoning_monitor.types import (
    METRIC_ORDER,
    Distribution,
    MetricSnapshot,
    Query,
    QueryObservation,
    Sample,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reasoning_step_parser_preserves_structure_without_cap():
    paragraphs = [f"Step {index}: calculate {index}.5 carefully." for index in range(15)]
    text = "\n\n".join([*paragraphs, "Final answer:\nAnswer: 42"])

    steps = reasoning_steps_from_text(text)

    assert len(steps) == 16
    assert "0.5" in steps[0]
    assert steps[-1] == "Final answer: Answer: 42"


def test_reasoning_step_parser_sentence_fallback_preserves_decimals():
    steps = reasoning_steps_from_text("First compute 0.25. Then double it. Answer: 0.5")

    assert steps == ["First compute 0.25.", "Then double it.", "Answer: 0.5"]

# rqeval parity tests skip gracefully when rqeval is not installed.
try:
    from metrics.accuracy import AccuracyMetric as _check_rqeval
    RQEVAL_AVAILABLE = True
except ImportError:
    RQEVAL_AVAILABLE = False

requires_rqeval = pytest.mark.skipif(
    not RQEVAL_AVAILABLE,
    reason="rqeval not installed (pip install rqeval)"
)


def make_obs(answer: str, gold: str, k: int = 3, l: int = 3,
             tokens: int = 200, steps: list[str] | None = None,
             dataset: str = "gsm8k") -> QueryObservation:
    q = Query(qid="q1", t=1, dataset=dataset, domain="math", prompt="p",
              gold_answer=gold)
    s = Sample(answer=answer, reasoning_steps=steps or ["a", "b"],
               latency_s=1.0, n_tokens=tokens)
    return QueryObservation(query=q, runs=[s] * k, paraphrase_runs=[s] * l)


# ── normalisation ─────────────────────────────────────────────────────────────

def test_numeric_answers_compare_as_numbers():
    assert answers_equal("42", "42.0")
    assert answers_equal("The answer is 42", "42", "gsm8k")
    assert not answers_equal("13", "3")   # substring match would pass this
    assert answers_equal("Answer: 14,000", "14000", "e2h_gsm8k")
    assert not answers_equal("Answer: 4,000", "14000", "e2h_gsm8k")


def test_normalize_strips_formatting():
    # v0.1.3: "Answer: Paris!" → marker extraction strips the prefix → "paris"
    assert normalize_answer("  Answer: Paris!  ") == "paris"
    # Without a marker: generic normalisation applies
    assert normalize_answer("paris") == "paris"


def test_normalize_answer_dataset_aware():
    """v0.1.2+: dataset-specific extraction."""
    assert normalize_answer("Yes, that is correct.", "strategyqa") == "yes"
    assert normalize_answer("No, it is not.", "strategyqa") == "no"
    assert normalize_answer("The answer is B", "arc_challenge") == "b"
    assert normalize_answer(r"Step 1: compute 3+4=7. \boxed{7}", "math") == "7"
    assert normalize_answer(r"\boxed{\frac{1}{2}}", "math") == \
           normalize_answer(r"The answer is \boxed{\frac{1}{2}}.", "math")
    assert normalize_answer("First I have 3 apples then buy 4 more = 7", "gsm8k") == "7"


def test_fraction_normalization():
    """v0.1.3: \frac{a}{b} and a/b → decimal for cross-format matching."""
    assert normalize_answer(r"\frac{1}{2}", "math") == normalize_answer("0.5", "math")
    assert normalize_answer(r"\boxed{\frac{3}{4}}", "math") == normalize_answer("0.75", "math")
    assert normalize_answer("1/4", "gsm8k") == normalize_answer("0.25", "gsm8k")
    assert answers_equal(r"\boxed{\frac{1}{2}}", "0.5", "math")


def test_math_answer_normalization_strips_complete_latex_delimiters():
    gold = r"\frac{16}{65}"
    assert answers_equal(r"$\frac{16}{65}$", gold, "math")
    assert answers_equal(r"$$\frac{16}{65}$$", gold, "math")
    assert answers_equal(r"\(\frac{16}{65}\)", gold, "math")
    assert answers_equal(r"\[\frac{16}{65}\]", gold, "math")


# ── metrics ───────────────────────────────────────────────────────────────────

def test_cq_is_one_for_correct_answer():
    m = CorrectnessQuality(window_size=2)
    assert m.per_query_value(make_obs("42", "42")) == 1.0
    assert m.per_query_value(make_obs("41", "42")) == 0.0


def test_window_mean_matches_naive_recomputation():
    m = CorrectnessQuality(window_size=5)
    values, results = [], []
    for i in range(30):
        correct = i % 3 != 0
        obs = make_obs("42" if correct else "41", "42")
        values.append(1.0 if correct else 0.0)
        results.append(m.update(obs))

    for i, got in enumerate(results):
        if i < 4:
            assert got is None
        else:
            expected = sum(values[i - 4: i + 1]) / 5
            assert got == pytest.approx(expected, abs=1e-12)


def test_cs_pairwise_agreement():
    m = ConsistencyScore(window_size=2)
    q = Query(qid="q", t=1, dataset="t", domain="math", prompt="p", gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    obs = QueryObservation(query=q, runs=[mk("A"), mk("A"), mk("B")],
                           paraphrase_runs=[])
    assert m.per_query_value(obs) == pytest.approx(1 / 3)


def test_cs_k_propagates():
    """v0.1.2: k_runs constructor arg must be respected (K/L config bug fix).

    runs = [A, A, B, A]
    K=3 → uses [A, A, B]: pairs (A,A)✓ (A,B)✗ (A,B)✗ → 1/3
    K=4 → uses [A, A, B, A]: pairs AA✓ AB✗ AA✓ AB✗ AA✓ BA✗ → 3/6 = 1/2
    """
    q = Query(qid="q", t=1, dataset="t", domain="math", prompt="p", gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    obs = QueryObservation(query=q, runs=[mk("A"), mk("A"), mk("B"), mk("A")],
                           paraphrase_runs=[])
    m3 = ConsistencyScore(window_size=2, k_runs=3)
    m4 = ConsistencyScore(window_size=2, k_runs=4)
    assert m3.per_query_value(obs) == pytest.approx(1 / 3)
    assert m4.per_query_value(obs) == pytest.approx(3 / 6)


def test_rs_correct_primary():
    """v0.1.2: RS compares paraphrases against gold (not primary call)."""
    m = RobustnessScore(window_size=2)
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p",
              gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    obs = QueryObservation(query=q, runs=[mk("42")],
                           paraphrase_runs=[mk("42"), mk("42"), mk("7")])
    assert m.per_query_value(obs) == pytest.approx(2 / 3)


def test_rs_nan_for_wrong_primary():
    """v0.1.2: RS is NaN when primary is wrong (C-subset exclusion)."""
    m = RobustnessScore(window_size=2)
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p",
              gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    obs = QueryObservation(query=q, runs=[mk("99")],
                           paraphrase_runs=[mk("99"), mk("99"), mk("99")])
    assert math.isnan(m.per_query_value(obs))


def test_rs_l_propagates():
    """v0.1.2: l_paraphrases constructor arg must be respected (K/L bug fix)."""
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p",
              gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    # 4 paraphrases: 42 42 42 7.  L=3 → 2/3 agree. L=4 → 3/4 agree.
    obs = QueryObservation(query=q, runs=[mk("42")],
                           paraphrase_runs=[mk("42"), mk("42"), mk("7"), mk("42")])
    m3 = RobustnessScore(window_size=2, l_paraphrases=3)
    m4 = RobustnessScore(window_size=2, l_paraphrases=4)
    assert m3.per_query_value(obs) == pytest.approx(2 / 3)
    assert m4.per_query_value(obs) == pytest.approx(3 / 4)


def test_es_harmonic_mean():
    """v0.1.2: ES = 2·CQ·(1-T)/(CQ+(1-T)); CQ=0 → ES=0."""
    m = EfficiencyScore(window_size=2, reference_tokens=512)
    correct_cheap  = m.per_query_value(make_obs("42", "42", tokens=100))
    correct_costly = m.per_query_value(make_obs("42", "42", tokens=400))
    wrong          = m.per_query_value(make_obs("41", "42", tokens=50))
    assert correct_cheap > correct_costly > 0.0
    assert wrong == pytest.approx(0.0)
    assert 0.0 <= correct_costly <= correct_cheap <= 1.0


def test_ss_pairwise_trace_similarity():
    """v0.1.3-final: SS = pairwise trace similarity (rqeval/BERTScore definition).
    Identical traces → SS=1.0; single run → 1.0 by convention.
    """
    from reasoning_monitor.metrics.coherence import JaccardTraceSim
    m = StabilityScore(window_size=2, scorer=JaccardTraceSim())

    # K=3 identical reasoning traces → Jaccard = 1.0 → SS = 1.0
    obs_same = make_obs("42", "42", steps=["step A", "step B", "step C"])
    assert m.per_query_value(obs_same) == pytest.approx(1.0)

    # Completely different traces → Jaccard similarity is low
    from reasoning_monitor.types import Query, QueryObservation, Sample
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p", gold_answer="42")
    mk = lambda steps: Sample(answer="42", reasoning_steps=steps, latency_s=1.0, n_tokens=10)
    obs_diff = QueryObservation(
        query=q,
        runs=[mk(["alpha beta gamma"]), mk(["xyz uvw pqr"]), mk(["one two three"])],
        paraphrase_runs=[],
    )
    ss_diff = m.per_query_value(obs_diff)
    assert 0.0 <= ss_diff < 1.0, f"Different traces must give SS < 1.0, got {ss_diff}"


def test_ss_window_accumulates():
    """v0.1.3-final: SS window accumulates pairwise similarity values."""
    from reasoning_monitor.metrics.coherence import JaccardTraceSim
    W = 3
    m = StabilityScore(window_size=W, scorer=JaccardTraceSim())
    # Feed W+1 observations; window should return a value after W
    results = []
    for i in range(W + 2):
        obs = make_obs("42", "42", steps=["step a", "step b"])
        v = m.update(obs)
        results.append(v)
    assert results[W-1] is not None, "Window should be full at index W-1"
    assert all(0.0 <= r <= 1.0 for r in results if r is not None)


def test_all_metrics_stay_in_unit_interval():
    """RS returns NaN for C-subset exclusions; skip NaN before asserting.
    SS uses explicit JaccardTraceSim to avoid HuggingFace network calls in CI."""
    from reasoning_monitor.metrics.coherence import JaccardTraceSim
    for name in ("CQ", "CS", "RS", "LS", "ES", "SS"):
        kwargs = {"scorer": JaccardTraceSim()} if name == "SS" else {}
        m = build_metric(name, window_size=3, **kwargs)
        last = None
        for i in range(20):
            last = m.update(make_obs("42" if i % 4 else "41", "42"))
        if last is not None and not math.isnan(last):
            assert 0.0 <= last <= 1.0, f"{name} left [0,1]: {last}"


def test_window_size_must_be_at_least_two():
    with pytest.raises(ValueError):
        CorrectnessQuality(window_size=1)


# ── detectors ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [CUSUM, PageHinkley, EWMA, DDM, ADWIN])
def test_detector_is_quiet_on_a_stationary_stream(cls):
    det = cls()
    rng = __import__("random").Random(7)
    alarms = [det.update(0.8 + rng.gauss(0, 0.02), t, "CQ") for t in range(200)]
    fired = [a for a in alarms if a is not None]
    assert len(fired) <= 2, f"{cls.__name__} raised {len(fired)} false alarms"


@pytest.mark.parametrize("cls", [CUSUM, PageHinkley, EWMA, DDM])
def test_detector_finds_a_large_sudden_drop(cls):
    det = cls()
    rng = __import__("random").Random(11)
    fired_at = None
    for t in range(300):
        value = 0.9 if t < 150 else 0.3
        alarm = det.update(value + rng.gauss(0, 0.02), t, "CQ")
        if alarm and t >= 150 and fired_at is None:
            fired_at = t
    assert fired_at is not None, f"{cls.__name__} missed a 0.6 drop"
    assert fired_at - 150 < 60, f"{cls.__name__} took {fired_at - 150} steps"


def test_page_hinkley_statistic():
    """v0.1.2: alarm condition is (max_cum - cum) > threshold (was: cum > threshold)."""
    det = PageHinkley(threshold=0.1, delta=0.001)
    rng = __import__("random").Random(42)
    # Stationary at 0.9: no alarm expected
    for t in range(50):
        a = det.update(0.9 + rng.gauss(0, 0.01), t, "CQ")
        assert a is None, f"False alarm at t={t}"
    # Sharp drop to 0.3: alarm expected
    fired = False
    for t in range(50, 150):
        a = det.update(0.3 + rng.gauss(0, 0.01), t, "CQ")
        if a is not None:
            fired = True
            break
    assert fired, "PageHinkley missed a 0.6 downward shift"


def test_assignment_covers_every_distribution():
    for dist in Distribution:
        assert dist in THEORETICAL_ASSIGNMENT


def test_assignment_follows_the_declared_distribution():
    """v0.1.3: CQ windowed → CUSUM (Beta via windowed_distribution); raw → DDM."""
    cq = build_metric("CQ", 5)
    es = build_metric("ES", 5)
    # Primary (windowed) detector: CQ uses windowed_distribution=BETA → CUSUM
    assert assign_detector(cq).name == "cusum"
    assert assign_detector(es).name == "cusum"
    # Raw detector: CQ per-query is Bernoulli → DDM
    from reasoning_monitor.detectors.assignment import assign_raw_detector
    assert assign_raw_detector(cq).name == "ddm"


def test_three_detector_strategies():
    """v0.1.3: three named strategies for H2 ablation."""
    from reasoning_monitor.detectors.assignment import assign_detector
    cq = build_metric("CQ", 5)
    # distribution_based: CQ windowed → CUSUM
    assert assign_detector(cq, strategy="distribution_based").name == "cusum"
    # proposal_mapping: CQ → CUSUM (as stated in proposal)
    assert assign_detector(cq, strategy="proposal_mapping").name == "cusum"
    # adwin_all: everything → ADWIN
    assert assign_detector(cq, strategy="adwin_all").name == "adwin"
    ls = build_metric("LS", 5)
    # proposal_mapping: LS → page_hinkley
    assert assign_detector(ls, strategy="proposal_mapping").name == "page_hinkley"
    # adwin_all: LS → adwin
    assert assign_detector(ls, strategy="adwin_all").name == "adwin"


def test_override_wins_over_theory():
    cq = build_metric("CQ", 5)
    det = assign_detector(cq, {"CQ": {"name": "ewma", "lam": 0.3}})
    assert det.name == "ewma"


@requires_rqeval
def test_raw_detector_registered():
    """v0.1.3: CQ raw → DDM; CQ primary (windowed) → CUSUM.
    Requires rqeval because ReasoningMonitor instantiates rqeval adapters."""
    from reasoning_monitor.models.clients import SimulatedClient
    from reasoning_monitor.monitor import MonitorConfig, ReasoningMonitor

    client = SimulatedClient(name="test", seed=0)
    mon = ReasoningMonitor(client, MonitorConfig())
    # Raw (per-query Bernoulli) detector
    assert "CQ" in mon.raw_detectors
    assert mon.raw_detectors["CQ"].name == "ddm"
    # Primary (windowed Beta) detector — now CUSUM, not DDM (v0.1.2→v0.1.3 fix)
    assert mon.detectors["CQ"].name == "cusum"
    assert "CS" not in mon.raw_detectors
    assert "SS" not in mon.raw_detectors


def test_publication_mode_blocks_lexical():
    """publication_mode=True requires coherence_scorer='nli' AND stability_scorer='bertscore'."""
    from reasoning_monitor.monitor import MonitorConfig
    # coherence_scorer="lexical" rejected
    with pytest.raises(ValueError, match="publication_mode"):
        MonitorConfig(publication_mode=True, coherence_scorer="lexical", stability_scorer="bertscore")
    # coherence_scorer="auto" also rejected
    with pytest.raises(ValueError, match="publication_mode"):
        MonitorConfig(publication_mode=True, coherence_scorer="auto", stability_scorer="bertscore")
    # stability_scorer="auto" rejected (bug fix: auto may fall back to Jaccard)
    with pytest.raises(ValueError, match="bertscore"):
        MonitorConfig(publication_mode=True, coherence_scorer="nli", stability_scorer="auto")
    # stability_scorer="jaccard" rejected
    with pytest.raises(ValueError, match="bertscore"):
        MonitorConfig(publication_mode=True, coherence_scorer="nli", stability_scorer="jaccard")
    # Only nli + bertscore is accepted
    cfg = MonitorConfig(publication_mode=True, coherence_scorer="nli", stability_scorer="bertscore")
    assert cfg.coherence_scorer == "nli"
    assert cfg.stability_scorer == "bertscore"


# ── scenarios ─────────────────────────────────────────────────────────────────

def _make_ds():
    from reasoning_monitor.data.loaders import load_dataset_queries
    return {
        "gsm8k": load_dataset_queries("gsm8k", 150, allow_synthetic=True),
    }


def test_st1_stationary_is_mixed():
    """ST1: mixed E/M/H, all difficulties present, no change point."""
    from reasoning_monitor.data.scenarios import (
        SCENARIOS,
        build_stream,
        effective_change_point,
    )
    spec = SCENARIOS["ST1"]
    stream = build_stream(spec, _make_ds(), seed=1)
    assert len(stream) == spec.n_queries
    assert effective_change_point(spec) == -1  # stationary


def test_st2_composition_shift_easy_to_hard():
    """ST2: Easy→Medium→Hard staged regimes with two change points."""
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream
    spec = SCENARIOS["ST2"]
    stream = build_stream(spec, _make_ds(), seed=1)
    assert spec.change_points == [50, 100]
    # First regime is easy, last is hard
    assert stream[0].difficulty == "easy"
    assert stream[-1].difficulty == "hard"
    # Middle regime is medium
    assert stream[74].difficulty == "medium"


def test_resized_st2_preserves_all_regimes_and_change_points():
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream, resize_regimes

    spec = resize_regimes(SCENARIOS["ST2"], 12)
    stream = build_stream(spec, _make_ds(), seed=1)

    assert spec.n_queries == 36
    assert spec.change_points == [12, 24]
    assert [stream[i].difficulty for i in (0, 12, 24)] == [
        "easy", "medium", "hard"
    ]


def test_st3_stationary_easy_only():
    """ST3: Easy-only stationary control — single difficulty throughout."""
    from reasoning_monitor.data.scenarios import (
        SCENARIOS,
        build_stream,
        effective_change_point,
    )
    spec = SCENARIOS["ST3"]
    stream = build_stream(spec, _make_ds(), seed=1)
    assert set(q.difficulty for q in stream) == {"easy"}
    assert effective_change_point(spec) == -1


def test_st6_reverse_hard_to_easy():
    """ST6: Hard→Medium→Easy reverse/recovery test."""
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream
    spec = SCENARIOS["ST6"]
    stream = build_stream(spec, _make_ds(), seed=1)
    assert stream[0].difficulty == "hard"
    assert stream[-1].difficulty == "easy"
    assert spec.change_points == [50, 100]


def test_within_regime_is_permuted_not_sorted():
    """Within a regime, order must be random permutation, not difficulty-sorted.
    Two different seeds must give different within-regime orderings."""
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream
    ds = _make_ds()
    a = build_stream(SCENARIOS["ST3"], ds, seed=1)
    b = build_stream(SCENARIOS["ST3"], ds, seed=2)
    # Same difficulty pool, but different order under different seeds
    assert [q.qid for q in a] != [q.qid for q in b]


def test_multiseed_streams_are_independent():
    """build_multiseed_streams produces distinct realisations per seed."""
    from reasoning_monitor.data.scenarios import SCENARIOS, build_multiseed_streams
    ds = _make_ds()
    streams = build_multiseed_streams(SCENARIOS["ST2"], ds, seeds=[1, 2, 3])
    assert len(streams) == 3
    # First query differs across seeds (independent permutations)
    first_qids = {s[0].qid for s in streams}
    assert len(first_qids) > 1


def test_stream_is_deterministic_under_a_fixed_seed():
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream
    ds = _make_ds()
    a = build_stream(SCENARIOS["ST2"], ds, seed=99)
    b = build_stream(SCENARIOS["ST2"], ds, seed=99)
    assert [q.qid for q in a] == [q.qid for q in b]


def test_effective_change_point_staged_and_stationary():
    """Staged streams return first regime boundary; stationary return -1."""
    from reasoning_monitor.data.scenarios import SCENARIOS, effective_change_point
    assert effective_change_point(SCENARIOS["ST2"]) == 50   # first boundary
    assert effective_change_point(SCENARIOS["ST6"]) == 50
    assert effective_change_point(SCENARIOS["ST1"]) == -1   # stationary
    assert effective_change_point(SCENARIOS["ST3"]) == -1


# ── evaluation protocol v0.1.3-final ─────────────────────────────────────────

def test_three_alarm_categories_early():
    """P0: alarms in [ts, t*) must be early_detection, not discarded."""
    from reasoning_monitor.evaluation.metrics_eval import AlarmCategory, evaluate_detection
    from reasoning_monitor.types import Alarm
    alarm = Alarm(t=110, metric="CQ", detector="cusum", statistic=0.3, level="drift")
    d = evaluate_detection([alarm], effective_cp=120, metric="CQ", detector="cusum",
                           n_pre_observations=100, transition_start=100)
    assert d.detected
    assert d.early_detected
    assert d.first_alarm_category == AlarmCategory.EARLY_DETECTION.value
    assert d.delay == pytest.approx(-10.0)
    assert d.n_false_alarms == 0


def test_three_alarm_categories_false():
    """Alarm before ts is a false alarm."""
    from reasoning_monitor.evaluation.metrics_eval import evaluate_detection
    from reasoning_monitor.types import Alarm
    alarm = Alarm(t=90, metric="CQ", detector="cusum", statistic=0.3, level="drift")
    d = evaluate_detection([alarm], effective_cp=120, metric="CQ", detector="cusum",
                           n_pre_observations=100, transition_start=100)
    assert not d.detected
    assert d.n_false_alarms == 1
    assert d.delay is None


def test_sudden_no_early_zone():
    """Sudden scenario ts==t*: no early_detection zone."""
    from reasoning_monitor.evaluation.metrics_eval import AlarmCategory, evaluate_detection
    from reasoning_monitor.types import Alarm
    alarm = Alarm(t=105, metric="CQ", detector="cusum", statistic=0.3, level="drift")
    d = evaluate_detection([alarm], effective_cp=100, metric="CQ", detector="cusum",
                           n_pre_observations=100, transition_start=100)
    assert d.detected and not d.early_detected
    assert d.first_alarm_category == AlarmCategory.POST_THRESHOLD.value
    assert d.delay == pytest.approx(5.0)


def test_f1_event_based():
    """Scenario-level F1: TP=2, FP=1, FN=1 → F1=2/3."""
    from reasoning_monitor.evaluation.metrics_eval import compute_f1
    rows = [
        {"strategy": "dist", "metric": "CQ", "detector": "cusum", "level": "drift",
         "detected": True,  "n_false_alarms": 0, "effective_t_star": 100, "scenario": "S1"},
        {"strategy": "dist", "metric": "CQ", "detector": "cusum", "level": "drift",
         "detected": True,  "n_false_alarms": 0, "effective_t_star": 100, "scenario": "S2"},
        {"strategy": "dist", "metric": "CQ", "detector": "cusum", "level": "drift",
         "detected": False, "n_false_alarms": 0, "effective_t_star": 80,  "scenario": "S3"},
        {"strategy": "dist", "metric": "CQ", "detector": "cusum", "level": "drift",
         "detected": False, "n_false_alarms": 2, "effective_t_star": -1,  "scenario": "S0"},
    ]
    results = compute_f1(rows, strategy="dist", level="drift")
    assert len(results) == 1
    r = results[0]
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 1
    assert abs(r["f1"] - 2/3) < 0.001


# ── rqeval parity tests ───────────────────────────────────────────────────────

def _make_obs_with_runs(answer, gold, k=3, steps=None, paraphrases=None, dataset="gsm8k"):
    """Helper for parity tests with explicit runs."""
    from reasoning_monitor.types import Query, QueryObservation, Sample
    q = Query(qid="q", t=1, dataset=dataset, domain="math", prompt="p", gold_answer=gold)
    s = lambda a, st=None: Sample(answer=a, reasoning_steps=st or steps or ["s1","s2"],
                                  latency_s=1.0, n_tokens=100)
    para = [s(p) for p in (paraphrases or [])]
    return QueryObservation(query=q, runs=[s(answer)] * k, paraphrase_runs=para)


@requires_rqeval
def test_cq_matches_rqeval():
    """CQAdapter must equal AccuracyMetric.compute_per_instance."""
    from metrics.accuracy import AccuracyMetric

    from reasoning_monitor.metrics.rqeval_adapters import CQAdapter
    m = CQAdapter(window_size=2)
    for ans, gold in [("42", "42"), ("41", "42"), ("yes", "yes"), ("B", "B")]:
        obs = _make_obs_with_runs(ans, gold, dataset="gsm8k")
        rqeval_score = AccuracyMetric().compute_per_instance([ans], [gold])[0]
        assert m.per_query_value(obs) == pytest.approx(rqeval_score)


@requires_rqeval
def test_cs_matches_rqeval():
    """CSAdapter must equal ConsistencyMetric.compute_instance."""
    from metrics.consistency import ConsistencyMetric

    from reasoning_monitor.metrics.rqeval_adapters import CSAdapter
    K = 3
    m = CSAdapter(window_size=2, k_runs=K)
    for answers in [["A","A","A"], ["A","A","B"], ["A","B","C"]]:
        from reasoning_monitor.types import Query, QueryObservation, Sample
        q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p", gold_answer="A")
        mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
        obs2 = QueryObservation(query=q, runs=[mk(a) for a in answers], paraphrase_runs=[])
        rqeval_score = ConsistencyMetric({"consistency_runs": K}).compute_instance(answers, "A")
        assert m.per_query_value(obs2) == pytest.approx(float(rqeval_score))


@requires_rqeval
def test_rs_matches_rqeval_correct_original():
    """RSAdapter must equal RobustnessMetric for correct original answers."""
    from metrics.robustness import RobustnessMetric

    from reasoning_monitor.metrics.rqeval_adapters import RSAdapter
    from reasoning_monitor.types import Query, QueryObservation, Sample
    L = 3
    m = RSAdapter(window_size=2, l_paraphrases=L)
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p", gold_answer="42")
    mk = lambda a: Sample(answer=a, reasoning_steps=["s"], latency_s=1.0, n_tokens=10)
    paras = ["42", "42", "7"]
    obs = QueryObservation(query=q, runs=[mk("42")], paraphrase_runs=[mk(p) for p in paras])
    rqeval_score = RobustnessMetric({"robustness_perturbations": L}).compute_per_instance(
        ["42"], [paras], ["42"]
    )[0]
    assert m.per_query_value(obs) == pytest.approx(rqeval_score)


@requires_rqeval
def test_rs_returns_nan_for_wrong_original():
    """RSAdapter returns NaN for wrong originals (C-subset, NaN-skips in window)."""
    from reasoning_monitor.metrics.rqeval_adapters import RSAdapter
    m = RSAdapter(window_size=2)
    obs = _make_obs_with_runs("41", "42", paraphrases=["42","42","42"])
    assert math.isnan(m.per_query_value(obs))


@requires_rqeval
def test_ls_matches_rqeval():
    """LSAdapter must equal LogicalConsistencyMetric.compute_trace."""
    from metrics.logical_consistency import LogicalConsistencyMetric

    from reasoning_monitor.metrics.rqeval_adapters import LSAdapter
    m = LSAdapter(window_size=2)
    steps = ["step one", "step two"]
    obs = _make_obs_with_runs("42", "42", steps=steps)
    rqeval_score = LogicalConsistencyMetric().compute_trace(steps)
    assert m.per_query_value(obs) == pytest.approx(float(rqeval_score))


@requires_rqeval
def test_ss_matches_rqeval():
    """SSAdapter must equal ExplainabilityMetric.compute_instance."""
    from metrics.explainability import ExplainabilityMetric

    from reasoning_monitor.metrics.rqeval_adapters import SSAdapter
    from reasoning_monitor.types import Query, QueryObservation, Sample
    K = 3
    m = SSAdapter(window_size=2, k_runs=K)
    q = Query(qid="q", t=1, dataset="gsm8k", domain="math", prompt="p", gold_answer="42")
    traces = [["alpha beta", "gamma delta"], ["alpha beta", "gamma delta"], ["xyz uvw", "pqr"]]
    mk = lambda st: Sample(answer="42", reasoning_steps=st, latency_s=1.0, n_tokens=10)
    obs = QueryObservation(query=q, runs=[mk(t) for t in traces], paraphrase_runs=[])
    trace_strs = [" ".join(t) for t in traces]
    rqeval_score = ExplainabilityMetric({"stability_runs": K}).compute_instance(trace_strs)
    assert m.per_query_value(obs) == pytest.approx(float(rqeval_score), abs=0.001)


@requires_rqeval
def test_monitor_uses_rqeval_by_default():
    """ReasoningMonitor with use_rqeval=True must build rqeval adapter metrics."""
    from reasoning_monitor.metrics.rqeval_adapters import RQEVAL_ADAPTER_CLASSES
    from reasoning_monitor.models.clients import SimulatedClient
    from reasoning_monitor.monitor import MonitorConfig, ReasoningMonitor
    client = SimulatedClient(name="test", seed=0)
    mon = ReasoningMonitor(client, MonitorConfig(use_rqeval=True))
    for name, metric in mon.metrics.items():
        assert type(metric) in RQEVAL_ADAPTER_CLASSES.values(), \
            f"{name}: expected rqeval adapter, got {type(metric).__name__}"


def test_publication_mode_rejects_auto_scorer():
    """publication_mode=True must reject stability_scorer='auto' (bug fix)."""
    from reasoning_monitor.monitor import MonitorConfig
    with pytest.raises(ValueError, match="bertscore"):
        MonitorConfig(publication_mode=True, coherence_scorer="nli", stability_scorer="auto")
    # Only 'bertscore' is accepted
    cfg = MonitorConfig(publication_mode=True, coherence_scorer="nli", stability_scorer="bertscore")
    assert cfg.stability_scorer == "bertscore"


def test_publication_mode_requires_rqeval():
    """publication_mode=True must reject use_rqeval=False (safety guard)."""
    from reasoning_monitor.monitor import MonitorConfig
    with pytest.raises(ValueError, match="use_rqeval"):
        MonitorConfig(
            publication_mode=True, coherence_scorer="nli",
            stability_scorer="bertscore", use_rqeval=False,
        )
    # use_rqeval=True is accepted
    cfg = MonitorConfig(
        publication_mode=True, coherence_scorer="nli",
        stability_scorer="bertscore", use_rqeval=True,
    )
    assert cfg.use_rqeval is True


# ══════════════════════════════════════════════════════════════════════════════
# v0.2.1 — configuration reachability, preflight, raw-record persistence
#
# Each test below corresponds to a defect that shipped in v0.2.0 and was silent:
# a config value that could not reach the thing it configured, a scorer that
# fell back without saying so, or a generation that was paid for and discarded.
# ══════════════════════════════════════════════════════════════════════════════

def _tiny_stream(n: int = 6):
    from reasoning_monitor.data.loaders import load_all
    ds = load_all(["gsm8k"], limit_per_dataset=30, seed=1)
    return build_stream(SCENARIOS["ST2"], ds, seed=1)[:n]


def _monitor(**kwargs) -> ReasoningMonitor:
    cfg = MonitorConfig(window_size=3, use_rqeval=False, **kwargs)
    return ReasoningMonitor(SimulatedClient(name="t", seed=1), cfg)


# ── scenario names ────────────────────────────────────────────────────────────

def test_unknown_scenario_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError) as e:
        _validate_scenarios(["ST2", "S1"])
    assert "S1" in str(e.value) and "ST1" in str(e.value)


def test_quick_preset_uses_scenarios_that_exist():
    """--quick asked for 'S1'/'S2' until v0.2.1, so CI's smoke step always died."""
    cfg = apply_quick_preset(ExperimentConfig())
    _validate_scenarios(cfg.scenarios)
    assert cfg.use_rqeval is False
    assert cfg.coherence_scorer == "lexical"
    assert cfg.stability_scorer == "jaccard"


@pytest.mark.parametrize(
    "name",
    [
        "default.yaml",
        "live_asu.yaml",
        "step_band_calibration_llama_asu.yaml",
        "step_band_calibration_gpt_mistral_asu.yaml",
        "proposal_s1_pilot_gpt_llama_asu.yaml",
        "proposal_s1_answer_first_pilot_asu.yaml",
        "proposal_s1_answer_last_calibration_asu.yaml",
        "proposal_s1_answer_last_validation_asu.yaml",
        "proposal_s1_format_stress_asu.yaml",
        "proposal_s1_llama4_scout_stress_asu.yaml",
        "proposal_s1_llama4_scout_validation_asu.yaml",
        "proposal_s1_mistral_validation_asu.yaml",
        "proposal_s1_metric_health_gpt_llama4_asu.yaml",
    ],
)
def test_shipped_configs_load_into_experiment_config(name):
    """Every key in a shipped YAML must be a real field, every scenario real."""
    raw = yaml.safe_load((_REPO_ROOT / "configs" / name).read_text())
    cfg = ExperimentConfig(**raw)      # TypeError here = unreachable config key
    _validate_scenarios(cfg.scenarios, cfg.stream_design)


# ── config reachability ───────────────────────────────────────────────────────

def test_experiment_config_reaches_the_monitor():
    """k_repeats / L / scorers / publication_mode had no path out of the YAML."""
    cfg = ExperimentConfig(k_repeats=4, l_paraphrases=2, stability_scorer="jaccard",
                           coherence_scorer="lexical", use_rqeval=False)
    mon = _make_monitor({"backend": "simulated", "name": "m"}, W=5, cfg=cfg)
    assert mon.config.k_repeats == 4
    assert mon.config.l_paraphrases == 2
    assert mon.config.stability_scorer == "jaccard"
    assert mon.config.publication_mode is False


def test_publication_mode_is_reachable_from_experiment_config():
    with pytest.raises(ValueError, match="coherence_scorer"):
        _make_monitor(
            {"backend": "simulated", "name": "m"}, W=5,
            cfg=ExperimentConfig(publication_mode=True, coherence_scorer="lexical"),
        )


# ── preflight ─────────────────────────────────────────────────────────────────

def test_preflight_warns_when_ls_falls_back_to_the_lexical_scorer():
    report = _monitor(coherence_scorer="lexical").preflight()
    assert report["scorers"]["LS"]["ready"] is False
    assert any("LS scorer" in w for w in report["warnings"])
    assert report["problems"] == []          # lexical was asked for, so not fatal


def test_preflight_rejects_k_repeats_at_temperature_zero():
    """Identical repeats make CS and SS 1.0 by construction, not by measurement."""
    class _ZeroTempClient(SimulatedClient):
        temperature = 0.0

    mon = ReasoningMonitor(
        _ZeroTempClient(name="live-like", seed=1),
        MonitorConfig(window_size=3, k_repeats=5, use_rqeval=False),
    )
    report = mon.preflight()
    assert any("temperature=0" in p for p in report["problems"])
    with pytest.raises(ValueError, match="preflight failed"):
        mon.preflight(strict=True)


def test_preflight_is_quiet_when_repeats_are_sampled():
    class _SampledClient(SimulatedClient):
        temperature = 0.7

    mon = ReasoningMonitor(
        _SampledClient(name="live-like", seed=1),
        MonitorConfig(window_size=3, k_repeats=5, use_rqeval=False),
    )
    assert not any("temperature" in p for p in mon.preflight()["problems"])


# ── degenerate metrics ────────────────────────────────────────────────────────

def test_degenerate_metrics_flags_a_constant_channel():
    snaps = [MetricSnapshot(t=i, values={"LS": 1.0, "CQ": 0.5 + i / 100},
                            composite=0.5) for i in range(5)]
    result = MonitorResult(snapshots=snaps, alarms=[], per_query=pd.DataFrame())
    assert result.degenerate_metrics() == ["LS"]


# ── raw record persistence ────────────────────────────────────────────────────

def test_every_model_call_is_recorded_verbatim():
    stream = _tiny_stream(4)
    result = _monitor(k_repeats=3, l_paraphrases=2).run(stream)
    assert len(result.records) == len(stream) * (3 + 2)
    first = result.records[0]
    assert first["raw_text"] and first["prompt"] and first["ok"] is True
    assert {r["role"] for r in result.records} == {"run", "paraphrase"}


def test_run_experiment_writes_records_and_per_query(tmp_path):
    """The campaign must be re-scorable offline without re-issuing any call."""
    run_experiment(ExperimentConfig(
        output_dir=str(tmp_path), items_per_dataset=30, scenarios=["ST3"],
        window_sizes=[3], k_repeats=2, l_paraphrases=1, use_rqeval=False,
        run_ablation=False, run_strategy_comparison=False,
        models=[{"backend": "simulated", "name": "sim"}],
    ))
    data = tmp_path / "data"
    records = list(data.glob("records_*.jsonl"))
    assert records, "no raw generations were persisted"
    rows = [json.loads(line) for line in records[0].read_text().splitlines()]
    assert rows and all("raw_text" in r and "gold_answer" in r for r in rows)
    assert list(data.glob("per_query_*.csv"))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "preflight" in manifest and "degenerate_metrics" in manifest


def test_pilot_query_cap_limits_model_calls(tmp_path):
    manifest = run_experiment(ExperimentConfig(
        output_dir=str(tmp_path), items_per_dataset=30, scenarios=["ST3"],
        max_queries_per_scenario=7, window_sizes=[3], k_repeats=2,
        l_paraphrases=1, use_rqeval=False, run_ablation=False,
        run_strategy_comparison=False,
        models=[{"backend": "simulated", "name": "sim"}],
    ))

    assert manifest["config"]["max_queries_per_scenario"] == 7
    records = next((tmp_path / "data").glob("records_*.jsonl"))
    assert len(records.read_text().splitlines()) == 7 * (2 + 1)


def test_replication_campaign_uses_only_selected_dataset(tmp_path):
    manifest = run_experiment(ExperimentConfig(
        output_dir=str(tmp_path), datasets=["gsm8k"], items_per_dataset=30,
        scenarios=["ST3"], seeds=[7], window_sizes=[3], k_repeats=1,
        l_paraphrases=0, use_rqeval=False, run_ablation=False,
        run_strategy_comparison=False, run_replication=True,
        replication_dataset="arc_challenge", replication_scenarios=["ST3"],
        replication_seeds=[11], replication_items=30,
        models=[{"backend": "simulated", "name": "sim"}],
    ))

    replication_dir = tmp_path / "replication_arc_challenge"
    nested = json.loads((replication_dir / "manifest.json").read_text())
    assert manifest["replication"]["dataset"] == "arc_challenge"
    assert manifest["replication"]["seeds"] == [11]
    assert set(nested["datasets"]) == {"arc_challenge"}
    assert nested["scenarios"] == ["ST3"]
    assert nested["config"]["run_replication"] is False


@requires_rqeval
def test_es_per_query_value_is_a_number_not_nan():
    """ES_raw was NaN for every row: the adapter discarded its own computation."""
    from reasoning_monitor.metrics.rqeval_adapters import ESAdapter
    m = ESAdapter(window_size=3)
    v = m.per_query_value(make_obs("42", "42", tokens=100))
    assert not math.isnan(v) and 0.0 <= v <= 1.0


def test_missing_rs_does_not_stop_other_metric_clocks():
    """A C-subset exclusion must not suppress the whole snapshot."""
    from reasoning_monitor.types import Query, QueryObservation, Sample

    class _Metric:
        def __init__(self, value):
            self.value = value

        def per_query_value(self, _obs):
            return self.value

        def update(self, _obs):
            return self.value

    query = Query("q", 1, "gsm8k", "math", "1+1", "2")
    sample = Sample("2", ("1+1=2",), 0.0, 1)
    observation = QueryObservation(query, (sample,), ())
    monitor = _monitor(k_repeats=1, l_paraphrases=0)
    monitor._observe = lambda _q: observation
    monitor.metrics = {
        name: _Metric(float("nan") if name == "RS" else 0.8)
        for name in METRIC_ORDER
    }

    result = monitor.run([query])

    assert len(result.snapshots) == 1
    assert "RS" not in result.snapshots[0].values
    assert set(result.snapshots[0].values) == set(METRIC_ORDER) - {"RS"}
    assert result.snapshots[0].composite == pytest.approx(0.8)


def test_staged_scenario_is_evaluated_per_transition():
    """ST2 must emit separate easy-to-medium and medium-to-hard rows."""
    from types import SimpleNamespace

    from reasoning_monitor.experiment import ExperimentConfig, _evaluate_scenario
    from reasoning_monitor.types import Alarm, MetricSnapshot

    snapshots = [
        MetricSnapshot(t=t, values={m: 0.9 - t / 1000 for m in METRIC_ORDER}, composite=0.5)
        for t in range(1, 151)
    ]
    result = SimpleNamespace(snapshots=snapshots, alarms=[
        Alarm(t=55, metric="CQ", detector="cusum", statistic=1.0),
        Alarm(t=110, metric="CQ", detector="cusum", statistic=1.0),
    ])
    detector = SimpleNamespace(name="cusum")
    monitor = SimpleNamespace(
        detectors={m: detector for m in METRIC_ORDER}, raw_detectors={}
    )
    changes, detections = _evaluate_scenario(
        result, SCENARIOS["ST2"], "model", 20, "ST2",
        ExperimentConfig(guard=0), monitor,
    )

    assert {row["transition"] for row in changes} == {
        "easy_to_medium", "medium_to_hard"
    }
    cq = [row for row in detections if row["metric"] == "CQ"]
    assert [row["effective_t_star"] for row in cq] == [50, 100]
    assert [row["delay"] for row in cq] == [5.0, 10.0]


def test_cross_seed_alarm_order_reports_concordance_and_pairwise_rates():
    from reasoning_monitor.evaluation.metrics_eval import cross_seed_alarm_order

    rows = []
    delays = {1: {"SS": -5, "CS": -2, "CQ": 3},
              2: {"SS": -4, "CS": -1, "CQ": 2},
              3: {"SS": -3, "CS": 0, "CQ": 4}}
    for seed, metric_delays in delays.items():
        for metric, delay in metric_delays.items():
            rows.append({
                "seed": seed, "scenario": "ST2", "model": "m",
                "transition": "easy_to_medium", "window_size": 20,
                "strategy": "distribution_based", "metric": metric,
                "detected": True, "delay": delay, "level": "drift",
                "effective_t_star": 50,
            })

    summary, ranks = cross_seed_alarm_order(
        pd.DataFrame(rows), n_permutations=99, random_seed=7
    )

    assert len(summary) == 1
    assert summary.iloc[0]["kendalls_w"] == pytest.approx(1.0)
    assert summary.iloc[0]["ss_before_cq_rate"] == pytest.approx(1.0)
    assert summary.iloc[0]["cs_before_cq_rate"] == pytest.approx(1.0)
    assert len(ranks) == 9


def test_strategy_comparison_replays_saved_metrics_without_generation(tmp_path, monkeypatch):
    from reasoning_monitor.experiment import _run_strategy_comparison

    (tmp_path / "data").mkdir()
    snapshots = [
        MetricSnapshot(t=t, values={m: 0.9 - t / 1000 for m in METRIC_ORDER}, composite=0.5)
        for t in range(1, 151)
    ]
    per_query = pd.DataFrame({
        "t": range(1, 151),
        "CQ_raw": [1.0 if t < 75 else 0.0 for t in range(1, 151)],
    })
    saved = MonitorResult(snapshots, [], per_query, records=[])
    cfg = ExperimentConfig(
        scenarios=["ST2"], window_sizes=[20], seeds=[1],
        models=[{"backend": "simulated", "name": "m"}],
        use_rqeval=False, run_ablation=False,
    )
    monkeypatch.setattr(
        ReasoningMonitor, "run",
        lambda *_args, **_kwargs: pytest.fail("strategy replay generated new samples"),
    )

    frame = _run_strategy_comparison(
        cfg, {}, tmp_path, {("ST2", "m", 1, 20): saved}
    )

    assert set(frame["strategy"]) == {
        "distribution_based", "proposal_mapping", "adwin_all"
    }
    assert (tmp_path / "data" / "h2_strategy_comparison.csv").exists()


def test_gsm8k_solution_steps_are_preserved_and_drive_difficulty():
    from reasoning_monitor.data.loaders import (
        _solution_step_count,
        difficulty_proxy_summary,
        tag_difficulty_terciles,
    )

    assert _solution_step_count("gsm8k", "<<2+3=5>> then <<5*4=20>> #### 20") == 2
    queries = [
        Query(
            qid=f"q{i}", t=0, dataset="gsm8k", domain="math",
            prompt=f"Problem {i} with 1 and 2", gold_answer=str(i),
            solution_steps=i,
        )
        for i in range(1, 10)
    ]
    tiers = tag_difficulty_terciles(queries, seed=1)
    assert max(q.solution_steps for q in tiers["easy"]) < min(
        q.solution_steps for q in tiers["medium"]
    )
    summary = difficulty_proxy_summary(queries, seed=1)
    assert summary["monotonic_solution_steps"] is True


def test_solution_step_bands_are_fixed_and_exclude_unlabelled_items():
    from reasoning_monitor.data.loaders import (
        difficulty_proxy_summary,
        tag_difficulty_step_bands,
    )

    queries = [
        Query(
            qid=f"q{i}", t=0, dataset="gsm8k", domain="math",
            prompt=f"Problem {i}", gold_answer=str(i), solution_steps=steps,
        )
        for i, steps in enumerate([1, 2, 3, 4, 5, 8, None], start=1)
    ]

    tiers = tag_difficulty_step_bands(queries, seed=7)
    assert {q.solution_steps for q in tiers["easy"]} == {1, 2}
    assert {q.solution_steps for q in tiers["medium"]} == {3, 4}
    assert {q.solution_steps for q in tiers["hard"]} == {5, 8}
    assert "q7" not in {q.qid for items in tiers.values() for q in items}

    summary = difficulty_proxy_summary(queries, seed=7, scheme="solution_steps")
    assert summary["scheme"] == "solution_steps"
    assert summary["n_input"] == 7
    assert summary["n_excluded"] == 1
    assert summary["monotonic_solution_steps"] is True


def test_solution_step_scheme_builds_declared_stationary_tiers():
    from reasoning_monitor.data.loaders import LoadedDataset
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream, resize_regimes

    queries = [
        Query(
            qid=f"q{i}", t=0, dataset="gsm8k", domain="math",
            prompt=f"Problem {i}", gold_answer=str(i), solution_steps=steps,
        )
        for i, steps in enumerate([1, 2, 3, 4, 5, 6] * 5, start=1)
    ]
    dataset = LoadedDataset("gsm8k", queries, "test", "hash")
    spec = resize_regimes(SCENARIOS["ST4"], 5)

    stream = build_stream(
        spec,
        {"gsm8k": dataset},
        seed=3,
        difficulty_scheme="solution_steps",
    )

    assert len(stream) == 5
    assert {q.difficulty for q in stream} == {"medium"}
    assert all(q.solution_steps in {3, 4} for q in stream)


# ── accepted-proposal multi-dataset S1 ───────────────────────────────────────

def _proposal_datasets(n: int = 40):
    from reasoning_monitor.data.loaders import LoadedDataset

    domains = {"gsm8k": "math", "arc_challenge": "science", "math": "math_hard"}
    return {
        name: LoadedDataset(
            name=name,
            queries=[
                Query(
                    qid=f"{name}-{i}", t=0, dataset=name, domain=domain,
                    prompt=f"{name} problem {i}", gold_answer=str(i),
                )
                for i in range(n)
            ],
            source="test",
            content_hash=f"hash-{name}",
        )
        for name, domain in domains.items()
    }


def test_proposal_s1_is_gradual_balanced_unique_and_deterministic():
    from reasoning_monitor.data.proposal_scenarios import build_proposal_s1_stream
    from reasoning_monitor.data.scenarios import effective_change_point

    spec, stream, plan = build_proposal_s1_stream(
        _proposal_datasets(), seed=7, queries_per_dataset=30
    )
    _spec2, stream2, _plan2 = build_proposal_s1_stream(
        _proposal_datasets(), seed=7, queries_per_dataset=30
    )

    assert len(stream) == 90
    assert len({query.qid for query in stream}) == 90
    assert {name: sum(query.dataset == name for query in stream) for name in (
        "gsm8k", "arc_challenge", "math"
    )} == {"gsm8k": 30, "arc_challenge": 30, "math": 30}
    assert [query.qid for query in stream] == [query.qid for query in stream2]
    assert {row["dataset"] for row in plan if row["phase"] == 1} == {"gsm8k"}
    assert {row["dataset"] for row in plan if row["phase"] == 6} == {"math"}
    phase2 = [row["dataset"] for row in plan if row["phase"] == 2]
    assert phase2.count("gsm8k") == 11 and phase2.count("arc_challenge") == 4
    assert effective_change_point(spec) == 45


def test_proposal_s1_uses_a_separate_scenario_catalogue():
    _validate_scenarios(["S1"], "proposal_multidataset")
    _validate_scenarios(["S1"], "proposal_answer_order_calibration")
    _validate_scenarios(["S1"], "proposal_frozen_selection")
    with pytest.raises(ValueError, match="stream_design"):
        _validate_scenarios(["S1"], "e2h_gsm8k")
    with pytest.raises(ValueError, match="Only proposal S1"):
        _validate_scenarios(["S2"], "proposal_multidataset")


def test_saved_records_can_be_reordered_without_generation():
    from dataclasses import replace

    class _NoGenerateClient:
        name = "replay-only"
        temperature = 0.7

        def generate(self, *_args, **_kwargs):
            pytest.fail("record replay called generate()")

    config = MonitorConfig(
        window_size=2, k_repeats=1, l_paraphrases=0,
        metrics=("CQ",), weights={"CQ": 1.0}, use_rqeval=False,
    )
    stream = _tiny_stream(5)
    original = ReasoningMonitor(
        SimulatedClient(name="source", seed=7), config
    ).run(stream)
    reordered = [
        replace(query, t=index)
        for index, query in enumerate(reversed(stream), start=1)
    ]

    replay = ReasoningMonitor(_NoGenerateClient(), config).run_from_records(
        reordered, original.records
    )

    assert replay.per_query["qid"].tolist() == [query.qid for query in reordered]
    assert [record["source_t"] for record in replay.records] == [5, 4, 3, 2, 1]
    assert [record["t"] for record in original.records] == [1, 2, 3, 4, 5]
    original_cq = original.per_query.set_index("qid")["CQ_raw"].to_dict()
    replay_cq = replay.per_query.set_index("qid")["CQ_raw"].to_dict()
    assert replay_cq == original_cq


def test_s1_order_controls_are_matched_stationary_and_deterministic():
    from reasoning_monitor.data.proposal_scenarios import build_proposal_s1_stream
    from reasoning_monitor.evaluation.order_controls import build_s1_order_controls

    _spec, stream, _plan = build_proposal_s1_stream(
        _proposal_datasets(), seed=7, queries_per_dataset=30
    )
    controls = build_s1_order_controls(stream, seed=10042)
    controls_again = build_s1_order_controls(stream, seed=10042)

    assert set(controls) == {
        "ordered_s1", "global_shuffled", "stationary_gsm8k",
        "stationary_arc_challenge", "stationary_math",
    }
    assert {query.qid for query in controls["global_shuffled"]} == {
        query.qid for query in stream
    }
    assert [query.qid for query in controls["global_shuffled"]] != [
        query.qid for query in stream
    ]
    for dataset in ("gsm8k", "arc_challenge", "math"):
        arm = controls[f"stationary_{dataset}"]
        assert len(arm) == 30
        assert {query.dataset for query in arm} == {dataset}
        assert [query.t for query in arm] == list(range(1, 31))
    assert {
        arm: [query.qid for query in queries]
        for arm, queries in controls.items()
    } == {
        arm: [query.qid for query in queries]
        for arm, queries in controls_again.items()
    }


def test_s1_permutations_are_unique_matched_and_deterministic():
    from reasoning_monitor.evaluation.permutation_controls import (
        generate_matched_permutations,
    )

    first = generate_matched_permutations(12, 100, seed=21042)
    second = generate_matched_permutations(12, 100, seed=21042)

    assert first == second
    assert len(first) == len(set(first)) == 100
    assert tuple(range(12)) not in first
    assert all(sorted(permutation) == list(range(12)) for permutation in first)


def test_s1_decline_score_has_decline_direction():
    from reasoning_monitor.evaluation.permutation_controls import decline_score

    declining = decline_score([1, 1, 1, 0, 0, 0])
    stationary = decline_score([1, 0, 1, 1, 0, 1])
    improving = decline_score([0, 0, 0, 1, 1, 1])

    assert declining > stationary
    assert stationary == pytest.approx(0.0)
    assert improving == pytest.approx(-declining)


def test_s1_cusum_selects_smallest_common_threshold():
    from reasoning_monitor.evaluation.cusum_calibration import select_common_threshold

    frame = pd.DataFrame([
        {"model": "a", "threshold": 0.5, "specificity_pass": False},
        {"model": "b", "threshold": 0.5, "specificity_pass": True},
        {"model": "a", "threshold": 1.0, "specificity_pass": True},
        {"model": "b", "threshold": 1.0, "specificity_pass": True},
        {"model": "a", "threshold": 2.0, "specificity_pass": True},
        {"model": "b", "threshold": 2.0, "specificity_pass": True},
    ])

    assert select_common_threshold(frame, ["a", "b"]) == 1.0


def test_s1_cusum_calibration_and_validation_permutations_are_disjoint():
    from reasoning_monitor.evaluation.permutation_controls import (
        generate_matched_permutations,
    )

    calibration = generate_matched_permutations(20, 100, seed=31042)
    validation = generate_matched_permutations(20, 100, seed=41042)

    assert not (set(calibration) & set(validation))


def test_experiment_propagates_detector_overrides():
    cfg = ExperimentConfig(
        models=[{"backend": "simulated", "name": "sim"}],
        detector_overrides={
            "CQ": {"name": "cusum", "threshold": 7.5, "drift": 0.01, "warmup": 20}
        },
        run_strategy_comparison=False,
        use_rqeval=False,
    )

    monitor = _make_monitor(cfg.models[0], 10, cfg)

    assert monitor.detectors["CQ"].name == "cusum"
    assert monitor.detectors["CQ"].threshold == 7.5
    assert monitor.detectors["CQ"].drift == 0.01
    assert monitor.detectors["CQ"].warmup == 20


def test_proposal_s1_cq_delay_gate_uses_frozen_limit():
    from reasoning_monitor.experiment import _proposal_s1_cq_delay_gate

    cfg = ExperimentConfig(
        proposal_s1_max_cq_delay=10,
        run_strategy_comparison=False,
    )
    detections = pd.DataFrame([
        {
            "scenario": "S1", "metric": "CQ", "level": "drift",
            "model": "pass", "seed": 42, "window_size": 10,
            "detector": "cusum", "detected": True, "delay": 10,
            "n_false_alarms": 0,
        },
        {
            "scenario": "S1", "metric": "CQ", "level": "drift",
            "model": "late", "seed": 42, "window_size": 10,
            "detector": "cusum", "detected": True, "delay": 11,
            "n_false_alarms": 0,
        },
        {
            "scenario": "S1", "metric": "CS", "level": "drift",
            "model": "ignored", "seed": 42, "window_size": 10,
            "detector": "cusum", "detected": True, "delay": 0,
            "n_false_alarms": 0,
        },
    ])

    gate = _proposal_s1_cq_delay_gate(cfg, detections)

    assert gate["model"].tolist() == ["pass", "late"]
    assert gate["gate_pass"].tolist() == [True, False]


def test_proposal_s1_kl_config_locks_budget_detector_and_delay():
    from scripts.estimate_campaign import estimate

    raw = yaml.safe_load(
        (_REPO_ROOT / "configs" / "proposal_s1_kl_validation_gpt_llama_asu.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = ExperimentConfig(**raw)

    assert cfg.k_repeats == 5 and cfg.l_paraphrases == 3
    assert cfg.detector_overrides["CQ"]["threshold"] == 7.5
    assert cfg.proposal_s1_max_cq_delay == 10
    assert cfg.run_strategy_comparison is False
    assert estimate(raw)["model_calls"] == 1440


def test_kl_repeat_diversity_rate_uses_verbatim_k_runs():
    from reasoning_monitor.evaluation.kl_validation import repeat_diversity_rate

    records = []
    for qid, texts in {
        "q1": ["same"] * 5,
        "q2": ["a", "b", "a", "b", "a"],
    }.items():
        records.extend(
            {"qid": qid, "role": "run", "raw_text": text}
            for text in texts
        )
    records.append({"qid": "q2", "role": "paraphrase", "raw_text": "ignored"})

    diverse, complete, rate = repeat_diversity_rate(records, 5)

    assert (diverse, complete) == (1, 2)
    assert rate == pytest.approx(0.5)


def test_detector_development_rebuilds_rs_and_es_windows():
    from reasoning_monitor.evaluation.detector_development import (
        build_window_signal,
    )

    indices = (0, 1, 2, 3)
    rs = build_window_signal(
        "RS",
        [1.0, float("nan"), 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [10, 20, 10, 20],
        indices,
        window_size=2,
    )
    es = build_window_signal(
        "ES",
        [0.0] * 4,
        [1.0, 0.0, 1.0, 1.0],
        [10, 20, 10, 20],
        indices,
        window_size=2,
    )

    assert rs == [(3, 0.5), (4, 0.5)]
    assert es == [(2, 0.5), (3, 0.5), (4, 0.5)]


def test_optimized_adwin_grid_matches_detector_implementation():
    from reasoning_monitor.detectors.detectors import build_detector
    from reasoning_monitor.evaluation.detector_development import (
        _adwin_first_alarms,
    )

    signal = [(t, 1.0 if t < 45 else 0.0) for t in range(1, 91)]
    deltas = (0.05, 0.002, 0.00001)
    optimized = _adwin_first_alarms(signal, deltas)

    expected = {}
    for delta in deltas:
        detector = build_detector("adwin", delta=delta)
        expected[delta] = None
        for t, value in signal:
            alarm = detector.update(value, t, "CS")
            if alarm is not None:
                expected[delta] = t
                break

    assert optimized == expected


def test_e2h_llama_kl_config_locks_budget_and_method():
    from scripts.estimate_campaign import estimate

    raw = yaml.safe_load(
        (_REPO_ROOT / "configs" / "e2h_stepband_llama_kl_pilot_asu.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = ExperimentConfig(**raw)

    assert cfg.seeds == [43]
    assert cfg.scenarios == ["ST2"]
    assert cfg.difficulty_scheme == "solution_steps"
    assert cfg.queries_per_regime == 30
    assert cfg.window_sizes == [10]
    assert cfg.k_repeats == 5 and cfg.l_paraphrases == 3
    assert cfg.detector_strategy == "proposal_mapping"
    assert estimate(raw)["model_calls"] == 720


def test_e2h_cq_band_gate_uses_frozen_adjacent_gap():
    from reasoning_monitor.evaluation.e2h_kl_validation import cq_band_gate

    frame = pd.DataFrame({
        "t": range(1, 91),
        "difficulty": ["easy"] * 30 + ["medium"] * 30 + ["hard"] * 30,
        "CQ_raw": [1.0] * 30 + [0.8] * 30 + [0.7] * 30,
    })

    passed = cq_band_gate(frame)
    frame.loc[60:89, "CQ_raw"] = 0.71
    failed = cq_band_gate(frame)

    assert passed["cq_gate_pass"] is True
    assert passed["medium_hard_gap"] == pytest.approx(0.10)
    assert failed["cq_gate_pass"] is False


def test_arc_prompt_contains_choices_and_numeric_key_is_mapped():
    from reasoning_monitor.data.loaders import _arc_prompt_and_gold

    prompt, gold = _arc_prompt_and_gold({
        "question": "Which option?",
        "choices": {"label": ["A", "B"], "text": ["first", "second"]},
        "answerKey": "2",
    })

    assert "A. first" in prompt and "B. second" in prompt
    assert gold == "B"


def test_math_gold_extraction_handles_nested_and_last_boxed_expression():
    from reasoning_monitor.data.loaders import _extract_gold

    solution = r"First \\boxed{1}, finally \\boxed{\\frac{3}{4}}."
    assert _extract_gold("math", solution) == r"\\frac{3}{4}"


def test_scientific_dataset_load_can_forbid_synthetic_fallback(monkeypatch, tmp_path):
    from reasoning_monitor.data import loaders

    monkeypatch.setattr(loaders, "_load_from_hub", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="allow_synthetic=False"):
        loaders.load_dataset_queries(
            "arc_challenge", cache_dir=tmp_path, allow_synthetic=False
        )


def test_live_client_retries_transient_errors(monkeypatch):
    import io
    import json
    import urllib.error
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    attempts = {"n": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {"content": "work. Answer: 2"},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 4},
            }).encode()

    def urlopen(_request, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError("url", 429, "rate limited", {}, io.BytesIO())
        return _Response()

    sleeps = []
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)
    client = OpenAICompatibleClient(
        "live", "model", api_key_env="TEST_API_KEY", max_retries=3, backoff_base=0.5
    )

    sample = client.generate("1+1")

    assert sample.ok is True
    assert sample.finish_reason == "stop"
    assert attempts["n"] == 3
    assert sleeps == [0.5, 1.0]

    attempts["n"] = 0
    sleeps.clear()
    patient = OpenAICompatibleClient(
        "live", "model", api_key_env="TEST_API_KEY", max_retries=3, backoff_base=0.5,
        rate_limit_backoff=60.0,
    )
    assert patient.generate("1+1").ok is True
    assert sleeps == [60.0, 60.0]


def test_live_client_records_permanent_failure(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError("url", 400, "bad request", {}, io.BytesIO())
        ),
    )
    client = OpenAICompatibleClient("live", "model", api_key_env="TEST_API_KEY")

    sample = client.generate("bad")

    assert sample.ok is False
    assert "HTTP 400" in sample.raw_text


def test_live_client_preserves_http_error_body(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(
                "url", 400, "bad request", {},
                io.BytesIO(b'{"error":"unknown model"}'),
            )
        ),
    )
    client = OpenAICompatibleClient("live", "model", api_key_env="TEST_API_KEY")

    sample = client.generate("bad")

    assert 'body={"error":"unknown model"}' in sample.raw_text


def test_live_client_can_omit_sampling_parameters(monkeypatch):
    import json
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"completion_tokens": 1},
            }).encode()

    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAICompatibleClient(
        "probe", "model", api_key_env="TEST_API_KEY",
        temperature=None, max_tokens=None,
    )

    assert client.generate("test").ok is True
    assert "temperature" not in captured
    assert "max_tokens" not in captured


def test_live_client_can_forward_per_call_seed(monkeypatch):
    import json
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"completion_tokens": 1},
            }).encode()

    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAICompatibleClient(
        "probe", "model", api_key_env="TEST_API_KEY", send_seed=True
    )

    assert client.generate("test", seed=37).ok is True
    assert captured["seed"] == 37


def test_live_client_can_send_caller_managed_message_history(monkeypatch):
    import json
    import urllib.request

    from reasoning_monitor.models.clients import OpenAICompatibleClient

    monkeypatch.setenv("TEST_API_KEY", "secret")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "FINAL_ANSWER: 7"}}],
                "usage": {"completion_tokens": 3},
            }).encode()

    def urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAICompatibleClient("live", "model", api_key_env="TEST_API_KEY")
    messages = [
        {"role": "system", "content": "Continue the case."},
        {"role": "user", "content": "Initial question"},
        {"role": "assistant", "content": "FINAL_ANSWER: 5"},
        {"role": "user", "content": "Follow-up question"},
    ]

    sample = client.generate_messages(messages)

    assert sample.answer == "7"
    assert captured["messages"] == messages


def test_make_paraphrases_prefers_frozen_query_variants():
    from reasoning_monitor.models.clients import make_paraphrases

    query = Query(
        qid="frozen",
        t=1,
        dataset="gsm8k",
        domain="math",
        prompt="Original prompt",
        gold_answer="1",
        paraphrases=("First rewrite", "Second rewrite", "Third rewrite"),
    )

    assert make_paraphrases(query, 2) == ["First rewrite", "Second rewrite"]
    with pytest.raises(ValueError, match="has 3 frozen paraphrases but L=4"):
        make_paraphrases(query, 4)


def test_checkpoint_resume_does_not_repeat_completed_calls(tmp_path):
    class _CountingClient(SimulatedClient):
        def __init__(self):
            super().__init__(name="counting", seed=1)
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return super().generate(*args, **kwargs)

    stream = _tiny_stream(3)
    checkpoint = tmp_path / "records.jsonl"
    first_client = _CountingClient()
    first = ReasoningMonitor(
        first_client, MonitorConfig(window_size=2, k_repeats=1, l_paraphrases=0,
                                    use_rqeval=False)
    )
    first.run(stream[:2], checkpoint_path=checkpoint)
    assert first_client.calls == 2

    resumed_client = _CountingClient()
    resumed = ReasoningMonitor(
        resumed_client, MonitorConfig(window_size=2, k_repeats=1, l_paraphrases=0,
                                      use_rqeval=False)
    )
    result = resumed.run(stream, checkpoint_path=checkpoint, resume=True)

    assert resumed_client.calls == 1
    assert len(result.records) == 3


def test_checkpoint_resume_completes_only_missing_calls_within_query(tmp_path):
    class _CountingClient(SimulatedClient):
        def __init__(self):
            super().__init__(name="counting", seed=1)
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return super().generate(*args, **kwargs)

    stream = _tiny_stream(1)
    checkpoint = tmp_path / "records.jsonl"
    first = ReasoningMonitor(
        _CountingClient(),
        MonitorConfig(window_size=2, k_repeats=1, l_paraphrases=0,
                      use_rqeval=False),
    )
    first.run(stream, checkpoint_path=checkpoint)

    client = _CountingClient()
    resumed = ReasoningMonitor(
        client,
        MonitorConfig(window_size=2, k_repeats=3, l_paraphrases=0,
                      use_rqeval=False),
    )
    result = resumed.run(stream, checkpoint_path=checkpoint, resume=True)

    assert client.calls == 2
    assert len(result.records) == 3
    assert len(checkpoint.read_text().splitlines()) == 3


def test_gateway_probe_cases_isolate_controls():
    from reasoning_monitor.models.clients import gateway_probe_cases

    cases = gateway_probe_cases()
    assert cases[0] == ("baseline", {})
    assert [name for name, _options in cases] == [
        "baseline",
        "disable_enable_history",
        "disable_enable_search",
        "disable_prompt_context",
        "disable_all",
    ]
    assert cases[-1][1] == {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
