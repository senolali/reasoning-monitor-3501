import json
from pathlib import Path

from reasoning_monitor.metrics.dimensions import ConsistencyScore, RobustnessScore
from reasoning_monitor.metrics.rqeval_adapters import CSAdapter, RSAdapter
from reasoning_monitor.metrics.semantic_answers import load_semantic_policy
from reasoning_monitor.types import Query, QueryObservation, Sample

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "semantic_answer_policy_v1.yaml"
CASES_PATH = ROOT / "configs" / "semantic_answer_policy_v1_cases.json"


def _sample(answer: str) -> Sample:
    return Sample(
        answer=answer,
        reasoning_steps=("reason",),
        latency_s=0.1,
        n_tokens=10,
        raw_text=answer,
    )


def _observation(runs, paraphrases=()):
    query = Query(
        qid="q1",
        t=1,
        dataset="gsm8k",
        domain="math",
        prompt="How long?",
        gold_answer="95",
    )
    return QueryObservation(
        query=query,
        runs=tuple(_sample(answer) for answer in runs),
        paraphrase_runs=tuple(_sample(answer) for answer in paraphrases),
    )


def test_frozen_semantic_policy_cases():
    policy = load_semantic_policy(str(POLICY_PATH))
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    assert len(cases) == 20
    assert all(
        policy.equivalent(case["left"], case["right"], case["dataset"])
        is case["expected"]
        for case in cases
    )


def test_semantic_rs_uses_primary_explicit_unit_without_changing_c_subset():
    observation = _observation(
        ["95 minutes"],
        ["95 minutes", "95 minutes", "5700 seconds"],
    )

    legacy = RobustnessScore(window_size=2, l_paraphrases=3)
    semantic = RobustnessScore(
        window_size=2,
        l_paraphrases=3,
        semantic_answer_policy=str(POLICY_PATH),
    )

    assert legacy.per_query_value(observation) == 2 / 3
    assert semantic.per_query_value(observation) == 1.0


def test_semantic_rs_rejects_same_number_in_a_wrong_explicit_unit():
    observation = _observation(
        ["95 minutes"],
        ["95 minutes", "95 minutes", "95 seconds"],
    )
    semantic = RobustnessScore(
        window_size=2,
        l_paraphrases=3,
        semantic_answer_policy=str(POLICY_PATH),
    )

    assert semantic.per_query_value(observation) == 2 / 3


def test_semantic_cs_treats_converted_units_as_the_same_answer():
    observation = _observation(["95 minutes", "5700 seconds"])

    legacy = ConsistencyScore(window_size=2, k_runs=2)
    semantic = ConsistencyScore(
        window_size=2,
        k_runs=2,
        semantic_answer_policy=str(POLICY_PATH),
    )

    assert legacy.per_query_value(observation) == 0.0
    assert semantic.per_query_value(observation) == 1.0


def test_rqeval_adapters_use_the_opt_in_semantic_policy():
    observation = _observation(
        ["95 minutes", "5700 seconds"],
        ["95 minutes", "95 minutes", "5700 seconds"],
    )
    cs = CSAdapter(
        window_size=2,
        k_runs=2,
        semantic_answer_policy=str(POLICY_PATH),
    )
    rs = RSAdapter(
        window_size=2,
        l_paraphrases=3,
        semantic_answer_policy=str(POLICY_PATH),
    )

    assert cs.per_query_value(observation) == 1.0
    assert rs.per_query_value(observation) == 1.0
