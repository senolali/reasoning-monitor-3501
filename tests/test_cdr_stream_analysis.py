"""Synthetic-only tests for the CDR-Stream analysis pipeline (no network, no GPU)."""

import random

import numpy as np
import pytest

from reasoning_monitor import cdr_stream_analysis as ca
from reasoning_monitor.data import cdr_stream as cs

P_CORRECT = {"E": 0.95, "M": 0.75, "H": 0.25}


def coherence(steps):
    return 1.0 - 0.5 * any("contra" in s for s in steps)


def similarity(left, right):
    out = []
    for a, b in zip(left, right, strict=True):
        sa, sb = set(a.split()), set(b.split())
        out.append(len(sa & sb) / max(len(sa | sb), 1))
    return out


def response(rng, correct, gold, truncated=False):
    answer = str(gold if correct else gold + rng.randint(1, 9))
    text = f"1. step {rng.randint(0, 3)}\n2. step two\nFINAL_ANSWER: {answer}"
    return {"raw_text": text, "answer": answer, "correct": correct, "ok": True,
            "n_tokens": 2048 if truncated else rng.randint(80, 200), "truncated": truncated}


def synthetic_records(seed=0, per_scenario=4, models=("a", "b"), truncate=()):
    rng = random.Random(seed)
    records = []
    for model in models:
        for scenario in ca.CHANGE_POINTS:
            schedule = cs.regime_schedule(scenario)
            for index in range(per_scenario):
                dialogue = f"{scenario}/t/{index:02d}"
                for turn, regime in enumerate(schedule, start=1):
                    gold = rng.randint(10, 99)
                    base = {"model": model, "dialogue_id": dialogue, "scenario": scenario,
                            "template": "t", "turn": turn, "regime": regime,
                            "level": {"E": 1, "M": 5, "H": 9}[regime], "family": "SUM",
                            "gold": gold}
                    p = P_CORRECT[regime]
                    calls = [("dialogue", "primary", 0), ("dialogue", "repeat", 1),
                             ("dialogue", "repeat", 2), ("dialogue", "paraphrase", 0),
                             ("dialogue", "paraphrase", 1), ("control", "primary", 0)]
                    for arm, role, i in calls:
                        cut = (model, dialogue, turn, role) in truncate
                        records.append({**base, "arm": arm, "role": role, "call_index": i,
                                        **response(rng, rng.random() < p and not cut, gold,
                                                   truncated=cut)})
    return records


@pytest.fixture(scope="module")
def frame():
    return ca.score_turns(synthetic_records(), coherence, similarity)


def test_truncated_primary_is_missing_and_sensitivity_scores_it_wrong():
    cut = ("a", "S2/t/00", 10, "primary")
    records = synthetic_records(models=("a",), per_scenario=1, truncate={cut})
    main = ca.score_turns(records, coherence, similarity)
    sens = ca.score_turns(records, coherence, similarity, truncated_as_incorrect=True)
    row = main[(main.dialogue_id == "S2/t/00") & (main.turn == 10)].iloc[0]
    assert np.isnan(row.CQ) and np.isnan(row.LS) and np.isnan(row.CS) and row.primary_truncated
    assert sens[(sens.dialogue_id == "S2/t/00") & (sens.turn == 10)].iloc[0].CQ == 0.0


def test_rs_is_c_subset_and_cs_uses_k_runs(frame):
    wrong = frame[frame.CQ == 0]
    assert wrong.RS.isna().all()
    assert frame.RS[frame.CQ == 1].notna().any()
    assert set(frame.CS.dropna().round(4)) <= {0.0, 0.3333, 1.0}
    assert frame.ES.notna().sum() > 0 and frame.groupby(["model", "dialogue_id"]).ES.apply(
        lambda s: s.iloc[:ca.ES_WINDOW - 1].isna().all()).all()


def test_sign_flip_exact_and_holm():
    result = ca.stratified_sign_flip([-1, -1, -1, -1], ["m"] * 4)
    assert result["p"] == pytest.approx(1 / 16)
    assert ca.holm({"x": 0.01, "y": 0.04})["y"] == pytest.approx(0.04)
    assert ca.holm({"x": 0.01, "y": 0.04})["x"] == pytest.approx(0.02)


def test_h1_finds_synthetic_task_drift_but_not_null(frame):
    h1 = ca.analyze_h1(frame)
    assert h1["h1a"]["S2|CQ"]["mean"] < -0.4 and h1["h1a"]["S2|CQ"]["p"] < 0.01
    # control arm follows the same task drift, so the net contrast is near zero
    assert abs(h1["h1b"]["S2"]["mean"]) < 0.2
    deltas = h1["deltas"]
    assert set(deltas.transition) == {"S1-a", "S1-b", "S2", "S3-a", "S3-b"}


def test_monitoring_calibrates_on_nulls_and_detects_sudden_drift(frame):
    monitoring = ca.run_monitoring(frame, windows=(10,))
    assert len(monitoring["null_dialogues"]["CQ"]) == 2 * 2 * 4
    cell = monitoring["cells"]["W10|CQ|cusum"]
    assert cell["value"] is not None and cell["loo_far"] <= 0.25
    summary = ca.summarize_strategies(monitoring, window=10)
    assert summary["distribution_based"]["CQ"]["detection_rate"] > 0.3
    h3 = ca.analyze_h3(monitoring, window=10)
    assert h3["S2"]["n_dialogues"] == 8 and "first_to_alarm_share" in h3["S2"]


def test_descriptive_outputs(frame):
    desc = ca.analyze_descriptive(ca.dialogue_deltas(frame))
    assert desc["h4"]["a"]["CQ"]["mean"] < 0
    assert desc["h5"]["support"] > 0


def test_stage_a_excludes_not_acquired_metrics(frame):
    available = {"a": ca.METRICS + ("CQ_ctrl",), "b": ("CQ", "LS", "ES")}
    staged = ca.restrict_to_available(frame, available)
    assert staged.loc[staged.model == "b", "CS"].isna().all()
    assert staged.loc[staged.model == "b", "CQ"].notna().any()
    monitoring = ca.run_monitoring(staged, windows=(10,), available=available)
    assert len(monitoring["null_dialogues"]["CS"]) == 8
    assert len(monitoring["null_dialogues"]["CQ"]) == 16
    out = monitoring["outcomes"]
    assert not ((out.metric == "CS") & (out.model == "b")).any()


def test_control_arm_scoring_and_screen():
    records = synthetic_records(models=("a",), per_scenario=6)
    control = ca.score_control_turns(records, coherence)
    assert control["CS"].isna().all() and control["CQ"].notna().all()
    monitoring = ca.run_monitoring(control, windows=(20,),
                                   available={"a": ca.CONTROL_METRICS})
    assert len(monitoring["null_dialogues"]["CQ"]) == 12
    screen = ca.control_screen(monitoring)
    assert set(screen["checks"]) == {"distribution_based", "proposal_mapping"}
    boot = ca.h2_paired_bootstrap(monitoring)
    assert "distribution_based-adwin_all" in boot["CQ"]
