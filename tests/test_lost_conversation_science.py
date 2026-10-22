import math
from pathlib import Path

from reasoning_monitor.lost_conversation_science import (
    EXPECTED_CALLS,
    cumulative_resolution_lag,
    load_science_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/lost_conversation_v2_science_asu.yaml"


def test_science_plan_is_disjoint_locked_and_budgeted():
    config, manifest, _tasks, task_ids, report = load_science_plan(ROOT, CONFIG)
    assert len(task_ids) == 48
    assert not set(task_ids) & set(manifest["integration_task_ids"])
    assert report["expected_calls"] == EXPECTED_CALLS
    assert report["execution_ready"] is True
    assert report["execution_authorized"] is False
    assert config["authorization_recorded_at"] == "2026-10-02"
    assert config["outcome"] == "completed_and_locked_2026-10-02"
    assert config["completed_records"] == EXPECTED_CALLS
    assert config["analysis"]["auxiliary_metric"]["threshold"] is None
    assert report["checks"]["authorization_state_consistent"] is True


def test_crl_excludes_invalid_pairs_without_imputation():
    rows = []
    kinds = {
        1: ("need_more_information", "answer_attempt"),
        2: ("invalid", "answer_attempt"),
        3: ("answer_attempt", "answer_attempt"),
        4: ("need_more_information", "answer_attempt"),
    }
    for turn, (history_kind, fresh_kind) in kinds.items():
        for condition, kind in (("dialogue_history", history_kind),
                                ("fresh_prefix", fresh_kind)):
            rows.append({
                "model": "m", "task_id": "t", "turn": turn, "total_turns": 4,
                "condition": condition, "role": "primary", "response_kind": kind,
            })
    trajectory, summary = cumulative_resolution_lag(rows)
    row = summary.iloc[0]
    assert row.comparable_prefinal_turns == 2
    assert row.expected_prefinal_turns == 3
    assert math.isclose(row.prefinal_support_rate, 2 / 3)
    assert row.CRL == 0.5
    assert row.longest_positive_gap_run == 1
    assert bool(row.final_positive_readiness_gap) is True
    assert trajectory.loc[trajectory.turn == 2, "readiness_gap"].isna().all()


def test_crl_has_no_retrospective_threshold_or_support_filter():
    _config, _manifest, _tasks, _ids, report = load_science_plan(ROOT, CONFIG)
    auxiliary = report["analysis"]["auxiliary_metric"]
    assert auxiliary["threshold"] is None
    assert auxiliary["minimum_support_threshold"] is None
    assert auxiliary["report_support_count_and_rate"] is True
