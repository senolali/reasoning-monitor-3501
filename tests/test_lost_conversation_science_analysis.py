import pandas as pd

from reasoning_monitor.lost_conversation_science_analysis import (
    crl_accuracy_loss_sensitivity,
    format_valid_sensitivity,
    holm_adjust,
    paired_primary_endpoint,
)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_paired_endpoint_counts_invalid_as_failure():
    rows = []
    pairs = [(1.0, 1.0), (float("nan"), 1.0), (1.0, 0.0), (0.0, 1.0)]
    for index, (history, fresh) in enumerate(pairs):
        for condition, value in (("dialogue_history", history), ("fresh_prefix", fresh)):
            measured = pd.notna(value)
            rows.append({
                "model": "m", "task_id": str(index), "condition": condition,
                "turn": 2, "total_turns": 2, "CQ": value,
                "CQ_status": "measured" if measured else "invalid_final_answer",
            })
    _paired, report = paired_primary_endpoint(
        pd.DataFrame(rows), seed=1, repetitions=100)
    assert report["m"]["dialogue_success"] == 2
    assert report["m"]["fresh_success"] == 3
    assert report["m"]["fresh_only_successes"] == 2
    assert report["m"]["dialogue_only_successes"] == 1


def test_format_sensitivity_excludes_only_invalid_pairs():
    paired = pd.DataFrame({
        "model": ["m", "m"], "task_id": ["a", "b"],
        "dialogue_history": [0, 0], "fresh_prefix": [1, 1],
    })
    records = []
    for task_id in ("a", "b"):
        for condition in ("dialogue_history", "fresh_prefix"):
            records.append({
                "model": "m", "task_id": task_id, "condition": condition,
                "role": "primary", "turn": 2, "total_turns": 2,
                "contract_valid": not (task_id == "b" and condition == "dialogue_history"),
            })
    result = format_valid_sensitivity(paired, records)["m"]
    assert result["both_final_responses_semantically_valid"] == 1
    assert result["excluded_for_format_invalidity"] == 1
    assert result["paired_difference"] == 1.0


def test_accuracy_loss_target_is_independent_final_correctness():
    crl = pd.DataFrame({
        "model": ["m"] * 4,
        "task_id": ["a", "b", "c", "d"],
        "CRL": [0.8, 0.1, 0.7, 0.0],
        "prefinal_support_rate": [1.0] * 4,
        "final_positive_readiness_gap": [False] * 4,
    })
    paired = pd.DataFrame({
        "model": ["m"] * 4,
        "task_id": ["a", "b", "c", "d"],
        "dialogue_history": [0, 1, 0, 1],
        "fresh_prefix": [1, 1, 1, 1],
    })
    result = crl_accuracy_loss_sensitivity(
        crl, paired, seed=1, repetitions=20)
    assert result["m"]["target_cases"] == 2
    assert result["m"]["auroc"] == 1.0
    assert result["target_definition"] == "fresh_final_correct_and_dialogue_final_failed"
