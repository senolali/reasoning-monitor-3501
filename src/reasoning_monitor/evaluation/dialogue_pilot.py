"""Frozen call-free assessment for the CDR-v1 context-effect pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def assess_dialogue_pilot_model(
    model: str,
    records: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    frame = records_to_turn_frame(records)
    expected_conditions = [str(value) for value in gate["expected_conditions"]]
    condition_counts = Counter(str(value) for value in frame.get("condition", []))
    accuracies = {
        condition: float(frame.loc[frame["condition"] == condition, "CQ_raw"].mean())
        if not frame.empty
        else 0.0
        for condition in expected_conditions
    }
    context_penalty = accuracies["state_summary"] - accuracies["self_history"]
    error_propagation = accuracies["oracle_history"] - accuracies["self_history"]
    history_load = accuracies["state_summary"] - accuracies["oracle_history"]

    self_history = frame[frame["condition"] == "self_history"]
    first_failures = {}
    for dialogue_id, group in self_history.groupby("dialogue_id", sort=True):
        failures = group.loc[group["CQ_raw"] < 0.5, "turn"]
        first_failures[str(dialogue_id)] = (
            int(failures.min()) if not failures.empty else None
        )
    n_first_failures = sum(value is not None for value in first_failures.values())

    n_records = len(records)
    primary_records = [record for record in records if record.get("role") == "primary"]
    success_rate = (
        sum(bool(record.get("ok")) for record in records) / n_records if n_records else 0.0
    )
    marker_rate = float(frame["strict_final_marker"].mean()) if not frame.empty else 0.0
    agreement_rate = (
        float(frame["stored_answer_agreement"].mean()) if not frame.empty else 0.0
    )
    truncations = sum(
        str(record.get("finish_reason") or "").lower() == "length"
        for record in records
    )
    operational_checks = {
        "record_count": n_records == int(gate["expected_total_records_per_model"]),
        "primary_record_count": len(primary_records)
        == int(gate["expected_primary_records_per_model"]),
        "dialogues": sorted(frame["dialogue_id"].unique())
        == sorted(gate["expected_dialogues"]),
        "conditions": sorted(frame["condition"].unique())
        == sorted(expected_conditions),
        "condition_counts": all(
            condition_counts[condition]
            == len(gate["expected_dialogues"])
            * int(gate["expected_turns_per_dialogue"])
            for condition in expected_conditions
        ),
        "success": success_rate >= float(gate["minimum_success_rate"]),
        "strict_final_marker": marker_rate
        >= float(gate["minimum_strict_final_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
    }
    operational_checks = {
        name: bool(passed) for name, passed in operational_checks.items()
    }
    phenomenon_checks = {
        "context_penalty": context_penalty
        >= float(gate["minimum_context_penalty_for_expansion"]),
        "self_history_first_failures": n_first_failures
        >= int(gate["minimum_self_history_first_failures_for_expansion"]),
    }
    phenomenon_checks = {
        name: bool(passed) for name, passed in phenomenon_checks.items()
    }
    return {
        "model": model,
        "n_records": n_records,
        "success_rate": success_rate,
        "strict_final_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": int(truncations),
        "accuracy_by_condition": accuracies,
        "context_penalty": context_penalty,
        "error_propagation_penalty": error_propagation,
        "history_load_penalty": history_load,
        "self_history_first_failure_turn": first_failures,
        "self_history_first_failures": int(n_first_failures),
        "operational_checks": operational_checks,
        "phenomenon_checks": phenomenon_checks,
        "operational_pass": all(operational_checks.values()),
        "phenomenon_pass": all(phenomenon_checks.values()),
    }


def assess_dialogue_pilot(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    expected_models = [str(model) for model in gate["expected_models"]]
    results = {
        model: assess_dialogue_pilot_model(model, records_by_model.get(model, []), gate)
        for model in expected_models
    }
    operational_pass = all(result["operational_pass"] for result in results.values())
    qualifying_models = [
        model
        for model, result in results.items()
        if result["operational_pass"] and result["phenomenon_pass"]
    ]
    return {
        "analysis": "cdr_v1_context_effect_feasibility_gate",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "models": results,
        "qualifying_models": qualifying_models,
        "operational_pass": operational_pass,
        "phenomenon_pass": len(qualifying_models)
        >= int(gate["minimum_models_meeting_phenomenon_gate"]),
        "pass": operational_pass
        and len(qualifying_models)
        >= int(gate["minimum_models_meeting_phenomenon_gate"]),
        "gate": gate,
    }
