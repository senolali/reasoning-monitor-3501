"""Frozen assessment for the fresh CDR-v1.1 packed-history pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def assess_packed_dialogue_model(
    model: str,
    records: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    frame = records_to_turn_frame(records)
    conditions = [str(item) for item in gate["expected_conditions"]]
    post = frame[frame["turn"] > 1]
    accuracy = {
        condition: float(
            post.loc[post["condition"] == condition, "CQ_raw"].mean()
        )
        for condition in conditions
    }
    all_turn_accuracy = {
        condition: float(
            frame.loc[frame["condition"] == condition, "CQ_raw"].mean()
        )
        for condition in conditions
    }
    context_penalty = accuracy["state_summary"] - accuracy["self_history"]
    error_propagation = accuracy["oracle_history"] - accuracy["self_history"]
    history_load = accuracy["state_summary"] - accuracy["oracle_history"]

    self_history = post[post["condition"] == "self_history"]
    first_failures = {}
    for dialogue_id, group in self_history.groupby("dialogue_id", sort=True):
        failures = group.loc[group["CQ_raw"] < 0.5, "turn"]
        first_failures[str(dialogue_id)] = (
            int(failures.min()) if not failures.empty else None
        )
    n_first_failures = sum(value is not None for value in first_failures.values())

    n_records = len(records)
    condition_counts = Counter(str(record.get("condition")) for record in records)
    success_rate = (
        sum(bool(record.get("ok")) for record in records) / n_records
        if n_records
        else 0.0
    )
    marker_rate = float(frame["strict_final_marker"].mean()) if not frame.empty else 0.0
    agreement_rate = (
        float(frame["stored_answer_agreement"].mean()) if not frame.empty else 0.0
    )
    truncations = sum(
        str(record.get("finish_reason") or "").lower() == "length"
        for record in records
    )
    max_tokens = int(gate["max_output_tokens"])
    ceiling_hits = sum(int(record.get("n_tokens", 0)) >= max_tokens for record in records)
    transport_valid = all(
        len(record.get("messages") or []) == 2
        and record.get("history_transport") == "packed_system_context"
        and (
            int(record.get("turn", 0)) == 1
            or record.get("condition") == "state_summary"
            or "CALLER-MANAGED PRIOR DIALOGUE TRANSCRIPT"
            in record["messages"][0]["content"]
        )
        for record in records
    )

    operational_checks = {
        "record_count": n_records == int(gate["expected_records_per_model"]),
        "condition_counts": all(
            condition_counts[condition]
            == len(gate["expected_dialogues"])
            * int(gate["expected_turns_per_dialogue"])
            for condition in conditions
        ),
        "dialogues": sorted(frame["dialogue_id"].unique())
        == sorted(gate["expected_dialogues"]),
        "success": success_rate >= float(gate["minimum_success_rate"]),
        "strict_final_marker": marker_rate
        >= float(gate["minimum_strict_final_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "no_token_ceiling_hit": ceiling_hits
        <= int(gate["maximum_token_ceiling_hits"]),
        "packed_transport": transport_valid,
    }
    operational_checks = {
        name: bool(passed) for name, passed in operational_checks.items()
    }
    phenomenon_checks = {
        "state_summary_solvable": accuracy["state_summary"]
        >= float(gate["minimum_state_summary_accuracy"]),
        "oracle_history_solvable": accuracy["oracle_history"]
        >= float(gate["minimum_oracle_history_accuracy"]),
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
        "truncations": truncations,
        "token_ceiling_hits": ceiling_hits,
        "accuracy_all_turns": all_turn_accuracy,
        "accuracy_post_baseline": accuracy,
        "context_penalty": context_penalty,
        "error_propagation_penalty": error_propagation,
        "history_load_penalty": history_load,
        "self_history_first_failure_turn": first_failures,
        "self_history_first_failures": n_first_failures,
        "operational_checks": operational_checks,
        "phenomenon_checks": phenomenon_checks,
        "operational_pass": all(operational_checks.values()),
        "phenomenon_pass": all(phenomenon_checks.values()),
    }


def assess_packed_dialogue_pilot(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    models = {
        model: assess_packed_dialogue_model(
            model, records_by_model.get(model, []), gate
        )
        for model in gate["expected_models"]
    }
    operational_pass = all(result["operational_pass"] for result in models.values())
    qualifying = [
        model
        for model, result in models.items()
        if result["operational_pass"] and result["phenomenon_pass"]
    ]
    phenomenon_pass = len(qualifying) >= int(
        gate["minimum_models_meeting_phenomenon_gate"]
    )
    return {
        "analysis": "cdr_v1_1_packed_context_effect_feasibility_gate",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "models": models,
        "qualifying_models": qualifying,
        "operational_pass": operational_pass,
        "phenomenon_pass": phenomenon_pass,
        "pass": operational_pass and phenomenon_pass,
        "gate": gate,
    }

