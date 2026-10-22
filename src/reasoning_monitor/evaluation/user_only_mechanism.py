"""Frozen assessment for the CDR-v1.2 user-only mechanism arm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def _one_sided_exact_binomial(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return min(
        1.0,
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials),
    )


def _paired(frame: pd.DataFrame, better: str, worse: str) -> dict[str, Any]:
    left = frame[frame["condition"] == better].set_index(["dialogue_id", "turn"])
    right = frame[frame["condition"] == worse].set_index(["dialogue_id", "turn"])
    joined = left[["CQ_raw"]].join(
        right[["CQ_raw"]], how="inner", lsuffix="_better", rsuffix="_worse"
    )
    better_only = int(
        ((joined["CQ_raw_better"] == 1) & (joined["CQ_raw_worse"] == 0)).sum()
    )
    worse_only = int(
        ((joined["CQ_raw_better"] == 0) & (joined["CQ_raw_worse"] == 1)).sum()
    )
    discordant = better_only + worse_only
    return {
        "better_condition": better,
        "worse_condition": worse,
        "paired_cells": len(joined),
        "better_only": better_only,
        "worse_only": worse_only,
        "accuracy_difference": float(
            joined["CQ_raw_better"].mean() - joined["CQ_raw_worse"].mean()
        ),
        "one_sided_exact_p": _one_sided_exact_binomial(better_only, discordant),
    }


def assess_user_only_model(
    model: str,
    user_records: Sequence[dict[str, Any]],
    existing_records: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    user_frame = records_to_turn_frame(user_records)
    existing_frame = records_to_turn_frame(existing_records)
    existing_frame = existing_frame[
        existing_frame["condition"].isin(["self_history", "state_summary"])
    ]
    frame = pd.concat([existing_frame, user_frame], ignore_index=True)
    post = frame[frame["turn"] > 1]
    accuracy = {
        condition: float(post.loc[post["condition"] == condition, "CQ_raw"].mean())
        for condition in ("self_history", "user_only_history", "state_summary")
    }
    state_vs_user = _paired(post, "state_summary", "user_only_history")
    user_vs_self = _paired(post, "user_only_history", "self_history")
    self_vs_user = _paired(post, "self_history", "user_only_history")

    first_failures = {}
    user_post = post[post["condition"] == "user_only_history"]
    for dialogue_id, group in user_post.groupby("dialogue_id", sort=True):
        failed = group[group["CQ_raw"] == 0]
        first_failures[str(dialogue_id)] = (
            int(failed["turn"].min()) if not failed.empty else None
        )
    n_failure_dialogues = sum(value is not None for value in first_failures.values())
    n_records = len(user_records)
    success_rate = (
        sum(bool(record.get("ok")) for record in user_records) / n_records
        if n_records
        else 0.0
    )
    marker_rate = (
        float(user_frame["strict_final_marker"].mean())
        if not user_frame.empty
        else 0.0
    )
    agreement_rate = (
        float(user_frame["stored_answer_agreement"].mean())
        if not user_frame.empty
        else 0.0
    )
    truncations = sum(
        str(record.get("finish_reason") or "").lower() == "length"
        for record in user_records
    )
    ceiling_hits = sum(
        int(record.get("n_tokens", 0)) >= int(gate["max_output_tokens"])
        for record in user_records
    )
    transport_valid = all(
        len(record.get("messages") or []) == 2
        and record.get("condition") == "user_only_history"
        and record.get("history_transport") == "packed_user_only_context"
        and (
            int(record.get("turn", 0)) == 1
            or "CALLER-MANAGED PRIOR USER HISTORY"
            in record["messages"][0]["content"]
        )
        and "[PRIOR ASSISTANT" not in record["messages"][0]["content"]
        for record in user_records
    )
    operational_checks = {
        "record_count": n_records == int(gate["expected_records_per_model"]),
        "success": success_rate >= float(gate["minimum_success_rate"]),
        "strict_final_marker": marker_rate
        >= float(gate["minimum_strict_final_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "no_token_ceiling_hit": ceiling_hits
        <= int(gate["maximum_token_ceiling_hits"]),
        "transport": transport_valid,
    }
    operational_checks = {
        name: bool(passed) for name, passed in operational_checks.items()
    }
    mechanism_checks = {
        "history_load_gap": state_vs_user["accuracy_difference"]
        >= float(gate["minimum_history_load_gap"]),
        "history_load_exact_p": state_vs_user["one_sided_exact_p"]
        <= float(gate["maximum_one_sided_exact_p"]),
        "failure_dialogue_coverage": n_failure_dialogues
        >= int(gate["minimum_user_only_failure_dialogues"]),
    }
    mechanism_checks = {
        name: bool(passed) for name, passed in mechanism_checks.items()
    }
    return {
        "model": model,
        "n_user_only_records": n_records,
        "success_rate": success_rate,
        "strict_final_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "token_ceiling_hits": ceiling_hits,
        "post_baseline_accuracy": accuracy,
        "state_summary_vs_user_only": state_vs_user,
        "user_only_vs_self": user_vs_self,
        "self_vs_user_only": self_vs_user,
        "user_only_first_failure_turn": first_failures,
        "user_only_failure_dialogues": n_failure_dialogues,
        "operational_checks": operational_checks,
        "mechanism_checks": mechanism_checks,
        "operational_pass": all(operational_checks.values()),
        "mechanism_pass": all(mechanism_checks.values()),
    }


def assess_user_only_mechanism(
    user_records_by_model: dict[str, Sequence[dict[str, Any]]],
    existing_records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    models = {
        model: assess_user_only_model(
            model,
            user_records_by_model.get(model, []),
            existing_records_by_model.get(model, []),
            gate,
        )
        for model in gate["expected_models"]
    }
    operational_pass = all(result["operational_pass"] for result in models.values())
    qualifying = [
        model
        for model, result in models.items()
        if result["operational_pass"] and result["mechanism_pass"]
    ]
    mechanism_pass = len(qualifying) >= int(
        gate["minimum_models_meeting_history_load_gate"]
    )
    return {
        "analysis": "cdr_v1_2_user_only_history_mechanism_gate",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "models": models,
        "qualifying_models": qualifying,
        "operational_pass": operational_pass,
        "mechanism_pass": mechanism_pass,
        "pass": operational_pass and mechanism_pass,
        "gate": gate,
    }

