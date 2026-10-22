"""Post-hoc capacity-censoring diagnostic for the CDR-v1.2 mechanism arm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def _one_sided_exact_binomial(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return min(
        1.0,
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials),
    )


def diagnose_user_only_capacity_model(
    user_records: Sequence[dict[str, Any]],
    existing_records: Sequence[dict[str, Any]],
    *,
    max_output_tokens: int,
    minimum_history_load_gap: float,
    maximum_one_sided_exact_p: float,
    minimum_failure_dialogues: int,
) -> dict[str, Any]:
    """Bound the primary mechanism contrast under capacity-censored outputs."""
    user = records_to_turn_frame(user_records)
    state = records_to_turn_frame(existing_records)
    user = user[user["turn"] > 1].copy()
    state = state[
        (state["turn"] > 1) & (state["condition"] == "state_summary")
    ].copy()
    user["capacity_censored"] = user["n_tokens"] >= int(max_output_tokens)

    joined = state[
        ["dialogue_id", "turn", "CQ_raw"]
    ].merge(
        user[
            [
                "dialogue_id",
                "turn",
                "CQ_raw",
                "strict_final_marker",
                "capacity_censored",
            ]
        ],
        on=["dialogue_id", "turn"],
        how="inner",
        suffixes=("_state", "_user"),
        validate="one_to_one",
    )
    uncensored = joined[~joined["capacity_censored"]]
    n_total = len(joined)
    n_censored = int(joined["capacity_censored"].sum())
    marker_failures = int((~joined["strict_final_marker"]).sum())
    overlap = int(
        (joined["capacity_censored"] & ~joined["strict_final_marker"]).sum()
    )
    correct_total = int(joined["CQ_raw_user"].sum())
    censored_incorrect = int(
        (
            joined["capacity_censored"]
            & (joined["CQ_raw_user"] == 0)
        ).sum()
    )
    lower_accuracy = correct_total / n_total if n_total else 0.0
    upper_accuracy = (
        (correct_total + censored_incorrect) / n_total if n_total else 0.0
    )
    state_accuracy = float(joined["CQ_raw_state"].mean()) if n_total else 0.0

    state_only = int(
        (
            (uncensored["CQ_raw_state"] == 1)
            & (uncensored["CQ_raw_user"] == 0)
        ).sum()
    )
    user_only = int(
        (
            (uncensored["CQ_raw_state"] == 0)
            & (uncensored["CQ_raw_user"] == 1)
        ).sum()
    )
    discordant = state_only + user_only
    exact_p = _one_sided_exact_binomial(state_only, discordant)
    observed_failures = uncensored[uncensored["CQ_raw_user"] == 0]
    observed_failure_dialogues = int(observed_failures["dialogue_id"].nunique())
    minimum_gap = state_accuracy - upper_accuracy
    maximum_gap = state_accuracy - lower_accuracy
    robust = bool(
        minimum_gap >= float(minimum_history_load_gap)
        and exact_p <= float(maximum_one_sided_exact_p)
        and observed_failure_dialogues >= int(minimum_failure_dialogues)
    )

    n_uncensored = len(uncensored)
    uncensored_correct = int(uncensored["CQ_raw_user"].sum())
    return {
        "post_baseline_cells": n_total,
        "capacity_censored_cells": n_censored,
        "strict_marker_failures": marker_failures,
        "censored_marker_failure_overlap": overlap,
        "all_marker_failures_capacity_censored": marker_failures == overlap,
        "uncensored_complete_cases": {
            "n": n_uncensored,
            "correct": uncensored_correct,
            "accuracy": uncensored_correct / n_uncensored if n_uncensored else 0.0,
            "state_only_correct": state_only,
            "user_only_correct": user_only,
            "one_sided_exact_p": exact_p,
            "failure_dialogues": observed_failure_dialogues,
        },
        "user_only_accuracy_bounds": [lower_accuracy, upper_accuracy],
        "state_summary_accuracy": state_accuracy,
        "state_summary_advantage_bounds": [minimum_gap, maximum_gap],
        "primary_signal_robust_to_all_censored_answers_being_correct": robust,
    }


def diagnose_user_only_capacity(
    user_records_by_model: dict[str, Sequence[dict[str, Any]]],
    existing_records_by_model: dict[str, Sequence[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    models = {
        model: diagnose_user_only_capacity_model(
            user_records_by_model[model],
            existing_records_by_model[model],
            max_output_tokens=int(config["max_output_tokens"]),
            minimum_history_load_gap=float(config["minimum_history_load_gap"]),
            maximum_one_sided_exact_p=float(
                config["maximum_one_sided_exact_p"]
            ),
            minimum_failure_dialogues=int(config["minimum_failure_dialogues"]),
        )
        for model in config["expected_models"]
    }
    censored_cells = sum(
        result["capacity_censored_cells"] for result in models.values()
    )
    robust_models = [
        model
        for model, result in models.items()
        if result["primary_signal_robust_to_all_censored_answers_being_correct"]
    ]
    return {
        "analysis": "cdr_v1_2_user_only_capacity_sensitivity_diagnostic",
        "no_model_calls": True,
        "post_hoc_descriptive": True,
        "claim_supporting_analysis": False,
        "frozen_gate_preserved_as_failed": True,
        "models": models,
        "capacity_censored_cells": censored_cells,
        "robust_models": robust_models,
        "all_model_primary_signal_robust": len(robust_models) == len(models),
        "decision": "freeze_exact_message_capacity_retry_for_censored_cells",
        "scientific_claim_authorized": False,
    }
