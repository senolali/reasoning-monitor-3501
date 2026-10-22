"""Post-hoc, call-free sensitivity checks for the CDR-v1.3 validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame
from reasoning_monitor.evaluation.primary_dialogue_validation import (
    assess_primary_validation_model,
)


def _effect_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    primary = result["primary_checks"]
    longitudinal = result["longitudinal_checks"]
    return {
        "user_only_history_accuracy": result["post_baseline_accuracy"][
            "user_only_history"
        ],
        "mean_dialogue_history_load_gap": result[
            "mean_dialogue_history_load_gap"
        ],
        "history_load_sign_flip_p": result["history_load_sign_flip_p"],
        "positive_dialogue_gaps": result["positive_dialogue_gaps"],
        "primary_effect_checks_pass": all(
            primary[key]
            for key in (
                "mean_history_load_gap",
                "history_load_sign_flip",
                "positive_dialogue_coverage",
            )
        ),
        "mean_longitudinal_interaction": result[
            "mean_longitudinal_interaction"
        ],
        "longitudinal_sign_flip_p": result["longitudinal_sign_flip_p"],
        "positive_dialogue_interactions": result[
            "positive_dialogue_interactions"
        ],
        "longitudinal_checks_pass": all(longitudinal.values()),
    }


def _optimistic_ceiling_records(
    records: Sequence[dict[str, Any]], max_output_tokens: int
) -> list[dict[str, Any]]:
    adjusted = deepcopy(list(records))
    for record in adjusted:
        if (
            int(record.get("turn", 0)) > 1
            and record.get("condition") == "user_only_history"
            and int(record.get("n_tokens", 0)) >= max_output_tokens
        ):
            record["answer"] = str(record["gold_answer"])
    return adjusted


def diagnose_primary_validation_model(
    model: str,
    records: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Describe threshold granularity and bound capacity-censored cells."""
    observed = assess_primary_validation_model(model, records, gate)
    frame = records_to_turn_frame(records)
    post_summary = frame[
        (frame["turn"] > 1) & (frame["condition"] == "state_summary")
    ]
    summary_n = len(post_summary)
    summary_correct = int(post_summary["CQ_raw"].sum())
    threshold = float(gate["minimum_state_summary_accuracy"])
    required_correct = math.ceil((threshold * summary_n) - 1e-12)
    max_tokens = int(gate["max_output_tokens"])

    wrong_summary = post_summary[post_summary["CQ_raw"] == 0]
    fully_observed_wrong = wrong_summary[
        wrong_summary["ok"]
        & wrong_summary["strict_final_marker"]
        & wrong_summary["stored_answer_agreement"]
        & (wrong_summary["n_tokens"] < max_tokens)
    ]
    noncanonical = frame[
        (frame["turn"] > 1)
        & (~frame["strict_final_marker"])
        & (frame["CQ_raw"] == 1)
        & (frame["n_tokens"] < max_tokens)
    ]
    ceiling = frame[
        (frame["turn"] > 1)
        & (frame["condition"] == "user_only_history")
        & (frame["n_tokens"] >= max_tokens)
    ]

    optimistic_records = _optimistic_ceiling_records(records, max_tokens)
    optimistic = assess_primary_validation_model(model, optimistic_records, gate)
    observed_snapshot = _effect_snapshot(observed)
    optimistic_snapshot = _effect_snapshot(optimistic)
    return {
        "state_summary_threshold_resolution": {
            "post_baseline_cells": summary_n,
            "correct": summary_correct,
            "observed_accuracy": (
                summary_correct / summary_n if summary_n else 0.0
            ),
            "frozen_minimum_accuracy": threshold,
            "minimum_correct_required": required_correct,
            "correct_answer_shortfall": max(0, required_correct - summary_correct),
            "one_answer_resolution": 1 / summary_n if summary_n else 0.0,
        },
        "wrong_state_summary_cells": wrong_summary["qid"].tolist(),
        "fully_observed_wrong_state_summary_cells": fully_observed_wrong[
            "qid"
        ].tolist(),
        "wrong_state_summary_cells_are_parser_or_capacity_failures": bool(
            len(wrong_summary) > len(fully_observed_wrong)
        ),
        "noncanonical_but_correct_marker_cells": noncanonical["qid"].tolist(),
        "capacity_censored_user_only_cells": ceiling["qid"].tolist(),
        "capacity_scenarios": {
            "observed_censored_as_recorded": observed_snapshot,
            "optimistic_all_censored_answers_correct": optimistic_snapshot,
        },
        "primary_effect_robust_to_all_censored_answers_correct": bool(
            observed_snapshot["primary_effect_checks_pass"]
            and optimistic_snapshot["primary_effect_checks_pass"]
        ),
    }


def diagnose_primary_validation_sensitivity(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Run descriptive sensitivity checks without changing the frozen gate."""
    models = {
        model: diagnose_primary_validation_model(
            model, records_by_model.get(model, []), gate
        )
        for model in gate["expected_models"]
    }
    robust_models = [
        model
        for model, result in models.items()
        if result["primary_effect_robust_to_all_censored_answers_correct"]
    ]
    longitudinal_supported = any(
        result["capacity_scenarios"][
            "optimistic_all_censored_answers_correct"
        ]["longitudinal_checks_pass"]
        for result in models.values()
    )
    return {
        "analysis": "cdr_v1_3_posthoc_capacity_and_threshold_sensitivity",
        "no_model_calls": True,
        "post_hoc_descriptive": True,
        "claim_supporting_analysis": False,
        "frozen_gate_preserved_as_failed": True,
        "models": models,
        "primary_effect_robust_models": robust_models,
        "cross_model_primary_effect_robust": len(robust_models) == len(models),
        "longitudinal_signal_supported_under_optimistic_capacity_bound": (
            longitudinal_supported
        ),
        "decision": (
            "preserve_failed_gate_and_prepare_fresh_independent_replication_"
            "design_call_free"
        ),
        "additional_model_calls_authorized": False,
        "scientific_claim_authorized": False,
    }
