"""Cluster-aware frozen analysis for the CDR-v1.3 primary validation."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pandas as pd

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """One-sided exact sign-flip p-value for paired cluster differences."""
    nonzero = [float(value) for value in values if abs(float(value)) > 1e-12]
    if not nonzero:
        return 1.0
    observed = sum(nonzero) / len(nonzero)
    permuted = (
        sum(sign * value for sign, value in zip(signs, nonzero, strict=True))
        / len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero))
    )
    extreme = sum(value >= observed - 1e-12 for value in permuted)
    return extreme / (2 ** len(nonzero))


def _paired_cluster_statistics(
    frame: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    post = frame[frame["turn"] > 1]
    pivot = post.pivot(
        index=["dialogue_id", "turn"],
        columns="condition",
        values="CQ_raw",
    ).reset_index()
    pivot["history_load_gap"] = (
        pivot["state_summary"] - pivot["user_only_history"]
    )
    early = pivot[
        pivot["turn"].between(
            int(gate["early_turn_start"]), int(gate["early_turn_end"])
        )
    ]
    late = pivot[
        pivot["turn"].between(
            int(gate["late_turn_start"]), int(gate["late_turn_end"])
        )
    ]
    overall_by_dialogue = pivot.groupby("dialogue_id")["history_load_gap"].mean()
    early_by_dialogue = early.groupby("dialogue_id")["history_load_gap"].mean()
    late_by_dialogue = late.groupby("dialogue_id")["history_load_gap"].mean()
    interaction = late_by_dialogue - early_by_dialogue
    return {
        "paired_cells": len(pivot),
        "dialogue_history_load_gaps": {
            str(key): float(value) for key, value in overall_by_dialogue.items()
        },
        "mean_dialogue_history_load_gap": float(overall_by_dialogue.mean()),
        "history_load_sign_flip_p": exact_sign_flip_p(overall_by_dialogue.tolist()),
        "positive_dialogue_gaps": int((overall_by_dialogue > 0).sum()),
        "early_mean_history_load_gap": float(early_by_dialogue.mean()),
        "late_mean_history_load_gap": float(late_by_dialogue.mean()),
        "dialogue_longitudinal_interactions": {
            str(key): float(value) for key, value in interaction.items()
        },
        "mean_longitudinal_interaction": float(interaction.mean()),
        "longitudinal_sign_flip_p": exact_sign_flip_p(interaction.tolist()),
        "positive_dialogue_interactions": int((interaction > 0).sum()),
    }


def assess_primary_validation_model(
    model: str,
    records: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    frame = records_to_turn_frame(records)
    conditions = [str(item) for item in gate["expected_conditions"]]
    post = frame[frame["turn"] > 1]
    accuracy = {
        condition: float(post.loc[post["condition"] == condition, "CQ_raw"].mean())
        for condition in conditions
    }
    stats = _paired_cluster_statistics(frame, gate)
    n_records = len(records)
    counts = Counter(str(record.get("condition")) for record in records)
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
    ceiling_hits = sum(
        int(record.get("n_tokens", 0)) >= int(gate["max_output_tokens"])
        for record in records
    )
    transport_valid = all(
        len(record.get("messages") or []) == 2
        and (
            (
                record.get("condition") == "user_only_history"
                and record.get("history_transport") == "packed_user_only_context"
                and (
                    int(record.get("turn", 0)) == 1
                    or "CALLER-MANAGED PRIOR USER HISTORY"
                    in record["messages"][0]["content"]
                )
                and "[PRIOR ASSISTANT" not in record["messages"][0]["content"]
            )
            or (
                record.get("condition") == "state_summary"
                and record.get("history_transport") == "verified_state_summary"
                and "VERIFIED STATE BEFORE THIS TURN"
                in record["messages"][0]["content"]
            )
        )
        for record in records
    )
    operational_checks = {
        "record_count": n_records == int(gate["expected_records_per_model"]),
        "condition_counts": all(
            counts[condition]
            == len(gate["expected_dialogues"])
            * int(gate["expected_turns_per_dialogue"])
            for condition in conditions
        ),
        "dialogues": sorted(frame["dialogue_id"].unique())
        == sorted(gate["expected_dialogues"]),
        "paired_cells": stats["paired_cells"]
        == int(gate["expected_post_baseline_pairs_per_model"]),
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
    primary_checks = {
        "state_summary_accuracy": accuracy["state_summary"]
        >= float(gate["minimum_state_summary_accuracy"]),
        "mean_history_load_gap": stats["mean_dialogue_history_load_gap"]
        >= float(gate["minimum_mean_dialogue_history_load_gap"]),
        "history_load_sign_flip": stats["history_load_sign_flip_p"]
        <= float(gate["maximum_one_sided_sign_flip_p"]),
        "positive_dialogue_coverage": stats["positive_dialogue_gaps"]
        >= int(gate["minimum_positive_dialogue_gaps"]),
    }
    longitudinal_checks = {
        "mean_interaction": stats["mean_longitudinal_interaction"]
        >= float(gate["minimum_mean_longitudinal_interaction"]),
        "interaction_sign_flip": stats["longitudinal_sign_flip_p"]
        <= float(gate["maximum_longitudinal_sign_flip_p"]),
        "positive_dialogue_coverage": stats["positive_dialogue_interactions"]
        >= int(gate["minimum_positive_dialogue_interactions"]),
    }
    operational_checks = {key: bool(value) for key, value in operational_checks.items()}
    primary_checks = {key: bool(value) for key, value in primary_checks.items()}
    longitudinal_checks = {
        key: bool(value) for key, value in longitudinal_checks.items()
    }
    return {
        "model": model,
        "n_records": n_records,
        "success_rate": success_rate,
        "strict_final_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "token_ceiling_hits": ceiling_hits,
        "post_baseline_accuracy": accuracy,
        **stats,
        "operational_checks": operational_checks,
        "primary_checks": primary_checks,
        "longitudinal_checks": longitudinal_checks,
        "operational_pass": all(operational_checks.values()),
        "primary_pass": all(primary_checks.values()),
        "longitudinal_pass": all(longitudinal_checks.values()),
    }


def assess_primary_validation(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    models = {
        model: assess_primary_validation_model(
            model, records_by_model.get(model, []), gate
        )
        for model in gate["expected_models"]
    }
    operational_pass = all(result["operational_pass"] for result in models.values())
    primary_qualifying = [
        model
        for model, result in models.items()
        if result["operational_pass"] and result["primary_pass"]
    ]
    longitudinal_qualifying = [
        model
        for model, result in models.items()
        if result["operational_pass"] and result["longitudinal_pass"]
    ]
    primary_pass = len(primary_qualifying) >= int(
        gate["minimum_models_meeting_primary_gate"]
    )
    nonnegative_all = all(
        result["mean_longitudinal_interaction"] >= 0 for result in models.values()
    )
    longitudinal_pass = len(longitudinal_qualifying) >= int(
        gate["minimum_models_meeting_longitudinal_gate"]
    ) and (
        not bool(gate["require_nonnegative_interaction_all_models"])
        or nonnegative_all
    )
    return {
        "analysis": "cdr_v1_3_cluster_aware_primary_validation",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "models": models,
        "primary_qualifying_models": primary_qualifying,
        "longitudinal_qualifying_models": longitudinal_qualifying,
        "operational_pass": operational_pass,
        "primary_pass": primary_pass,
        "longitudinal_pass": longitudinal_pass,
        "pass": operational_pass and primary_pass and longitudinal_pass,
        "gate": gate,
    }
