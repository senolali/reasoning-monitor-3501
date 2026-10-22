"""Call-free exploratory diagnosis of the failed CDR-v1.1 pilot gate."""

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


def _paired_contrast(frame, better: str, worse: str) -> dict[str, Any]:
    left = frame[frame["condition"] == better].set_index(["dialogue_id", "turn"])
    right = frame[frame["condition"] == worse].set_index(["dialogue_id", "turn"])
    joined = left[["CQ_raw"]].join(
        right[["CQ_raw"]], how="inner", lsuffix="_better", rsuffix="_worse"
    )
    improved = int(
        ((joined["CQ_raw_better"] == 1) & (joined["CQ_raw_worse"] == 0)).sum()
    )
    worsened = int(
        ((joined["CQ_raw_better"] == 0) & (joined["CQ_raw_worse"] == 1)).sum()
    )
    discordant = improved + worsened
    return {
        "better_condition": better,
        "worse_condition": worse,
        "paired_cells": len(joined),
        "better_only": improved,
        "worse_only": worsened,
        "accuracy_difference": float(
            joined["CQ_raw_better"].mean() - joined["CQ_raw_worse"].mean()
        ),
        "one_sided_exact_p": _one_sided_exact_binomial(improved, discordant),
    }


def diagnose_packed_dialogue_model(
    model: str, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    frame = records_to_turn_frame(records)
    post = frame[frame["turn"] > 1]
    accuracy = {
        condition: float(post.loc[post["condition"] == condition, "CQ_raw"].mean())
        for condition in ("self_history", "oracle_history", "state_summary")
    }
    context_chars = {
        condition: float(
            post.loc[post["condition"] == condition, "context_chars"].mean()
        )
        for condition in ("self_history", "oracle_history", "state_summary")
    }
    failures = {}
    persistent = {}
    self_rows = post[post["condition"] == "self_history"]
    for dialogue_id, group in self_rows.groupby("dialogue_id", sort=True):
        group = group.sort_values("turn")
        failed = group[group["CQ_raw"] == 0]
        first = int(failed["turn"].min()) if not failed.empty else None
        failures[str(dialogue_id)] = first
        if first is None:
            persistent[str(dialogue_id)] = None
        else:
            tail = group[group["turn"] >= first]
            persistent[str(dialogue_id)] = {
                "failures": int((tail["CQ_raw"] == 0).sum()),
                "tail_turns": len(tail),
                "failure_rate": float((tail["CQ_raw"] == 0).mean()),
            }
    return {
        "model": model,
        "post_baseline_accuracy": accuracy,
        "mean_context_chars": context_chars,
        "state_summary_vs_self": _paired_contrast(
            post, "state_summary", "self_history"
        ),
        "state_summary_vs_oracle": _paired_contrast(
            post, "state_summary", "oracle_history"
        ),
        "self_vs_oracle": _paired_contrast(post, "self_history", "oracle_history"),
        "self_history_first_failure_turn": failures,
        "self_history_persistence_after_first_failure": persistent,
        "oracle_less_accurate_than_self": accuracy["oracle_history"]
        < accuracy["self_history"],
        "oracle_context_shorter_than_self": context_chars["oracle_history"]
        < context_chars["self_history"],
    }


def diagnose_packed_dialogue_pilot(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    models = {
        model: diagnose_packed_dialogue_model(model, records)
        for model, records in records_by_model.items()
    }
    failure_sets = []
    for records in records_by_model.values():
        frame = records_to_turn_frame(records)
        failed = frame[
            (frame["condition"] == "self_history")
            & (frame["turn"] > 1)
            & (frame["CQ_raw"] == 0)
        ]
        failure_sets.append(
            {(str(row.dialogue_id), int(row.turn)) for row in failed.itertuples()}
        )
    shared_failures = sorted(set.intersection(*failure_sets)) if failure_sets else []
    exploratory_signal = bool(models) and all(
        result["state_summary_vs_self"]["accuracy_difference"] >= 0.10
        and result["state_summary_vs_self"]["one_sided_exact_p"] < 0.05
        for result in models.values()
    )
    oracle_asymmetry = bool(models) and all(
        result["oracle_less_accurate_than_self"]
        and result["oracle_context_shorter_than_self"]
        for result in models.values()
    )
    return {
        "analysis": "cdr_v1_1_call_free_exploratory_diagnostic",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "preregistered_gate_remains_failed": True,
        "models": models,
        "shared_self_history_failure_cells": [
            {"dialogue_id": dialogue_id, "turn": turn}
            for dialogue_id, turn in shared_failures
        ],
        "exploratory_cross_model_context_signal": exploratory_signal,
        "oracle_control_information_asymmetry": oracle_asymmetry,
        "decision": (
            "freeze_format_matched_successor_design"
            if exploratory_signal and oracle_asymmetry
            else "stop_for_manual_review"
        ),
        "scientific_claim_authorized": False,
        "paid_calls_authorized": False,
    }

