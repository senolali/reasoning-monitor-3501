"""Call-free calibration signal diagnostic for RAT-v1."""

from __future__ import annotations

from typing import Any

import pandas as pd

from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (
    nested_grouped_validation,
    risk_set_summary,
)


def _validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    folds = result["folds"]
    return {
        **result,
        "positive_folds": int(sum(row["improvement"] > 0 for row in folds)),
        "n_folds": int(len(folds)),
    }


def assess_calibration_signal(
    risk_set: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline and full hazards without opening non-calibration splits."""
    expected_models = [str(model) for model in gate["expected_models"]]
    observed_models = sorted(risk_set["model"].astype(str).unique())
    if observed_models != sorted(expected_models):
        raise ValueError(
            f"signal diagnostic models are {observed_models}, expected "
            f"{sorted(expected_models)}"
        )
    observed_splits = set(risk_set["trajectory_split"].astype(str))
    if observed_splits != {str(gate["expected_split"])}:
        raise ValueError(
            f"signal diagnostic split is {sorted(observed_splits)}, expected "
            f"{gate['expected_split']}"
        )

    analysis = gate["analysis"]
    common = {
        "c_grid": tuple(float(value) for value in analysis["ridge_c_grid"]),
        "outer_splits": int(analysis["outer_splits"]),
        "inner_splits": int(analysis["inner_splits"]),
        "seed": int(analysis["seed"]),
    }
    combined = _validation_summary(nested_grouped_validation(risk_set, **common))
    model_results = {
        model: _validation_summary(
            nested_grouped_validation(
                risk_set[risk_set["model"] == model].reset_index(drop=True),
                **common,
            )
        )
        for model in expected_models
    }

    decision = gate["decision"]
    checks = {
        "combined_mean_improvement": combined["mean_improvement"]
        > float(decision["minimum_combined_mean_improvement"]),
        "combined_positive_folds": combined["positive_folds"]
        >= int(decision["minimum_combined_positive_folds"]),
        "all_models_mean_improvement": all(
            result["mean_improvement"]
            > float(decision["minimum_model_mean_improvement"])
            for result in model_results.values()
        ),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    return {
        "analysis": "rat_v1_post_calibration_signal_diagnostic",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "risk_set": risk_set_summary(risk_set),
        "combined": combined,
        "models": model_results,
        "checks": checks,
        "screen_pass": all(checks.values()),
        "decision_scope": gate["decision_scope"],
    }
