"""Finite, call-free metric ablation after the RAT-v1 calibration screen."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (
    BASELINE_NUMERIC,
    CATEGORICAL,
    PRIMARY_METRICS,
    risk_set_summary,
)


def _pipeline(metrics: tuple[str, ...], c_value: float) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), [*BASELINE_NUMERIC, *metrics]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
    return Pipeline([("features", transformer), ("hazard", classifier)])


def _select_c(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
    c_grid: Iterable[float],
    n_splits: int,
    seed: int,
) -> float:
    y = frame["failure_next"].astype(int).to_numpy()
    groups = frame["family_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=min(int(n_splits), int(pd.Series(groups).nunique())),
        shuffle=True,
        random_state=seed,
    )
    candidates = []
    for c_value in (float(value) for value in c_grid):
        losses = []
        for train_index, test_index in splitter.split(frame, y, groups):
            if len(np.unique(y[train_index])) < 2:
                losses = []
                break
            fitted = _pipeline(metrics, c_value)
            fitted.fit(frame.iloc[train_index], y[train_index])
            probability = fitted.predict_proba(frame.iloc[test_index])[:, 1]
            losses.append(
                float(log_loss(y[test_index], probability, labels=[0, 1]))
            )
        if losses:
            candidates.append((float(np.mean(losses)), c_value))
    if not candidates:
        raise ValueError("no ridge candidate had estimable grouped folds")
    return float(min(candidates, key=lambda row: (row[0], row[1]))[1])


def _nested_validation(
    frame: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    c_grid: tuple[float, ...],
    outer_splits: int,
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    y = frame["failure_next"].astype(int).to_numpy()
    groups = frame["family_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=min(int(outer_splits), int(pd.Series(groups).nunique())),
        shuffle=True,
        random_state=seed,
    )
    folds = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, y, groups), start=1
    ):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        baseline_c = _select_c(train, (), c_grid, inner_splits, seed + fold)
        full_c = _select_c(train, metrics, c_grid, inner_splits, seed + fold)
        baseline = _pipeline((), baseline_c)
        full = _pipeline(metrics, full_c)
        baseline.fit(train, y[train_index])
        full.fit(train, y[train_index])
        baseline_loss = float(
            log_loss(
                y[test_index], baseline.predict_proba(test)[:, 1], labels=[0, 1]
            )
        )
        full_loss = float(
            log_loss(y[test_index], full.predict_proba(test)[:, 1], labels=[0, 1])
        )
        folds.append(
            {
                "fold": fold,
                "families": int(test["family_id"].nunique()),
                "baseline_c": baseline_c,
                "full_c": full_c,
                "baseline_log_loss": baseline_loss,
                "full_log_loss": full_loss,
                "improvement": baseline_loss - full_loss,
            }
        )
    return {
        "folds": folds,
        "mean_improvement": float(np.mean([row["improvement"] for row in folds])),
        "positive_folds": int(sum(row["improvement"] > 0 for row in folds)),
        "n_folds": int(len(folds)),
    }


def _candidate_label(metrics: tuple[str, ...]) -> str:
    return "+".join(metric.removesuffix("_raw") for metric in metrics)


def assess_metric_ablation(
    risk_set: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the prospectively enumerated metric subsets on calibration only."""
    expected_models = [str(model) for model in gate["expected_models"]]
    observed_models = sorted(risk_set["model"].astype(str).unique())
    if observed_models != sorted(expected_models):
        raise ValueError(f"observed models {observed_models} != {expected_models}")
    if set(risk_set["trajectory_split"].astype(str)) != {gate["expected_split"]}:
        raise ValueError("metric ablation may use the calibration split only")

    analysis = gate["analysis"]
    common = {
        "c_grid": tuple(float(value) for value in analysis["ridge_c_grid"]),
        "outer_splits": int(analysis["outer_splits"]),
        "inner_splits": int(analysis["inner_splits"]),
        "seed": int(analysis["seed"]),
    }
    decision = gate["decision"]
    candidates = {}
    for configured in gate["candidate_metric_subsets"]:
        metrics = tuple(str(metric) for metric in configured)
        if not metrics or not set(metrics).issubset(PRIMARY_METRICS):
            raise ValueError(f"invalid metric subset: {metrics}")
        label = _candidate_label(metrics)
        combined = _nested_validation(risk_set, metrics, **common)
        models = {
            model: _nested_validation(
                risk_set[risk_set["model"] == model].reset_index(drop=True),
                metrics,
                **common,
            )
            for model in expected_models
        }
        checks = {
            "combined_mean_improvement": combined["mean_improvement"]
            > float(decision["minimum_combined_mean_improvement"]),
            "combined_positive_folds": combined["positive_folds"]
            >= int(decision["minimum_combined_positive_folds"]),
            "all_models_mean_improvement": all(
                result["mean_improvement"]
                > float(decision["minimum_model_mean_improvement"])
                for result in models.values()
            ),
        }
        checks = {name: bool(passed) for name, passed in checks.items()}
        candidates[label] = {
            "metrics": list(metrics),
            "combined": combined,
            "models": models,
            "checks": checks,
            "qualifies": all(checks.values()),
        }

    qualified = [
        label for label, result in candidates.items() if result["qualifies"]
    ]
    selected = None
    if qualified:
        selected = min(
            qualified,
            key=lambda label: (
                len(candidates[label]["metrics"]),
                -candidates[label]["combined"]["mean_improvement"],
                label,
            ),
        )
    return {
        "analysis": "rat_v1_post_calibration_metric_ablation",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "risk_set": risk_set_summary(risk_set),
        "candidates": candidates,
        "qualified_candidates": qualified,
        "selected_candidate": selected,
        "screen_pass": selected is not None,
        "decision_scope": gate["decision_scope"],
    }
