"""Frozen risk-set and holdout analysis for the independent RAT-v1 protocol."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRIMARY_METRICS = ("RS_raw", "LS_raw", "ES_raw")
BASELINE_NUMERIC = ("trajectory_level",)
CATEGORICAL = ("task_template", "model")
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True, slots=True)
class FittedHazardComparison:
    baseline: Pipeline
    full: Pipeline
    baseline_c: float
    full_c: float
    tuning: dict[str, Any]


def build_risk_set(per_query: pd.DataFrame) -> pd.DataFrame:
    """Create level-t predictors of first CQ failure at level t+1.

    A model-family contributes a row only while CQ has remained correct through
    the current level. Consequently each trajectory contributes at most one
    event and observations after its first failure are excluded by design.
    """
    required = {
        "model",
        "family_id",
        "trajectory_level",
        "trajectory_split",
        "domain",
        "qid",
        "CQ_raw",
        "RS_raw",
        "LS_raw",
        "ES_raw",
        "call_failed",
    }
    missing = sorted(required - set(per_query.columns))
    if missing:
        raise ValueError(f"per-query data missing RAT-v1 columns: {missing}")
    duplicate = per_query.duplicated(["model", "family_id", "trajectory_level"])
    if duplicate.any():
        raise ValueError("duplicate model/family/level rows in RAT-v1 per-query data")

    rows: list[dict[str, Any]] = []
    grouped = per_query.groupby(["model", "family_id"], sort=True, dropna=False)
    for (model, family_id), family in grouped:
        ordered = family.sort_values("trajectory_level")
        levels = ordered["trajectory_level"].astype(int).tolist()
        if levels != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"{model}/{family_id} has levels {levels}, expected 1..6")
        if ordered["call_failed"].astype(bool).any():
            raise ValueError(f"{model}/{family_id} contains failed calls")
        if ordered["CQ_raw"].isna().any():
            raise ValueError(f"{model}/{family_id} contains missing CQ")
        if ordered["trajectory_split"].nunique(dropna=False) != 1:
            raise ValueError(f"{model}/{family_id} crosses trajectory splits")
        if ordered["domain"].nunique(dropna=False) != 1:
            raise ValueError(f"{model}/{family_id} changes task template")

        records = ordered.to_dict("records")
        for current, following in zip(records, records[1:], strict=False):
            if float(current["CQ_raw"]) < 0.5:
                break
            predictor_values = [float(current[name]) for name in PRIMARY_METRICS]
            if not np.isfinite(predictor_values).all():
                raise ValueError(
                    f"{model}/{family_id}/L{current['trajectory_level']} has "
                    "non-finite primary predictors while at risk"
                )
            event = int(float(following["CQ_raw"]) < 0.5)
            rows.append(
                {
                    "model": str(model),
                    "family_id": str(family_id),
                    "trajectory_split": str(current["trajectory_split"]),
                    "task_template": str(current["domain"]),
                    "trajectory_level": int(current["trajectory_level"]),
                    "next_level": int(following["trajectory_level"]),
                    "qid_t": str(current["qid"]),
                    "qid_next": str(following["qid"]),
                    "failure_next": event,
                    **{
                        name: value
                        for name, value in zip(
                            PRIMARY_METRICS, predictor_values, strict=True
                        )
                    },
                }
            )
            if event:
                break
    return pd.DataFrame(rows)


def risk_set_summary(risk_set: pd.DataFrame) -> dict[str, Any]:
    if risk_set.empty:
        return {
            "rows": 0,
            "families_entered": 0,
            "events": 0,
            "censored_families": 0,
            "events_by_model": {},
        }
    family_events = risk_set.groupby(["model", "family_id"])["failure_next"].max()
    return {
        "rows": int(len(risk_set)),
        "families_entered": int(len(family_events)),
        "events": int(risk_set["failure_next"].sum()),
        "censored_families": int((family_events == 0).sum()),
        "events_by_model": {
            str(model): int(group["failure_next"].sum())
            for model, group in risk_set.groupby("model", sort=True)
        },
    }


def _feature_columns(full: bool) -> tuple[list[str], list[str]]:
    numeric = list(BASELINE_NUMERIC)
    if full:
        numeric.extend(PRIMARY_METRICS)
    return numeric, list(CATEGORICAL)


def _pipeline(full: bool, c_value: float) -> Pipeline:
    numeric, categorical = _feature_columns(full)
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
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


def _require_estimable(frame: pd.DataFrame) -> None:
    required = {"family_id", "failure_next", *CATEGORICAL, *BASELINE_NUMERIC, *PRIMARY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"risk set missing analysis columns: {missing}")
    outcomes = frame["failure_next"].astype(int)
    if set(outcomes.unique()) != {0, 1}:
        raise ValueError("risk set must contain both failures and non-failures")
    if frame[list(PRIMARY_METRICS)].isna().any().any():
        raise ValueError("risk set contains missing primary predictors")
    if frame["family_id"].nunique() < 4:
        raise ValueError("risk set requires at least four independent families")


def select_ridge_c(
    frame: pd.DataFrame,
    *,
    full: bool,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    n_splits: int = 4,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Select ridge strength by family-grouped stratified cross-validation."""
    _require_estimable(frame)
    y = frame["failure_next"].astype(int).to_numpy()
    groups = frame["family_id"].astype(str).to_numpy()
    split_count = min(int(n_splits), int(pd.Series(groups).nunique()))
    if split_count < 2:
        raise ValueError("ridge selection requires at least two family folds")
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    candidates: list[dict[str, Any]] = []
    for c_value in [float(value) for value in c_grid]:
        fold_losses: list[float] = []
        valid = True
        for train_index, test_index in splitter.split(frame, y, groups):
            y_train = y[train_index]
            if len(np.unique(y_train)) < 2:
                valid = False
                break
            model = _pipeline(full, c_value)
            model.fit(frame.iloc[train_index], y_train)
            probability = model.predict_proba(frame.iloc[test_index])[:, 1]
            fold_losses.append(
                float(log_loss(y[test_index], probability, labels=[0, 1]))
            )
        candidates.append(
            {
                "c": c_value,
                "fold_log_loss": fold_losses,
                "mean_log_loss": float(np.mean(fold_losses)) if valid else float("inf"),
                "valid": valid,
            }
        )
    valid_candidates = [row for row in candidates if row["valid"]]
    if not valid_candidates:
        raise ValueError("no ridge candidate had estimable grouped folds")
    best = min(valid_candidates, key=lambda row: (row["mean_log_loss"], row["c"]))
    return {"best_c": best["c"], "candidates": candidates, "n_splits": split_count}


def fit_hazard_comparison(
    training: pd.DataFrame,
    *,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    n_splits: int = 4,
    seed: int = 20260926,
) -> FittedHazardComparison:
    """Tune and fit the frozen baseline and full hazard models."""
    baseline_tuning = select_ridge_c(
        training, full=False, c_grid=c_grid, n_splits=n_splits, seed=seed
    )
    full_tuning = select_ridge_c(
        training, full=True, c_grid=c_grid, n_splits=n_splits, seed=seed
    )
    y = training["failure_next"].astype(int)
    baseline = _pipeline(False, baseline_tuning["best_c"])
    full = _pipeline(True, full_tuning["best_c"])
    baseline.fit(training, y)
    full.fit(training, y)
    return FittedHazardComparison(
        baseline=baseline,
        full=full,
        baseline_c=float(baseline_tuning["best_c"]),
        full_c=float(full_tuning["best_c"]),
        tuning={"baseline": baseline_tuning, "full": full_tuning},
    )


def nested_grouped_validation(
    frame: pd.DataFrame,
    *,
    c_grid: Iterable[float] = DEFAULT_C_GRID,
    outer_splits: int = 4,
    inner_splits: int = 3,
    seed: int = 20260926,
) -> dict[str, Any]:
    """Estimate pre-holdout gain with family-grouped nested validation."""
    _require_estimable(frame)
    y = frame["failure_next"].astype(int).to_numpy()
    groups = frame["family_id"].astype(str).to_numpy()
    split_count = min(int(outer_splits), int(pd.Series(groups).nunique()))
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    folds: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, y, groups), start=1
    ):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        fitted = fit_hazard_comparison(
            train,
            c_grid=c_grid,
            n_splits=inner_splits,
            seed=seed + fold,
        )
        baseline_probability = fitted.baseline.predict_proba(test)[:, 1]
        full_probability = fitted.full.predict_proba(test)[:, 1]
        baseline_loss = float(
            log_loss(y[test_index], baseline_probability, labels=[0, 1])
        )
        full_loss = float(log_loss(y[test_index], full_probability, labels=[0, 1]))
        folds.append(
            {
                "fold": fold,
                "families": int(test["family_id"].nunique()),
                "baseline_c": fitted.baseline_c,
                "full_c": fitted.full_c,
                "baseline_log_loss": baseline_loss,
                "full_log_loss": full_loss,
                "improvement": baseline_loss - full_loss,
            }
        )
    return {
        "folds": folds,
        "mean_improvement": float(np.mean([row["improvement"] for row in folds])),
    }


def _cluster_bootstrap_improvement(
    frame: pd.DataFrame,
    baseline_probability: np.ndarray,
    full_probability: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    families = frame["family_id"].astype(str).unique()
    positions = {
        family: np.flatnonzero(frame["family_id"].astype(str).to_numpy() == family)
        for family in families
    }
    y = frame["failure_next"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sampled = rng.choice(families, size=len(families), replace=True)
        row_index = np.concatenate([positions[family] for family in sampled])
        baseline_loss = log_loss(
            y[row_index], baseline_probability[row_index], labels=[0, 1]
        )
        full_loss = log_loss(y[row_index], full_probability[row_index], labels=[0, 1])
        estimates[index] = baseline_loss - full_loss
    return {
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "one_sided_p": float((1 + np.sum(estimates <= 0)) / (len(estimates) + 1)),
        "bootstrap_iterations": int(n_bootstrap),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def evaluate_locked_holdout(
    fitted: FittedHazardComparison,
    holdout: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260926,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate frozen models once on holdout with family-clustered inference."""
    _require_estimable(holdout)
    y = holdout["failure_next"].astype(int).to_numpy()
    baseline_probability = fitted.baseline.predict_proba(holdout)[:, 1]
    full_probability = fitted.full.predict_proba(holdout)[:, 1]
    baseline_loss = float(log_loss(y, baseline_probability, labels=[0, 1]))
    full_loss = float(log_loss(y, full_probability, labels=[0, 1]))
    overall_bootstrap = _cluster_bootstrap_improvement(
        holdout,
        baseline_probability,
        full_probability,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    model_results: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for offset, (model, group) in enumerate(holdout.groupby("model", sort=True)):
        position = holdout.index.get_indexer(group.index)
        group_y = group["failure_next"].astype(int).to_numpy()
        group_baseline = baseline_probability[position]
        group_full = full_probability[position]
        bootstrap = _cluster_bootstrap_improvement(
            group.reset_index(drop=True),
            group_baseline,
            group_full,
            n_bootstrap=n_bootstrap,
            seed=seed + offset + 1,
        )
        result = {
            "rows": int(len(group)),
            "families": int(group["family_id"].nunique()),
            "events": int(group_y.sum()),
            "baseline_log_loss": float(
                log_loss(group_y, group_baseline, labels=[0, 1])
            ),
            "full_log_loss": float(log_loss(group_y, group_full, labels=[0, 1])),
            "improvement": float(
                log_loss(group_y, group_baseline, labels=[0, 1])
                - log_loss(group_y, group_full, labels=[0, 1])
            ),
            **bootstrap,
        }
        model_results[str(model)] = result
        raw_p[str(model)] = float(bootstrap["one_sided_p"])

    adjusted = _holm_adjust(raw_p)
    supported_models = []
    for model, result in model_results.items():
        result["holm_adjusted_p"] = adjusted[model]
        result["supports_improvement"] = bool(
            result["ci95"][0] > 0 and adjusted[model] < alpha
        )
        if result["supports_improvement"]:
            supported_models.append(model)
    return {
        "baseline_log_loss": baseline_loss,
        "full_log_loss": full_loss,
        "improvement": baseline_loss - full_loss,
        **overall_bootstrap,
        "models": model_results,
        "supported_models": supported_models,
        "claim_supported": len(supported_models) >= 2,
    }
