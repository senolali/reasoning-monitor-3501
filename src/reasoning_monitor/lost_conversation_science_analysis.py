"""Confirmatory task-level and cluster-permutation analysis for LIC-RQ-v2."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted((value, name) for name, value in p_values.items() if value is not None)
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(valid)
    for rank, (value, name) in enumerate(valid):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def paired_primary_endpoint(metrics: pd.DataFrame, *, seed: int,
                            repetitions: int) -> tuple[pd.DataFrame, dict]:
    finals = metrics[metrics.turn == metrics.total_turns].copy()
    finals["success"] = ((finals.CQ_status == "measured") & (finals.CQ == 1.0)).astype(int)
    paired = finals.pivot(index=["model", "task_id"], columns="condition",
                          values="success").reset_index()
    rng = np.random.default_rng(seed)
    results = {}
    p_values = {}
    for model, group in paired.groupby("model"):
        history = group.dialogue_history.to_numpy(dtype=int)
        fresh = group.fresh_prefix.to_numpy(dtype=int)
        history_only = int(np.sum((history == 1) & (fresh == 0)))
        fresh_only = int(np.sum((history == 0) & (fresh == 1)))
        discordant = history_only + fresh_only
        p_value = (float(binomtest(history_only, discordant, 0.5).pvalue)
                   if discordant else 1.0)
        differences = fresh - history
        draws = rng.integers(0, len(group), size=(repetitions, len(group)))
        bootstrap = differences[draws].mean(axis=1)
        results[model] = {
            "tasks": len(group),
            "dialogue_success": int(history.sum()),
            "dialogue_success_rate": float(history.mean()),
            "fresh_success": int(fresh.sum()),
            "fresh_success_rate": float(fresh.mean()),
            "paired_risk_difference_fresh_minus_dialogue": float(differences.mean()),
            "paired_risk_difference_bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "fresh_only_successes": fresh_only,
            "dialogue_only_successes": history_only,
            "discordant_pairs": discordant,
            "mcnemar_exact_two_sided_p": p_value,
        }
        p_values[model] = p_value
    adjusted = holm_adjust(p_values)
    for model, value in adjusted.items():
        results[model]["holm_adjusted_p_across_models"] = value
    return paired, results


def crl_discrimination(crl: pd.DataFrame, *, seed: int,
                       repetitions: int) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    model_p = {}
    for model, group in crl.groupby("model"):
        observed = group.dropna(subset=["CRL", "final_positive_readiness_gap"]).copy()
        y = observed.final_positive_readiness_gap.astype(bool).astype(int).to_numpy()
        score = observed.CRL.to_numpy(dtype=float)
        cases = int(y.sum())
        controls = int(len(y) - cases)
        row = {
            "tasks": len(group),
            "complete_cases": len(observed),
            "target_cases": cases,
            "target_controls": controls,
            "mean_crl": float(group.CRL.mean()),
            "mean_support_rate": float(group.prefinal_support_rate.mean()),
            "auroc": None,
            "permutation_one_sided_p": None,
            "bootstrap_95_ci": None,
        }
        if cases and controls:
            auc = float(roc_auc_score(y, score))
            permuted = np.empty(repetitions)
            for index in range(repetitions):
                permuted[index] = roc_auc_score(rng.permutation(y), score)
            p_value = float((1 + np.sum(permuted >= auc)) / (repetitions + 1))
            boot = []
            for _index in range(repetitions):
                selection = rng.integers(0, len(y), size=len(y))
                sampled_y = y[selection]
                if sampled_y.min() != sampled_y.max():
                    boot.append(roc_auc_score(sampled_y, score[selection]))
            row.update({
                "auroc": auc,
                "permutation_one_sided_p": p_value,
                "bootstrap_95_ci": ([float(np.quantile(boot, 0.025)),
                                      float(np.quantile(boot, 0.975))]
                                     if boot else None),
            })
        results[model] = row
        model_p[model] = row["permutation_one_sided_p"]
    adjusted = holm_adjust(model_p)
    for model, value in adjusted.items():
        results[model]["holm_adjusted_p_across_models"] = value

    task_ids = sorted(crl.task_id.unique())

    def stratified_auc(frame: pd.DataFrame,
                       task_weights: Mapping[str, int] | None = None) -> tuple[float | None, int]:
        numerator = 0.0
        pairs = 0
        for _model, group in frame.groupby("model"):
            observed = group.dropna(subset=["CRL", "final_positive_readiness_gap"])
            y = observed.final_positive_readiness_gap.astype(bool).astype(int)
            weights = np.array([
                task_weights.get(task_id, 0) if task_weights is not None else 1
                for task_id in observed.task_id
            ], dtype=float)
            keep = weights > 0
            observed = observed[keep]
            y = y[keep]
            weights = weights[keep]
            cases = float(weights[y.to_numpy() == 1].sum())
            controls = float(weights[y.to_numpy() == 0].sum())
            if cases and controls:
                weight = cases * controls
                numerator += roc_auc_score(
                    y, observed.CRL, sample_weight=weights) * weight
                pairs += weight
        return ((float(numerator / pairs) if pairs else None), int(pairs))

    pooled_auc, pooled_pairs = stratified_auc(crl)
    target = {
        (row.model, row.task_id): row.final_positive_readiness_gap
        for row in crl.itertuples()
    }
    null = []
    for _index in range(repetitions):
        permuted = rng.permutation(task_ids)
        mapping = dict(zip(task_ids, permuted, strict=True))
        frame = crl.copy()
        frame["final_positive_readiness_gap"] = [
            target.get((row.model, mapping[row.task_id])) for row in frame.itertuples()
        ]
        value, _pairs = stratified_auc(frame)
        if value is not None:
            null.append(value)
    boot = []
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    for _index in range(repetitions):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        counts = np.bincount([task_index[task_id] for task_id in sampled],
                             minlength=len(task_ids))
        weights = dict(zip(task_ids, counts, strict=True))
        value, _pairs = stratified_auc(crl, weights)
        if value is not None:
            boot.append(value)
    results["pooled_model_stratified"] = {
        "source_tasks": len(task_ids),
        "comparable_case_control_pairs": pooled_pairs,
        "auroc": pooled_auc,
        "task_clustered_permutation_one_sided_p": (
            float((1 + np.sum(np.asarray(null) >= pooled_auc)) / (len(null) + 1))
            if pooled_auc is not None else None),
        "task_clustered_bootstrap_95_ci": (
            [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
            if boot else None),
    }
    return results


def crl_accuracy_loss_sensitivity(
    crl: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    seed: int,
    repetitions: int,
    excluded_model_tasks: set[tuple[str, str]] | None = None,
) -> dict:
    """Test CRL against fresh-correct/dialogue-failed final accuracy loss.

    This target is independent of the readiness marker used to construct CRL.
    It is a post-hoc robustness analysis, not a replacement primary endpoint.
    """
    excluded = excluded_model_tasks or set()
    target = {
        (row.model, row.task_id): bool(row.fresh_prefix == 1
                                       and row.dialogue_history == 0)
        for row in paired.itertuples()
        if (row.model, row.task_id) not in excluded
    }
    frame = crl[
        [(row.model, row.task_id) in target for row in crl.itertuples()]
    ].copy()
    frame["final_positive_readiness_gap"] = [
        target[(row.model, row.task_id)] for row in frame.itertuples()
    ]
    result = crl_discrimination(frame, seed=seed, repetitions=repetitions)
    result["target_definition"] = "fresh_final_correct_and_dialogue_final_failed"
    result["post_hoc_robustness_analysis"] = True
    result["excluded_model_task_units"] = len(excluded)
    return result


def format_valid_sensitivity(paired: pd.DataFrame, semantic_records: list[dict]) -> dict:
    validity = {
        (row["model"], row["task_id"], row["condition"]): bool(row["contract_valid"])
        for row in semantic_records
        if row["role"] == "primary" and int(row["turn"]) == int(row["total_turns"])
    }
    results = {}
    for model, group in paired.groupby("model"):
        complete = group[
            [validity[(model, task_id, "dialogue_history")]
             and validity[(model, task_id, "fresh_prefix")]
             for task_id in group.task_id]
        ]
        history = complete.dialogue_history.to_numpy(dtype=int)
        fresh = complete.fresh_prefix.to_numpy(dtype=int)
        history_only = int(np.sum((history == 1) & (fresh == 0)))
        fresh_only = int(np.sum((history == 0) & (fresh == 1)))
        discordant = history_only + fresh_only
        results[model] = {
            "both_final_responses_semantically_valid": len(complete),
            "excluded_for_format_invalidity": len(group) - len(complete),
            "dialogue_success_rate": float(history.mean()) if len(complete) else None,
            "fresh_success_rate": float(fresh.mean()) if len(complete) else None,
            "paired_difference": (float((fresh - history).mean()) if len(complete) else None),
            "fresh_only_successes": fresh_only,
            "dialogue_only_successes": history_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(history_only, discordant, 0.5).pvalue)
                if discordant else 1.0),
        }
    return results


def _weighted_residual(values: np.ndarray, design: np.ndarray,
                       sqrt_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weighted_design = design * sqrt_weight[:, None]
    pseudo_inverse = np.linalg.pinv(weighted_design)
    weighted_values = values * sqrt_weight
    residual = weighted_values - weighted_design @ (pseudo_inverse @ weighted_values)
    return residual, weighted_design @ pseudo_inverse


def rqeval_cluster_permutation(
    metrics: pd.DataFrame,
    records: list[dict],
    paired_outcomes: pd.DataFrame,
    *,
    seed: int,
    repetitions: int,
) -> dict:
    """Test lower pre-final metrics against final dialogue failure.

    The outcome is permuted as a complete source-task bundle across all models.
    Observation weights give each model-task equal total weight. This preserves
    task clustering while adjusting for model, turn, information fraction and
    packed-context character length.
    """
    primary_context = {
        (row["model"], row["task_id"], row["condition"], int(row["turn"])):
        sum(len(str(message.get("content", ""))) for message in row["messages"])
        for row in records if row["role"] == "primary"
    }
    outcome = {
        (row.model, row.task_id): 1 - int(row.dialogue_history)
        for row in paired_outcomes.itertuples()
    }
    pre = metrics[(metrics.condition == "dialogue_history")
                  & (metrics.turn < metrics.total_turns)].copy()
    pre["context_characters"] = [
        primary_context[(row.model, row.task_id, row.condition, int(row.turn))]
        for row in pre.itertuples()
    ]
    pre["final_failure"] = [outcome[(row.model, row.task_id)] for row in pre.itertuples()]
    task_ids = sorted(pre.task_id.unique())
    rng = np.random.default_rng(seed)
    results = {}
    raw_p = {}
    for metric in ("CS", "LS", "SS"):
        data = pre.dropna(subset=[metric]).copy()
        counts = data.groupby(["model", "task_id"])[metric].transform("size")
        weight = 1.0 / counts.to_numpy(dtype=float)
        sqrt_weight = np.sqrt(weight)
        model_dummies = pd.get_dummies(data.model, drop_first=True, dtype=float)
        continuous = np.column_stack([
            data.turn.to_numpy(dtype=float),
            data.information_fraction.to_numpy(dtype=float),
            np.log1p(data.context_characters.to_numpy(dtype=float)),
        ])
        mean = continuous.mean(axis=0)
        scale = continuous.std(axis=0)
        scale[scale == 0] = 1.0
        continuous = (continuous - mean) / scale
        design = np.column_stack([
            np.ones(len(data)), model_dummies.to_numpy(dtype=float), continuous,
        ])
        risk = 1.0 - data[metric].to_numpy(dtype=float)
        target = data.final_failure.to_numpy(dtype=float)
        risk_residual, _projection = _weighted_residual(risk, design, sqrt_weight)
        target_residual, _ = _weighted_residual(target, design, sqrt_weight)
        denominator = np.linalg.norm(risk_residual) * np.linalg.norm(target_residual)
        statistic = (float(np.dot(risk_residual, target_residual) / denominator)
                     if denominator else math.nan)
        null = np.empty(repetitions)
        models = data.model.to_numpy()
        tasks = data.task_id.to_numpy()
        weighted_design = design * sqrt_weight[:, None]
        pseudo_inverse = np.linalg.pinv(weighted_design)
        for index in range(repetitions):
            permuted_tasks = rng.permutation(task_ids)
            mapping = dict(zip(task_ids, permuted_tasks, strict=True))
            permuted_target = np.array([
                outcome[(model, mapping[task])] for model, task in zip(models, tasks, strict=True)
            ], dtype=float)
            weighted_target = permuted_target * sqrt_weight
            residual = weighted_target - weighted_design @ (pseudo_inverse @ weighted_target)
            null_denominator = np.linalg.norm(risk_residual) * np.linalg.norm(residual)
            null[index] = (np.dot(risk_residual, residual) / null_denominator
                           if null_denominator else 0.0)
        p_value = (float((1 + np.sum(null >= statistic)) / (repetitions + 1))
                   if math.isfinite(statistic) else None)
        results[metric] = {
            "measured_turns": len(data),
            "model_task_units": int(data[["model", "task_id"]].drop_duplicates().shape[0]),
            "source_tasks": int(data.task_id.nunique()),
            "direction": "lower_metric_predicts_final_dialogue_failure",
            "covariates": ["model", "turn", "information_fraction",
                           "log1p_context_characters"],
            "task_clustered_partial_correlation": statistic,
            "permutation_one_sided_p": p_value,
        }
        raw_p[metric] = p_value
    adjusted = holm_adjust(raw_p)
    for metric, value in adjusted.items():
        results[metric]["holm_adjusted_p"] = value
    return results
