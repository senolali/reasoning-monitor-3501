"""Exploratory next-turn prediction on saved CDR-v1.4 calibration records."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reasoning_monitor.dialogue_experiment import records_to_turn_frame

FEATURE_SETS = {
    "turn_only": ["turn"],
    "turn_context": ["turn", "context_chars"],
    "turn_context_tokens": ["turn", "context_chars", "n_tokens", "token_delta"],
    "turn_context_latency": ["turn", "context_chars", "latency_s"],
}


def prediction_rows(records):
    frame = records_to_turn_frame(records)
    frame = frame[frame.condition == "user_only_history"].copy()
    if frame.duplicated(["dialogue_id", "turn"]).any():
        raise ValueError("duplicate dialogue turns")
    if not frame.ok.all():
        raise ValueError("failed calls require separate missingness analysis")
    rows = []
    for _, group in frame.groupby("dialogue_id", sort=True):
        group = group.sort_values("turn").copy()
        if group.turn.tolist() != list(range(1, 13)):
            raise ValueError("expected twelve consecutive turns")
        group["token_delta"] = group.n_tokens.diff().fillna(0)
        group["next_error"] = (1 - group.CQ_raw.shift(-1))
        group["before_first_error"] = group.CQ_raw.cummin().astype(bool)
        rows.append(group.iloc[:-1])
    return pd.concat(rows, ignore_index=True)


def grouped_predictions(frame, features, group_column):
    predictions = np.zeros(len(frame))
    for group in sorted(frame[group_column].unique()):
        test = frame[group_column] == group
        train = ~test
        y = frame.loc[train, "next_error"].astype(int)
        if y.empty:
            raise ValueError("at least two groups required")
        if y.nunique() == 1:
            predictions[test] = float(y.iloc[0])
        else:
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
            )
            estimator.fit(frame.loc[train, features], y)
            predictions[test] = estimator.predict_proba(frame.loc[test, features])[:, 1]
    return predictions


def score_predictions(frame, probability):
    y = frame.next_error.to_numpy(dtype=int)
    at_risk = frame.before_first_error.to_numpy(dtype=bool)
    event = at_risk & (y == 1)
    safe = at_risk & (y == 0)
    alarm = probability >= 0.5
    return {
        "rows": len(y),
        "next_turn_errors": int(y.sum()),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        "first_failure_opportunities": int(event.sum()),
        "first_failures_warned_one_turn_ahead": int((alarm & event).sum()),
        "pre_first_failure_false_alarms": int((alarm & safe).sum()),
        "pre_first_failure_negative_opportunities": int(safe.sum()),
        "pre_first_failure_false_alarm_rate": (
            float(alarm[safe].mean()) if safe.any() else None
        ),
        "pre_first_failure_brier": (
            float(brier_score_loss(y[at_risk], probability[at_risk]))
            if at_risk.any() else None
        ),
    }


def analyse_warning(records):
    frame = prediction_rows(records)
    results = {}
    for grouping in ("dialogue_id", "template"):
        variants = {}
        for name, features in FEATURE_SETS.items():
            probability = grouped_predictions(frame, features, grouping)
            frame[f"p_{grouping}_{name}"] = probability
            variants[name] = score_predictions(frame, probability)
        for value in variants.values():
            value["brier_gain_over_turn_only"] = (
                variants["turn_only"]["brier"] - value["brier"]
            )
            value["brier_gain_over_turn_context"] = (
                variants["turn_context"]["brier"] - value["brier"]
            )
        results[grouping] = variants
    return results, frame
