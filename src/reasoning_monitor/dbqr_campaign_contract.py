"""Frozen call budget, framing variants and seven-panel DBQR plot contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("CQ", "CS", "RS", "LS", "ES", "SS", "RQ6_mean")


def load_budget_contract(path: str | Path) -> tuple[dict, dict]:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if tuple(config.get("metrics", ())) != METRICS:
        raise ValueError("The seven-metric plot contract changed")
    if config.get("k_repeats") != 5 or config.get("l_framing_variants") != 2:
        raise ValueError("DBQR-RQ-v1 requires frozen K=5 and L=2")
    models = config.get("models") or []
    conditions = config.get("conditions") or []
    per_cell = int(config["k_repeats"]) + int(config["l_framing_variants"])
    stages = {}
    for name, stage in config["stages"].items():
        cells = (int(stage["dialogues"]) * int(stage["turns_per_dialogue"])
                 * len(models) * len(conditions))
        calls = cells * per_cell
        stages[name] = {
            "cells": cells,
            "logical_calls": calls,
            "maximum_request_attempts": calls * (int(config["max_retries"]) + 1),
            "maximum_output_tokens": calls * int(config["max_output_tokens_per_call"]),
            "maximum_input_characters": calls * int(config["max_input_characters_per_call"]),
        }
    practice = config["stages"]["practice_transport"]
    staged_cells = (len(practice["staged_review_turns"]) * len(models) * len(conditions))
    authorized_actions = []
    practice = config["stages"]["practice_transport"]
    if practice.get("stage_review_authorized"):
        authorized_actions.append("practice_transport.stage_review")
    if practice.get("full_execution_authorized"):
        authorized_actions.append("practice_transport.full_execution")
    report = {
        "protocol_id": config["protocol_id"],
        "no_model_calls": True,
        "execution_authorized": False,
        "authorized_actions": authorized_actions,
        "models": [model["name"] for model in models],
        "conditions": list(conditions),
        "calls_per_cell": per_cell,
        "stages": stages,
        "practice_staged_review_calls_included_in_total": staged_cells * per_cell,
        "all_planned_stages_logical_calls": sum(row["logical_calls"] for row in stages.values()),
        "training_plus_test_logical_calls": (
            stages["training_calibration"]["logical_calls"]
            + stages["test_confirmation"]["logical_calls"]),
        "training_plus_test_maximum_output_tokens": (
            stages["training_calibration"]["maximum_output_tokens"]
            + stages["test_confirmation"]["maximum_output_tokens"]),
        "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    return config, report


def framed_prompts(original_prompt: str, config: dict) -> list[str]:
    variants = config.get("framing_variants") or []
    if len(variants) != config.get("l_framing_variants") or len(set(variants)) != len(variants):
        raise ValueError("Framing variants are missing or duplicated")
    return [f"{prefix}\n\n{original_prompt}" for prefix in variants]


def plot_seven_metrics(frame: pd.DataFrame, output_dir: str | Path, *, title: str) -> list[Path]:
    required = {"model", "condition", "turn", *METRICS}
    if not required <= set(frame):
        raise ValueError(f"Missing plot columns: {sorted(required - set(frame))}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"self_history": "#0072B2", "oracle_answer_history": "#009E73"}
    for model, model_frame in frame.groupby("model", sort=True):
        figure, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True,
                                    constrained_layout=True)
        for metric, axis in zip(METRICS, axes.flat, strict=False):
            axis.set_title(metric + (" (complete cases)" if metric == "RQ6_mean" else ""))
            if metric == "ES":
                axis.axvspan(1, 4.5, color="#bbbbbb", alpha=0.18)
                axis.text(0.02, 0.92, "turns 1-4: warmup", transform=axis.transAxes,
                          fontsize=8)
            measured = False
            for condition in ("self_history", "oracle_answer_history"):
                subset = model_frame[model_frame["condition"] == condition]
                grouped = subset.groupby("turn")[metric]
                mean = grouped.mean()
                count = grouped.count()
                finite = mean[np.isfinite(mean)]
                if not finite.empty:
                    measured = True
                    axis.plot(finite.index, finite.values, "o-", color=colors[condition],
                              label=condition)
                    axis.text(
                        0.02,
                        0.08 if condition == "self_history" else 0.02,
                        f"{condition}: n/turn={int(count.min())}..{int(count.max())}",
                        transform=axis.transAxes,
                        fontsize=7,
                        color=colors[condition],
                    )
            if measured:
                axis.legend(loc="upper right", fontsize=7)
            else:
                axis.text(0.5, 0.5, "NOT MEASURED", ha="center", va="center",
                          transform=axis.transAxes, color="#777777")
            axis.set_ylim(-0.03, 1.08)
            axis.set_xticks(range(1, 11))
            axis.set_xlabel("Dialogue turn")
            axis.set_ylabel("Score")
            axis.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        figure.suptitle(f"{title}\n{model} | six RQEval metrics + complete-case mean")
        for extension in ("png", "pdf"):
            path = output / f"dbqr_seven_metrics_{model}.{extension}"
            figure.savefig(path, dpi=150)
            paths.append(path)
        plt.close(figure)
    return paths


def fixture_plot_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "model": "call_free_fixture",
            "condition": record["condition"],
            "turn": int(record["turn"]),
            "CQ": float(record["correct"]),
        }
        row.update({metric: np.nan for metric in METRICS if metric != "CQ"})
        rows.append(row)
    return pd.DataFrame(rows)


def write_budget_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
