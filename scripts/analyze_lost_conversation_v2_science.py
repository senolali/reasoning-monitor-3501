"""Run the frozen call-free confirmatory LIC-RQ-v2 science analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.lost_conversation import sha256_file  # noqa: E402
from reasoning_monitor.lost_conversation_science_analysis import (  # noqa: E402
    crl_discrimination,
    format_valid_sensitivity,
    paired_primary_endpoint,
    rqeval_cluster_permutation,
)
from reasoning_monitor.lost_conversation_semantic_rescore import semantic_records  # noqa: E402

RESULTS = ROOT / "results/lost_conversation_v2_science"
OUTPUT = RESULTS / "confirmatory"
SEED = 20261002
REPETITIONS = 10_000


def _plot_primary(endpoint: dict, output: Path) -> list[str]:
    models = list(endpoint)
    history = [endpoint[model]["dialogue_success_rate"] for model in models]
    fresh = [endpoint[model]["fresh_success_rate"] for model in models]
    x = np.arange(len(models))
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    width = 0.36
    axis.bar(x - width / 2, history, width, label="Dialogue history", color="#0072B2")
    axis.bar(x + width / 2, fresh, width, label="Fresh prefix", color="#D55E00")
    for index, model in enumerate(models):
        difference = endpoint[model]["paired_risk_difference_fresh_minus_dialogue"]
        axis.text(index, max(history[index], fresh[index]) + 0.035,
                  f"Delta={difference:+.3f}", ha="center", fontsize=9)
    axis.set(xticks=x, xticklabels=models, ylim=(0, 1.12), ylabel="Final success rate",
             title="LIC-RQ-v2 paired final-turn success (48 tasks per model)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"paired_final_success.{extension}"
        fig.savefig(path, dpi=160)
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_crl(crl: pd.DataFrame, output: Path) -> list[str]:
    models = list(crl.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.5), sharey=True,
                             constrained_layout=True)
    for axis, model in zip(axes, models, strict=True):
        data = crl[crl.model == model].dropna(
            subset=["CRL", "final_positive_readiness_gap"])
        groups = [data.loc[~data.final_positive_readiness_gap.astype(bool), "CRL"],
                  data.loc[data.final_positive_readiness_gap.astype(bool), "CRL"]]
        if all(len(group) for group in groups):
            axis.boxplot(groups, tick_labels=["No final gap", "Final gap"], widths=0.6)
        else:
            axis.text(0.5, 0.5, "AUROC unavailable:\nonly one target class",
                      transform=axis.transAxes, ha="center", va="center")
        for position, values in enumerate(groups, 1):
            if len(values):
                jitter = np.linspace(-0.08, 0.08, len(values))
                axis.scatter(position + jitter, values, alpha=0.65, s=18, color="#0072B2")
        axis.set(title=model, xlabel="Final readiness target", ylim=(-0.04, 1.04))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Cumulative Resolution Lag (CRL)")
    fig.suptitle("Pre-final CRL by final context-specific unresolved state")
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"crl_discrimination.{extension}"
        fig.savefig(path, dpi=160)
        paths.append(path.name)
    plt.close(fig)
    return paths


def main() -> int:
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    crl = pd.read_csv(RESULTS / "crl_by_task.csv")
    records_path = RESULTS / "data/records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    paired, primary = paired_primary_endpoint(
        metrics, seed=SEED, repetitions=REPETITIONS)
    crl_result = crl_discrimination(crl, seed=SEED, repetitions=REPETITIONS)
    sensitivity = format_valid_sensitivity(paired, semantic_records(records))
    rqeval = rqeval_cluster_permutation(
        metrics, records, paired, seed=SEED, repetitions=REPETITIONS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = _plot_primary(primary, OUTPUT) + _plot_crl(crl, OUTPUT)
    report = {
        "analysis": "LIC-RQ-v2 frozen science confirmatory analysis",
        "no_model_calls": True,
        "records_sha256": sha256_file(records_path),
        "seed": SEED,
        "permutation_or_bootstrap_repetitions": REPETITIONS,
        "primary_endpoint": primary,
        "format_valid_sensitivity": sensitivity,
        "auxiliary_crl": crl_result,
        "rqeval_early_warning": rqeval,
        "figures": figures,
        "missingness_policy": "no_imputation",
        "context_length_operationalization": "packed message character count",
        "claims": {
            "primary_confirmed_if_holm_p_below_0_05": [
                model for model, row in primary.items()
                if row["holm_adjusted_p_across_models"] < 0.05
            ],
            "crl_model_stratified_warning_supported": (
                crl_result["pooled_model_stratified"]
                ["task_clustered_permutation_one_sided_p"] < 0.05),
            "crl_nominal_model_signals": [
                model for model, row in crl_result.items()
                if model != "pooled_model_stratified"
                and row["permutation_one_sided_p"] is not None
                and row["permutation_one_sided_p"] < 0.05
            ],
            "rqeval_warning_supported_after_holm": [
                metric for metric, row in rqeval.items()
                if row["holm_adjusted_p"] is not None and row["holm_adjusted_p"] < 0.05
            ],
        },
    }
    (OUTPUT / "assessment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
