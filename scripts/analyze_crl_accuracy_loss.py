"""Post-hoc call-free CRL robustness test against independent accuracy loss."""

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
    crl_accuracy_loss_sensitivity,
    paired_primary_endpoint,
)

RESULTS = ROOT / "results/lost_conversation_v2_science"
OUTPUT = RESULTS / "robustness_accuracy_loss"
SEED = 20261002
REPETITIONS = 10_000


def _plot(frame: pd.DataFrame, paired: pd.DataFrame, output: Path) -> list[str]:
    targets = {
        (row.model, row.task_id): bool(row.fresh_prefix == 1
                                       and row.dialogue_history == 0)
        for row in paired.itertuples()
    }
    data = frame.copy()
    data["accuracy_loss"] = [targets[(row.model, row.task_id)] for row in data.itertuples()]
    models = list(data.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.5), sharey=True,
                             constrained_layout=True)
    for axis, model in zip(axes, models, strict=True):
        subset = data[data.model == model]
        groups = [subset.loc[~subset.accuracy_loss, "CRL"],
                  subset.loc[subset.accuracy_loss, "CRL"]]
        axis.boxplot(groups, tick_labels=["No accuracy loss", "Accuracy loss"], widths=0.6)
        for position, values in enumerate(groups, 1):
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(position + jitter, values, alpha=0.65, s=18, color="#0072B2")
        axis.set(title=model, xlabel="Fresh correct / dialogue failed", ylim=(-0.04, 1.04))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Cumulative Resolution Lag (CRL)")
    fig.suptitle("Post-hoc robustness: pre-final CRL versus independent final accuracy loss")
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"crl_accuracy_loss.{extension}"
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
    paired, _primary = paired_primary_endpoint(
        metrics, seed=SEED, repetitions=REPETITIONS)
    empty_units = {
        (row["model"], row["task_id"])
        for row in records
        if row["role"] == "primary" and not str(row.get("raw_text", "")).strip()
    }
    full = crl_accuracy_loss_sensitivity(
        crl, paired, seed=SEED, repetitions=REPETITIONS)
    transport_excluded = crl_accuracy_loss_sensitivity(
        crl, paired, seed=SEED, repetitions=REPETITIONS,
        excluded_model_tasks=empty_units)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis": "LIC-RQ-v2 CRL versus independent final accuracy-loss robustness",
        "post_hoc": True,
        "no_model_calls": True,
        "records_sha256": sha256_file(records_path),
        "seed": SEED,
        "permutations_and_bootstrap_repetitions": REPETITIONS,
        "target": "fresh_final_correct_and_dialogue_final_failed",
        "all_model_task_units": full,
        "empty_primary_model_task_units": [
            {"model": model, "task_id": task_id} for model, task_id in sorted(empty_units)
        ],
        "transport_affected_units_excluded": transport_excluded,
        "figures": _plot(crl, paired, OUTPUT),
        "interpretation_rule": (
            "This strengthens CRL only if the pooled model-stratified clustered "
            "permutation p is below 0.05 and the transport-excluded direction persists."
        ),
    }
    (OUTPUT / "assessment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

