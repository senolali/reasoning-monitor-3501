"""Call-free CDR-Stream early-warning forensic audit.

Reads saved turn metrics, never calls a model, and writes a separate exploratory
report under results/cdr_stream_v1_early_warning_forensics/.

    python scripts\forensic_cdr_early_warning.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.evaluation import early_warning_forensics as ew  # noqa: E402

OUTPUT = ROOT / "results/cdr_stream_v1_early_warning_forensics"
CONTROL_INPUTS = {
    "gpt4o_mini": ROOT / "results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini/control_turn_metrics.csv",
    "llama3_70b": ROOT / "results/cdr_stream_v1_control_monitoring/validation/llama3_70b/control_turn_metrics.csv",
    "gemma3_12b_it": ROOT / "results/cdr_stream_v1_control_monitoring/validation/gemma3_12b_it/control_turn_metrics.csv",
}
STAGE_A = ROOT / "results/cdr_stream_v1_main/analysis_stage_a/turn_metrics.csv"
CONTROL_METRICS = ("CQ", "LS", "ES")
ALL_METRICS = ("CQ", "CS", "RS", "LS", "ES", "SS")
COLORS = {"CQ": "#0072B2", "CS": "#E69F00", "RS": "#009E73", "LS": "#CC79A7",
          "ES": "#56B4E9", "SS": "#D55E00"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate(frame: pd.DataFrame, metrics: tuple[str, ...]) -> dict[str, dict]:
    selected = {}
    for metric in metrics:
        if metric not in frame or frame[metric].notna().sum() == 0:
            continue
        direction = "down" if metric == "CQ" else "both"
        nulls = ew.stream_arrays(frame, metric, ew.NULL_SCENARIOS).values()
        threshold, far = ew.select_threshold(nulls, direction=direction)
        selected[metric] = {"direction": direction, "threshold": threshold,
                            "development_null_far": far}
    return selected


def empirical_tables(frames: dict[str, pd.DataFrame], calibration: dict[str, dict], *,
                     label: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    alarms, lagged, summaries = [], [], []
    for model, frame in frames.items():
        available = tuple(metric for metric in calibration
                          if metric in frame and frame[metric].notna().any())
        cq_spec = calibration.get("CQ")
        if cq_spec is None or cq_spec["threshold"] is None:
            continue
        cq = ew.alarm_table(frame, "CQ", cq_spec["threshold"], direction="down")
        cq["model"], cq["dataset"] = model, label
        alarms.append(cq)
        for metric in available:
            spec = calibration[metric]
            lag = ew.lead_lag_auc(frame, metric, direction=spec["direction"])
            lag["model"], lag["dataset"] = model, label
            lagged.append(lag)
            if spec["threshold"] is None:
                continue
            table = ew.alarm_table(frame, metric, spec["threshold"],
                                   direction=spec["direction"])
            table["model"], table["dataset"] = model, label
            alarms.append(table)
            summary = ew.summarize_against_cq(table, cq)
            summaries.append({"dataset": label, "model": model, "metric": metric,
                              **spec, **summary})
    return (pd.concat(alarms, ignore_index=True), pd.concat(lagged, ignore_index=True),
            summaries)


def plot_injection(frame: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, part in frame.groupby("method"):
        label = "Raw fixed-baseline CUSUM" if method.startswith("fixed") else "W20 + warm-up20"
        axes[0].plot(part["delta"], part["warning_by_turn_33"], marker="o", label=label)
        axes[1].plot(part["delta"], part["detection_by_turn_67"], marker="o", label=label)
    axes[0].set_title("Injected precursor detected before outcome (t≤33)")
    axes[1].set_title("Detected within 33 turns after outcome (t≤67)")
    for ax in axes:
        ax.set_xlabel("Injected downward shift")
        ax.set_ylabel("Detection probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0].legend()
    return save(fig, "injection_sensitivity")


def plot_balance(summary: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control = summary[summary["dataset"] == "control"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (model, part) in zip(axes, control.groupby("model"), strict=False):
        for row in part.itertuples():
            ax.scatter(row.null_alarm_rate, row.detection_rate, s=70,
                       color=COLORS[row.metric], label=row.metric)
            ax.annotate(row.metric, (row.null_alarm_rate, row.detection_rate),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.axvline(0.05, color="black", ls="--", lw=1)
        ax.axhline(0.5, color="black", ls=":", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Stationary false-alarm rate")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Drift-stream detection rate")
    return save(fig, "empirical_detection_false_alarm")


def plot_lags(lagged: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control = lagged[lagged["dataset"] == "control"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (model, part) in zip(axes, control.groupby("model"), strict=False):
        for metric, metric_part in part.groupby("metric"):
            ax.plot(metric_part["lag"], metric_part["auc"], color=COLORS[metric],
                    label=metric, lw=1.7)
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Future CQ horizon (turns)")
        ax.set_ylim(0.35, 0.8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("AUROC among currently correct turns")
    axes[-1].legend(loc="best", fontsize=8)
    return save(fig, "lead_lag_auc")


def save(fig, stem: str) -> list[str]:
    import matplotlib.pyplot as plt

    files = []
    for suffix in ("png", "pdf"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=150)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def report_markdown(summary: dict, empirical: pd.DataFrame, lagged: pd.DataFrame) -> str:
    validation = empirical[empirical["dataset"] == "control"].copy()
    best = (lagged.sort_values("auc", ascending=False)
            .groupby(["dataset", "model", "metric"], as_index=False).first())
    lines = [
        "# CDR-Stream Early-Warning Forensic Audit",
        "",
        "**Status:** call-free, post-hoc sensitivity analysis; it does not replace frozen results.",
        "",
        "## Structural timing check",
        "",
        f"The historical W=20 plus warm-up=20 path cannot emit before source turn "
        f"{summary['timing']['legacy_first_possible_turn']}. S1/S3 first change at turn 34. "
        "It can therefore detect those changes only after onset, not warn before onset.",
        "",
        "## Empirical fixed-baseline sensitivity",
        "",
        "Thresholds were selected on GPT stationary control dialogues and transferred without "
        "refitting to Llama and Gemma. Non-CQ metrics were tested two-sided. Values below are "
        "exploratory and retain within-dialogue dependence.",
        "",
        "| Model | Metric | Null FAR | Detection | Leads CQ / both detected | Median lead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in validation.itertuples():
        lines.append(f"| {row.model} | {row.metric} | {row.null_alarm_rate:.3f} | "
                     f"{row.detection_rate:.3f} | {row.leads_cq}/{row.both_detected} | "
                     f"{row.median_lead if row.median_lead is not None else '—'} |")
    lines += ["", "## Descriptive lead-lag ceiling", "",
              "The table reports the best post-hoc AUROC over horizons 1–20 while CQ is still "
              "correct. It is a diagnostic upper bound, not a validated predictor.", "",
              "| Dataset | Model | Metric | Best lag | AUROC | n |",
              "|---|---|---:|---:|---:|---:|"]
    for row in best.itertuples():
        lines.append(f"| {row.dataset} | {row.model} | {row.metric} | {row.lag} | "
                     f"{row.auc:.3f} | {row.n} |")
    uncalibrated = [metric for metric, spec in
                    summary["calibration"]["dialogue_history_gpt_development"].items()
                    if spec["threshold"] is None]
    lines += ["", "## Uncalibratable signals", "",
              "In the GPT dialogue-history arm, the combined two-sided rule could not meet "
              "the 5% stationary false-alarm target for: " + ", ".join(uncalibrated) + ". "
              "They remain in the descriptive lead-lag table but are excluded from alarm claims."]
    lines += [
        "",
        "## Interpretation rule",
        "",
        "A candidate is not considered robust merely because it sometimes leads CQ. It must "
        "also preserve the stationary false-alarm target and transfer across validation models. "
        "If no metric satisfies all three requirements, the defensible conclusion remains that "
        "the six observed signals contain no reliable single-stream precursor under this design.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in (*CONTROL_INPUTS.values(), STAGE_A) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen analysis inputs: " + ", ".join(missing))

    controls = {model: pd.read_csv(path) for model, path in CONTROL_INPUTS.items()}
    gpt_calibration = calibrate(controls["gpt4o_mini"], CONTROL_METRICS)
    alarms_control, lag_control, summaries_control = empirical_tables(
        controls, gpt_calibration, label="control")

    stage = pd.read_csv(STAGE_A)
    stage_gpt = stage[stage["model"] == "gpt4o_mini"].copy()
    all_calibration = calibrate(stage_gpt, ALL_METRICS)
    alarms_stage, lag_stage, summaries_stage = empirical_tables(
        {"gpt4o_mini": stage_gpt}, all_calibration, label="dialogue_history")

    injection = ew.synthetic_injection()
    alarms = pd.concat([alarms_control, alarms_stage], ignore_index=True)
    lagged = pd.concat([lag_control, lag_stage], ignore_index=True)
    empirical = pd.DataFrame(summaries_control + summaries_stage)
    injection.to_csv(OUTPUT / "synthetic_injection.csv", index=False)
    alarms.to_csv(OUTPUT / "alarm_events.csv", index=False)
    empirical.to_csv(OUTPUT / "empirical_summary.csv", index=False)
    lagged.to_csv(OUTPUT / "lead_lag_auc.csv", index=False)

    figures = plot_injection(injection) + plot_balance(empirical) + plot_lags(lagged)
    summary = {
        "analysis": "cdr_stream_v1_early_warning_forensic_audit",
        "call_free": True,
        "exploratory": True,
        "replaces_frozen_analysis": False,
        "timing": {
            "legacy_window": 20,
            "legacy_detector_warmup": 20,
            "legacy_first_possible_turn": ew.first_possible_legacy_turn(),
            "first_change_turns": {"S1": 34, "S2": 51, "S3": 34},
        },
        "calibration": {
            "control_gpt_development": gpt_calibration,
            "dialogue_history_gpt_development": all_calibration,
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                          for path in (*CONTROL_INPUTS.values(), STAGE_A)},
        "synthetic_injection": injection.replace({np.nan: None}).to_dict(orient="records"),
        "empirical": empirical.replace({np.nan: None}).to_dict(orient="records"),
        "figures": figures,
        "limitations": [
            "Post-hoc sensitivity analysis; thresholds are not confirmatory.",
            "Only CQ, LS and ES exist in the K=1 control arm.",
            "All six metrics are available only for GPT dialogue-history Stage A.",
            "Lead-lag AUROC is descriptive and does not treat turns as independent.",
            "An exogenous difficulty change cannot be causally predicted from unchanged pre-change inputs.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "REPORT.md").write_text(
        report_markdown(summary, empirical, lagged) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "legacy_first_possible_turn": ew.first_possible_legacy_turn(),
        "empirical_rows": len(empirical),
        "lead_lag_rows": len(lagged),
        "model_calls": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
