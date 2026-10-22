"""Frozen Section 49 call-free normalized alarm-timing analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reasoning_monitor.types import METRIC_ORDER

ANALYSIS = "section49_gpt_normalized_detector_timing"
MODEL = "gpt4o_mini"
DATASET = "e2h_amc"
TAG = "ST2_gpt4o_mini_seed48_W10"
ALARM_LEVEL = "drift"
REFERENCE_METRIC = "CQ"
CANDIDATE_METRICS = ("CS", "LS", "ES", "SS")
DESCRIPTIVE_METRICS = ("RS",)
PHASE_LENGTH = 30
TRANSITIONS = (
    ("easy_to_medium", 30, 60),
    ("medium_to_hard", 60, 90),
)
MIN_LEAD_QUERIES = 1


def _is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_detector_analysis_config(config: dict[str, Any]) -> None:
    """Require the exact Section 49 analysis contract."""
    expected = {
        "analysis": ANALYSIS,
        "source_result_dir": "results/e2h_final_holdout_gpt4o_mini",
        "model": MODEL,
        "dataset": DATASET,
        "tag": TAG,
        "alarm_level": ALARM_LEVEL,
        "reference_metric": REFERENCE_METRIC,
        "candidate_metrics": list(CANDIDATE_METRICS),
        "descriptive_metrics": list(DESCRIPTIVE_METRICS),
        "phase_length": PHASE_LENGTH,
        "transitions": [
            {"name": name, "change_point": change_point, "interval_end": end}
            for name, change_point, end in TRANSITIONS
        ],
        "min_lead_queries": MIN_LEAD_QUERIES,
        "require_all_transitions": True,
        "require_clean_easy_baseline": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"result does not match frozen Section 49 config: {mismatches}")


def _validate_source_gate(data_dir: Path) -> pd.Series:
    gate_path = data_dir / "e2h_final_holdout_gate.csv"
    gate = pd.read_csv(gate_path)
    if len(gate) != 1:
        raise ValueError("Section 49 requires exactly one final-holdout gate row")
    row = gate.iloc[0]
    required_true = (
        "execution_valid",
        "metric_health_pass",
        "cq_holdout_gate_pass",
        "overall_gate_pass",
    )
    if str(row.get("model")) != MODEL or str(row.get("dataset")) != DATASET:
        raise ValueError("Section 49 source gate has the wrong model or dataset")
    if not all(_is_true(row[name]) for name in required_true):
        raise ValueError("Section 49 requires a source track that passed every gate")
    return row


def _phase_summary(per_query: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phases = (("easy", 1, 30), ("medium", 31, 60), ("hard", 61, 90))
    for metric in METRIC_ORDER:
        column = f"{metric}_raw"
        easy_values = pd.to_numeric(
            per_query.loc[per_query["difficulty"] == "easy", column],
            errors="coerce",
        ).dropna()
        easy_mean = float(easy_values.mean()) if len(easy_values) else float("nan")
        for phase, start, end in phases:
            values = pd.to_numeric(
                per_query.loc[per_query["difficulty"] == phase, column],
                errors="coerce",
            ).dropna()
            mean = float(values.mean()) if len(values) else float("nan")
            relative_loss = (
                (easy_mean - mean) / easy_mean
                if easy_mean == easy_mean and abs(easy_mean) > 1e-12 and mean == mean
                else float("nan")
            )
            rows.append(
                {
                    "metric": metric,
                    "phase": phase,
                    "phase_start": start,
                    "phase_end": end,
                    "n_evaluable": len(values),
                    "mean": mean,
                    "relative_loss_from_easy": relative_loss,
                }
            )
    return pd.DataFrame(rows)


def _first_alarm(
    alarms: pd.DataFrame,
    metric: str,
    change_point: int,
    interval_end: int,
) -> pd.Series | None:
    selected = alarms.loc[
        (alarms["metric"] == metric)
        & (alarms["t"] > change_point)
        & (alarms["t"] <= interval_end)
    ].sort_values("t")
    return None if selected.empty else selected.iloc[0]


def _alarm_timing(alarms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    primary = alarms.loc[alarms["level"] == ALARM_LEVEL].copy()
    primary["t"] = pd.to_numeric(primary["t"], errors="raise").astype(int)
    baseline_counts = {
        metric: int(len(primary.loc[(primary["metric"] == metric) & (primary["t"] <= 30)]))
        for metric in (REFERENCE_METRIC, *CANDIDATE_METRICS, *DESCRIPTIVE_METRICS)
    }

    timing_rows = []
    for transition, change_point, interval_end in TRANSITIONS:
        cq_alarm = _first_alarm(primary, REFERENCE_METRIC, change_point, interval_end)
        cq_t = None if cq_alarm is None else int(cq_alarm["t"])
        for metric in (REFERENCE_METRIC, *CANDIDATE_METRICS, *DESCRIPTIVE_METRICS):
            alarm = _first_alarm(primary, metric, change_point, interval_end)
            alarm_t = None if alarm is None else int(alarm["t"])
            delay = None if alarm_t is None else alarm_t - change_point
            lead = None if alarm_t is None or cq_t is None else cq_t - alarm_t
            timing_rows.append(
                {
                    "transition": transition,
                    "change_point": change_point,
                    "interval_end": interval_end,
                    "metric": metric,
                    "detector": None if alarm is None else str(alarm["detector"]),
                    "alarm_t": alarm_t,
                    "delay_queries": delay,
                    "normalized_delay": (None if delay is None else delay / PHASE_LENGTH),
                    "cq_alarm_t": cq_t,
                    "lead_over_cq_queries": lead,
                    "leads_cq": bool(
                        metric != REFERENCE_METRIC
                        and lead is not None
                        and lead >= MIN_LEAD_QUERIES
                    ),
                    "easy_baseline_alarm_count": baseline_counts[metric],
                    "easy_baseline_clean": baseline_counts[metric] == 0,
                }
            )
    timing = pd.DataFrame(timing_rows)

    gate_rows = []
    cq_baseline_clean = baseline_counts[REFERENCE_METRIC] == 0
    for metric in CANDIDATE_METRICS:
        rows = timing.loc[timing["metric"] == metric]
        detected_all = bool(rows["alarm_t"].notna().all())
        cq_detected_all = bool(rows["cq_alarm_t"].notna().all())
        leads_all = bool(rows["leads_cq"].all())
        baseline_clean = baseline_counts[metric] == 0
        passes = all(
            [
                cq_baseline_clean,
                baseline_clean,
                detected_all,
                cq_detected_all,
                leads_all,
            ]
        )
        gate_rows.append(
            {
                "metric": metric,
                "cq_easy_baseline_clean": cq_baseline_clean,
                "metric_easy_baseline_clean": baseline_clean,
                "detected_both_transitions": detected_all,
                "cq_detected_both_transitions": cq_detected_all,
                "leads_cq_both_transitions": leads_all,
                "early_warning_gate_pass": passes,
            }
        )
    gate = pd.DataFrame(gate_rows)
    return timing, gate, bool(gate["early_warning_gate_pass"].any())


def _write_report(
    path: Path,
    phase_summary: pd.DataFrame,
    timing: pd.DataFrame,
    gate: pd.DataFrame,
    overall_pass: bool,
) -> None:
    lines = [
        "# Section 49 GPT Normalized Detector Analysis",
        "",
        "This analysis uses only primary alarms produced by the frozen proposal ",
        "mapping during Section 47. No detector is refit and no model is called.",
        "RS is descriptive because its per-query coverage is sparse.",
        "",
        "## Phase-Normalized Metrics",
        "",
        "| Metric | Phase | Evaluable | Mean | Loss from Easy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in phase_summary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.phase} | {row.n_evaluable} | "
            f"{_display_decimal(row.mean, 3)} | "
            f"{_display_decimal(row.relative_loss_from_easy, 3)} |"
        )
    lines.extend(
        [
            "",
            "## Alarm Timing",
            "",
            "| Transition | Metric | Alarm t | Delay | Normalized delay | CQ lead |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in timing.itertuples(index=False):
        lines.append(
            f"| {row.transition} | {row.metric} | "
            f"{_display(row.alarm_t)} | {_display(row.delay_queries)} | "
            f"{_display(row.normalized_delay, 3)} | "
            f"{_display(row.lead_over_cq_queries)} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Early-Warning Gate",
            "",
            "| Metric | Clean baseline | Both detected | CQ both detected | "
            "Leads both | Gate |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in gate.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {_pass(row.metric_easy_baseline_clean)} | "
            f"{_pass(row.detected_both_transitions)} | "
            f"{_pass(row.cq_detected_both_transitions)} | "
            f"{_pass(row.leads_cq_both_transitions)} | "
            f"{_pass(row.early_warning_gate_pass)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "At least one preregistered non-CQ metric gives reproducible early "
                "warning before CQ across both transitions. This is single-model "
                "method evidence, not a cross-model confirmation."
                if overall_pass
                else "No preregistered non-CQ metric satisfies the complete two-transition "
                "early-warning rule. Do not retune detector thresholds or select a "
                "single favorable transition after this result."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _display(value: Any, digits: int | None = None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if digits is not None:
        return f"{float(value):.{digits}f}"
    return str(int(value))


def _display_decimal(value: Any, digits: int) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _pass(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def analyze_e2h_gpt_detectors(
    result_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Apply the frozen Section 49 analysis to saved GPT holdout outputs."""
    result_dir = Path(result_dir)
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_detector_analysis_config(config)

    data_dir = result_dir / "data"
    gate_path = data_dir / "e2h_final_holdout_gate.csv"
    source_manifest_path = result_dir / "e2h_final_holdout_manifest.json"
    per_query_path = data_dir / f"per_query_{TAG}.csv"
    alarms_path = data_dir / f"alarms_{TAG}.csv"
    _validate_source_gate(data_dir)

    per_query = pd.read_csv(per_query_path).sort_values("t").reset_index(drop=True)
    expected_difficulties = [band for band in ("easy", "medium", "hard") for _ in range(30)]
    if (
        len(per_query) != 90
        or per_query["difficulty"].astype(str).tolist() != expected_difficulties
    ):
        raise ValueError("Section 49 requires the frozen 30/30/30 ST2 phase order")
    alarms = pd.read_csv(alarms_path)
    required_alarm_columns = {"t", "metric", "detector", "statistic", "level"}
    if required_alarm_columns - set(alarms.columns):
        raise ValueError("Section 49 alarm file is missing required columns")

    phase_summary = _phase_summary(per_query)
    timing, gate, overall_pass = _alarm_timing(alarms)
    phase_summary.to_csv(data_dir / "section49_phase_normalized_metrics.csv", index=False)
    timing.to_csv(data_dir / "section49_alarm_timing.csv", index=False)
    gate.to_csv(data_dir / "section49_early_warning_gate.csv", index=False)

    result = {
        "analysis": ANALYSIS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "source_track_passed": True,
        "criteria": {
            "alarm_level": ALARM_LEVEL,
            "reference_metric": REFERENCE_METRIC,
            "candidate_metrics": list(CANDIDATE_METRICS),
            "descriptive_metrics": list(DESCRIPTIVE_METRICS),
            "transitions": [
                {
                    "name": name,
                    "change_point": change_point,
                    "interval_end": end,
                }
                for name, change_point, end in TRANSITIONS
            ],
            "phase_length": PHASE_LENGTH,
            "min_lead_queries": MIN_LEAD_QUERIES,
            "require_all_transitions": True,
            "require_clean_easy_baseline": True,
        },
        "overall_early_warning_gate_pass": overall_pass,
        "passing_metrics": gate.loc[gate["early_warning_gate_pass"], "metric"].tolist(),
        "source_sha256": {
            "config": _sha256(config_path),
            str(gate_path.relative_to(result_dir)): _sha256(gate_path),
            str(source_manifest_path.relative_to(result_dir)): _sha256(source_manifest_path),
            str(per_query_path.relative_to(result_dir)): _sha256(per_query_path),
            str(alarms_path.relative_to(result_dir)): _sha256(alarms_path),
        },
    }
    (result_dir / "section49_detector_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(
        result_dir / "SECTION49_GPT_DETECTOR_ANALYSIS.md",
        phase_summary,
        timing,
        gate,
        overall_pass,
    )
    return result
