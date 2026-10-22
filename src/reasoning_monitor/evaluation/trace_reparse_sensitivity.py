"""Section 51 call-free GPT LS/SS trace-reparse sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reasoning_monitor.detectors.assignment import assign_detector
from reasoning_monitor.metrics.rqeval_adapters import LSAdapter, SSAdapter
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.types import Query, QueryObservation, Sample

ANALYSIS = "section51_gpt_trace_reparse_sensitivity"
MODEL = "gpt4o_mini"
DATASET = "e2h_amc"
TAG = "ST2_gpt4o_mini_seed48_W10"
METRICS = ("LS", "SS")
WINDOW_SIZE = 10
TRANSITIONS = (
    ("easy_to_medium", 30, 60),
    ("medium_to_hard", 60, 90),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_trace_reparse_config(config: dict[str, Any]) -> None:
    expected = {
        "analysis": ANALYSIS,
        "source_result_dir": "results/e2h_final_holdout_gpt4o_mini",
        "model": MODEL,
        "dataset": DATASET,
        "tag": TAG,
        "metrics": list(METRICS),
        "parser": "structured_paragraphs_v1",
        "paragraph_boundary": "blank_lines",
        "merge_display_math_with_previous": True,
        "fallback_sentence_boundary": "punctuation_followed_by_whitespace",
        "max_segments": None,
        "preserve_final_answer": True,
        "window_size": WINDOW_SIZE,
        "detector_strategy": "proposal_mapping",
        "refit_detectors": False,
        "reference_metric": "CQ",
        "phase_length": 30,
        "transitions": [
            {"name": name, "change_point": change_point, "interval_end": end}
            for name, change_point, end in TRANSITIONS
        ],
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"result does not match frozen Section 51 config: {mismatches}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample(record: dict[str, Any]) -> Sample:
    raw = str(record.get("raw_text") or "")
    return Sample(
        answer=str(record.get("answer") or ""),
        reasoning_steps=reasoning_steps_from_text(raw),
        latency_s=float(record.get("latency_s") or 0.0),
        n_tokens=int(record.get("n_tokens") or 0),
        raw_text=raw,
        ok=bool(record.get("ok")),
        finish_reason=record.get("finish_reason"),
    )


def _observations(
    per_query: pd.DataFrame,
    records: list[dict[str, Any]],
) -> list[QueryObservation]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["qid"]), []).append(record)

    observations = []
    for row in per_query.sort_values("t").itertuples(index=False):
        qid = str(row.qid)
        group = grouped[qid]
        primary = next(
            record
            for record in group
            if record.get("role") == "run" and int(record.get("call_index", -1)) == 0
        )
        solution_steps = getattr(row, "solution_steps", None)
        query = Query(
            qid=qid,
            t=int(row.t),
            dataset=str(row.dataset),
            domain=str(primary.get("domain") or ""),
            prompt=str(primary.get("prompt") or ""),
            gold_answer=str(primary.get("gold_answer") or ""),
            difficulty=str(row.difficulty),
            solution_steps=(None if pd.isna(solution_steps) else int(solution_steps)),
            difficulty_rating=float(row.difficulty_rating),
            difficulty_rating_std=float(row.difficulty_rating_std),
            difficulty_quantile=float(row.difficulty_quantile),
            difficulty_model_avg_acc=(
                None
                if pd.isna(row.difficulty_model_avg_acc)
                else float(row.difficulty_model_avg_acc)
            ),
        )
        ordered = sorted(group, key=lambda item: int(item.get("call_index", -1)))
        runs = [_sample(record) for record in ordered if record.get("role") == "run"]
        paraphrases = [
            _sample(record) for record in ordered if record.get("role") == "paraphrase"
        ]
        if len(runs) != 5 or len(paraphrases) != 3:
            raise ValueError(f"incomplete Section 51 records for {qid}")
        observations.append(QueryObservation(query, runs, paraphrases))
    return observations


def _load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[int(item["t"])] = item
    return rows


def _score_corrected(
    observations: list[QueryObservation],
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    saved = _load_checkpoint(checkpoint_path)
    missing = [obs for obs in observations if obs.query.t not in saved]
    statuses: dict[str, Any] = {"checkpoint_rows_reused": len(saved)}
    if missing:
        metrics = {
            "LS": LSAdapter(WINDOW_SIZE),
            "SS": SSAdapter(WINDOW_SIZE, k_runs=5),
        }
        statuses["scorers"] = {
            name: metric.scorer_status() for name, metric in metrics.items()
        }
        if not all(status["ready"] for status in statuses["scorers"].values()):
            raise RuntimeError(f"Section 51 requires real NLI and BERTScore: {statuses}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as handle:
            for obs in missing:
                item = {
                    "t": obs.query.t,
                    "qid": obs.query.qid,
                    "difficulty": obs.query.difficulty,
                    "primary_segment_count": len(obs.primary.reasoning_steps),
                    "min_k_segment_count": min(len(run.reasoning_steps) for run in obs.runs),
                    "max_k_segment_count": max(len(run.reasoning_steps) for run in obs.runs),
                    "LS_corrected": metrics["LS"].per_query_value(obs),
                    "SS_corrected": metrics["SS"].per_query_value(obs),
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                saved[obs.query.t] = item
    else:
        statuses["scorers"] = "not_loaded_checkpoint_complete"
    frame = pd.DataFrame(saved.values()).sort_values("t").reset_index(drop=True)
    if len(frame) != 90:
        raise ValueError(f"Section 51 expected 90 corrected rows, found {len(frame)}")
    return frame, statuses


def _replay_detectors(corrected: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "LS": LSAdapter(WINDOW_SIZE),
        "SS": SSAdapter(WINDOW_SIZE, k_runs=5),
    }
    detectors = {
        name: assign_detector(metric, strategy="proposal_mapping")
        for name, metric in metrics.items()
    }
    buffers = {name: deque(maxlen=WINDOW_SIZE) for name in METRICS}
    rows = []
    for item in corrected.sort_values("t").itertuples(index=False):
        for name in METRICS:
            value = float(getattr(item, f"{name}_corrected"))
            buffers[name].append(value)
            if len(buffers[name]) < WINDOW_SIZE:
                continue
            window_mean = sum(buffers[name]) / WINDOW_SIZE
            alarm = detectors[name].update(window_mean, int(item.t), name)
            if alarm is not None:
                rows.append(
                    {
                        "source": "corrected",
                        "t": alarm.t,
                        "metric": alarm.metric,
                        "detector": alarm.detector,
                        "statistic": alarm.statistic,
                        "level": alarm.level,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["source", "t", "metric", "detector", "statistic", "level"],
    )


def _phase_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        old = frame[f"{metric}_old"]
        corrected = frame[f"{metric}_corrected"]
        for phase in ("easy", "medium", "hard"):
            selected = frame["difficulty"] == phase
            old_mean = float(old[selected].mean())
            corrected_mean = float(corrected[selected].mean())
            rows.append(
                {
                    "metric": metric,
                    "phase": phase,
                    "old_mean": old_mean,
                    "corrected_mean": corrected_mean,
                    "corrected_minus_old": corrected_mean - old_mean,
                }
            )
    return pd.DataFrame(rows)


def _associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        old = frame[f"{metric}_old"]
        corrected = frame[f"{metric}_corrected"]
        rows.append(
            {
                "metric": metric,
                "pearson_old_corrected": float(old.corr(corrected, method="pearson")),
                "spearman_old_corrected": float(old.corr(corrected, method="spearman")),
                "mean_absolute_change": float((corrected - old).abs().mean()),
                "max_absolute_change": float((corrected - old).abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _first_alarm(
    alarms: pd.DataFrame,
    metric: str,
    change_point: int,
    interval_end: int,
) -> int | None:
    selected = alarms.loc[
        (alarms["metric"] == metric)
        & (alarms["level"] == "drift")
        & (alarms["t"] > change_point)
        & (alarms["t"] <= interval_end)
    ]
    return None if selected.empty else int(selected["t"].min())


def _timing_and_gate(
    original_alarms: pd.DataFrame,
    corrected_alarms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = {
        "old": original_alarms.loc[original_alarms["metric"].isin(("CQ", *METRICS))],
        "corrected": pd.concat(
            [
                original_alarms.loc[original_alarms["metric"] == "CQ"].assign(
                    source="corrected"
                ),
                corrected_alarms,
            ],
            ignore_index=True,
        ),
    }
    timing_rows = []
    gate_rows = []
    for source, alarms in sources.items():
        alarms = alarms.copy()
        alarms["t"] = pd.to_numeric(alarms["t"], errors="raise").astype(int)
        cq_baseline_clean = not bool(
            (
                (alarms["metric"] == "CQ") & (alarms["level"] == "drift") & (alarms["t"] <= 30)
            ).any()
        )
        for metric in METRICS:
            baseline_clean = not bool(
                (
                    (alarms["metric"] == metric)
                    & (alarms["level"] == "drift")
                    & (alarms["t"] <= 30)
                ).any()
            )
            metric_detected = []
            cq_detected = []
            leads = []
            for transition, change_point, interval_end in TRANSITIONS:
                alarm_t = _first_alarm(alarms, metric, change_point, interval_end)
                cq_t = _first_alarm(alarms, "CQ", change_point, interval_end)
                lead = None if alarm_t is None or cq_t is None else cq_t - alarm_t
                timing_rows.append(
                    {
                        "source": source,
                        "transition": transition,
                        "metric": metric,
                        "alarm_t": alarm_t,
                        "cq_alarm_t": cq_t,
                        "lead_over_cq_queries": lead,
                    }
                )
                metric_detected.append(alarm_t is not None)
                cq_detected.append(cq_t is not None)
                leads.append(lead is not None and lead >= 1)
            gate_rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "cq_easy_baseline_clean": cq_baseline_clean,
                    "metric_easy_baseline_clean": baseline_clean,
                    "detected_both_transitions": all(metric_detected),
                    "cq_detected_both_transitions": all(cq_detected),
                    "leads_cq_both_transitions": all(leads),
                    "early_warning_gate_pass": all(
                        [
                            cq_baseline_clean,
                            baseline_clean,
                            all(metric_detected),
                            all(cq_detected),
                            all(leads),
                        ]
                    ),
                }
            )
    return pd.DataFrame(timing_rows), pd.DataFrame(gate_rows)


def _display(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    phase: pd.DataFrame,
    associations: pd.DataFrame,
    timing: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    lines = [
        "# Section 51 GPT Trace-Reparse Sensitivity",
        "",
        "No model calls were issued. LS/SS use structured_paragraphs_v1 with the",
        "same rqeval scorers, W=10 and frozen proposal detectors.",
        "",
        "## Phase Means",
        "",
        "| Metric | Phase | Old | Corrected | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in phase.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.phase} | {_display(row.old_mean)} | "
            f"{_display(row.corrected_mean)} | {_display(row.corrected_minus_old)} |"
        )
    lines.extend(
        [
            "",
            "## Value Association",
            "",
            "| Metric | Pearson | Spearman | Mean abs change | Max abs change |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in associations.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {_display(row.pearson_old_corrected)} | "
            f"{_display(row.spearman_old_corrected)} | "
            f"{_display(row.mean_absolute_change)} | "
            f"{_display(row.max_absolute_change)} |"
        )
    lines.extend(
        [
            "",
            "## Alarm Timing",
            "",
            "| Source | Transition | Metric | Alarm t | CQ t | CQ lead |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in timing.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.transition} | {row.metric} | "
            f"{_display(row.alarm_t, 0)} | {_display(row.cq_alarm_t, 0)} | "
            f"{_display(row.lead_over_cq_queries, 0)} |"
        )
    corrected_gate = gate.loc[gate["source"] == "corrected"]
    corrected_pass = bool(corrected_gate["early_warning_gate_pass"].any())
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "At least one corrected trace metric passes the original two-transition "
                "timing rule. Treat this only as post-diagnostic sensitivity evidence."
                if corrected_pass
                else "Neither corrected trace metric passes the original two-transition "
                "timing rule. The parser materially changes values but does not rescue "
                "the frozen early-warning claim."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_trace_reparse_sensitivity(
    result_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_trace_reparse_config(config)

    data_dir = result_dir / "data"
    records_path = data_dir / f"records_{TAG}.jsonl"
    per_query_path = data_dir / f"per_query_{TAG}.csv"
    alarms_path = data_dir / f"alarms_{TAG}.csv"
    checkpoint_path = data_dir / "section51_trace_reparse_checkpoint.jsonl"
    records = _read_jsonl(records_path)
    per_query = pd.read_csv(per_query_path).sort_values("t").reset_index(drop=True)
    if len(per_query) != 90:
        raise ValueError("Section 51 requires exactly 90 source queries")
    observations = _observations(per_query, records)
    corrected, scorer_status = _score_corrected(observations, checkpoint_path)

    merged = (
        per_query[["t", "qid", "difficulty", "LS_raw", "SS_raw"]]
        .rename(columns={"LS_raw": "LS_old", "SS_raw": "SS_old"})
        .merge(corrected, on=["t", "qid", "difficulty"], validate="one_to_one")
    )
    corrected_alarms = _replay_detectors(corrected)
    original_alarms = pd.read_csv(alarms_path)
    timing, gate = _timing_and_gate(original_alarms, corrected_alarms)
    phase = _phase_comparison(merged)
    associations = _associations(merged)

    merged.to_csv(data_dir / "section51_trace_reparse_per_query.csv", index=False)
    phase.to_csv(data_dir / "section51_trace_reparse_phase_comparison.csv", index=False)
    associations.to_csv(data_dir / "section51_trace_reparse_associations.csv", index=False)
    corrected_alarms.to_csv(data_dir / "section51_trace_reparse_alarms.csv", index=False)
    timing.to_csv(data_dir / "section51_trace_reparse_timing.csv", index=False)
    gate.to_csv(data_dir / "section51_trace_reparse_gate.csv", index=False)

    corrected_gate = gate.loc[gate["source"] == "corrected"]
    result = {
        "analysis": ANALYSIS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "scorer_status": scorer_status,
        "corrected_early_warning_gate_pass": bool(
            corrected_gate["early_warning_gate_pass"].any()
        ),
        "corrected_passing_metrics": corrected_gate.loc[
            corrected_gate["early_warning_gate_pass"], "metric"
        ].tolist(),
        "source_sha256": {
            "config": _sha256(config_path),
            str(records_path.relative_to(result_dir)): _sha256(records_path),
            str(per_query_path.relative_to(result_dir)): _sha256(per_query_path),
            str(alarms_path.relative_to(result_dir)): _sha256(alarms_path),
        },
    }
    (result_dir / "section51_trace_reparse_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        result_dir / "SECTION51_TRACE_REPARSE_SENSITIVITY.md",
        phase,
        associations,
        timing,
        gate,
    )
    return result
