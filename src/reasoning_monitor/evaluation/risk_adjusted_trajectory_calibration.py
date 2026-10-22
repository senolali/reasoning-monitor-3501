"""Frozen operational and event-density gate for RAT-v1 calibration outputs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (
    PRIMARY_METRICS,
    build_risk_set,
    risk_set_summary,
)
from reasoning_monitor.metrics.dimensions import answers_equal

_FINAL_INTEGER = re.compile(
    r"^FINAL_ANSWER:\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*$"
)
_NUMBERED_LINE = re.compile(r"^\s*\d+\.\s+\S")


def strict_final_integer(raw_text: str) -> str | None:
    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]
    if not lines:
        return None
    match = _FINAL_INTEGER.fullmatch(lines[-1])
    return match.group(1) if match else None


def reasoning_line_count(raw_text: str) -> int:
    lines = [line for line in str(raw_text).splitlines() if line.strip()]
    if lines and _FINAL_INTEGER.fullmatch(lines[-1].strip()):
        lines = lines[:-1]
    return sum(bool(_NUMBERED_LINE.match(line)) for line in lines)


def assess_calibration_model(
    model: str,
    records: list[dict[str, Any]],
    per_query: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Assess one preregistered model without changing calibration data."""
    frame = per_query.copy()
    frame["model"] = model
    strict_answers = [strict_final_integer(record.get("raw_text", "")) for record in records]
    observable = [answer is not None for answer in strict_answers]
    agreement = [
        answer is not None
        and answers_equal(answer, str(record.get("answer", "")), "controlled_trajectory")
        for answer, record in zip(strict_answers, records, strict=True)
    ]
    role_counts = Counter(str(record.get("role")) for record in records)
    truncations = sum(
        str(record.get("finish_reason") or "").lower() == "length"
        for record in records
    )
    step_violations = sum(
        reasoning_line_count(record.get("raw_text", "")) > int(gate["max_reasoning_lines"])
        for record in records
    )

    problems: list[str] = []
    try:
        risk = build_risk_set(frame)
        risk_summary = risk_set_summary(risk)
    except ValueError as exc:
        risk = pd.DataFrame()
        risk_summary = risk_set_summary(risk)
        problems.append(str(exc))

    level_one = frame[frame["trajectory_level"] == 1]
    level_one_correct = int(level_one["CQ_raw"].sum()) if not level_one.empty else 0
    family_events = (
        risk.groupby("family_id")["failure_next"].max()
        if not risk.empty
        else pd.Series(dtype=int)
    )
    censored = int((family_events == 0).sum())
    event_templates = int(
        risk.loc[risk.get("failure_next", pd.Series(dtype=int)) == 1, "task_template"].nunique()
    ) if not risk.empty else 0
    metric_distinct = {
        metric: int(risk[metric].nunique(dropna=True)) if metric in risk else 0
        for metric in PRIMARY_METRICS
    }

    n_records = len(records)
    success_rate = (
        sum(bool(record.get("ok")) for record in records) / n_records
        if n_records
        else 0.0
    )
    marker_rate = sum(observable) / n_records if n_records else 0.0
    agreement_rate = sum(agreement) / n_records if n_records else 0.0
    expected_roles = {
        "run": int(gate["expected_queries_per_model"]),
        "paraphrase": int(gate["expected_queries_per_model"])
        * int(gate["expected_l_paraphrases"]),
    }
    checks = {
        "record_count": n_records == int(gate["expected_records_per_model"]),
        "role_counts": dict(role_counts) == expected_roles,
        "per_query_count": len(frame) == int(gate["expected_queries_per_model"]),
        "family_count": frame["family_id"].nunique() == int(gate["expected_families"]),
        "calibration_only": set(frame["trajectory_split"].dropna()) == {"calibration"},
        "execution": success_rate >= float(gate["minimum_success_rate"]),
        "strict_marker": marker_rate >= float(gate["minimum_final_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "step_contract": step_violations
        <= int(gate["maximum_step_contract_violations"]),
        "level_one_entry": level_one_correct >= int(gate["minimum_level_one_correct"]),
        "minimum_events": risk_summary["events"] >= int(gate["minimum_first_failures"]),
        "minimum_censored": censored >= int(gate["minimum_censored_families"]),
        "event_template_coverage": event_templates
        >= int(gate["minimum_event_templates"]),
        "predictor_finite": not risk.empty
        and np.isfinite(risk[list(PRIMARY_METRICS)].to_numpy(dtype=float)).all(),
        "predictor_nondegenerate": all(
            count >= int(gate["minimum_distinct_predictor_values"])
            for count in metric_distinct.values()
        ),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    problems.extend(name for name, passed in checks.items() if not passed)
    return {
        "model": model,
        "n_records": n_records,
        "role_counts": dict(role_counts),
        "n_per_query": int(len(frame)),
        "success_rate": success_rate,
        "final_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": int(truncations),
        "step_contract_violations": int(step_violations),
        "level_one_correct": level_one_correct,
        "risk_set": risk_summary,
        "censored_families": censored,
        "event_templates": event_templates,
        "metric_distinct_values": metric_distinct,
        "checks": checks,
        "problems": problems,
        "pass": not problems,
    }
