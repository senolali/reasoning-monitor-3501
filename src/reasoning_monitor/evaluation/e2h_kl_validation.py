"""Frozen post-run gate for the Llama-only E2H solution-step K/L pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.evaluation.kl_validation import repeat_diversity_rate
from reasoning_monitor.types import METRIC_ORDER

BANDS = ("easy", "medium", "hard")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def cq_band_gate(per_query: pd.DataFrame, queries_per_band: int = 30) -> dict[str, Any]:
    """Apply the frozen monotonic CQ and adjacent-gap rule."""
    ordered = per_query.sort_values("t").reset_index(drop=True)
    expected = [band for band in BANDS for _ in range(queries_per_band)]
    labels = ordered["difficulty"].astype(str).str.lower().tolist()
    means = {
        band: float(ordered.loc[ordered["difficulty"] == band, "CQ_raw"].mean())
        for band in BANDS
    }
    easy_medium = means["easy"] - means["medium"]
    medium_hard = means["medium"] - means["hard"]
    monotonic = means["easy"] > means["medium"] > means["hard"]
    return {
        "phase_order_pass": labels == expected,
        "easy_cq": means["easy"],
        "medium_cq": means["medium"],
        "hard_cq": means["hard"],
        "easy_medium_gap": easy_medium,
        "medium_hard_gap": medium_hard,
        "strictly_monotonic": monotonic,
        "adjacent_gap_pass": easy_medium >= 0.10 and medium_hard >= 0.10,
        "cq_gate_pass": monotonic and easy_medium >= 0.10 and medium_hard >= 0.10,
    }


def _write_assessment(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# E2H Llama K/L Pilot Assessment",
        "",
        "This report applies the criteria frozen in DEVIATIONS_FROM_PROPOSAL.md ",
        "Section 40. Detector alarms are not used by this pilot gate.",
        "",
        "## Completion And Metric Health",
        "",
        f"- Records: {row['n_records']}/{row['expected_records']}",
        f"- Queries: {row['n_per_query_rows']}/{row['expected_queries']}",
        f"- Failed calls: {row['failed_calls']}",
        f"- K-repeat diversity: {row['k_repeat_diversity_rate']:.3f}",
        f"- RS evaluable queries: {row['rs_evaluable_n']}",
        f"- Metric-health gate: {'PASS' if row['metric_health_pass'] else 'FAIL'}",
        "",
        "## Frozen CQ Band Gate",
        "",
        "| Easy CQ | Medium CQ | Hard CQ | E-M gap | M-H gap | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {row['easy_cq']:.3f} | {row['medium_cq']:.3f} | "
            f"{row['hard_cq']:.3f} | {row['easy_medium_gap']:.3f} | "
            f"{row['medium_hard_gap']:.3f} | "
            f"{'PASS' if row['cq_gate_pass'] else 'FAIL'} |"
        ),
        "",
        "## Decision",
        "",
        (
            "The pilot gate passes. Saved records may proceed to the separately "
            "preregistered call-free detector calibration. This is not confirmatory."
            if row["overall_gate_pass"]
            else
            "The pilot gate fails. Do not tune solution-step bands or detector "
            "parameters against these ordered outcomes."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_llama_kl_pilot(result_dir: str | Path) -> dict[str, Any]:
    """Assess the completed paid pilot without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if config.get("stream_design") != "e2h_gsm8k":
        raise ValueError("result is not an E2H-GSM8K run")
    if config.get("scenarios") != ["ST2"]:
        raise ValueError("Section 40 requires only ST2")
    if config.get("difficulty_scheme") != "solution_steps":
        raise ValueError("Section 40 requires solution_steps difficulty bands")
    if int(config.get("queries_per_regime", 0)) != 30:
        raise ValueError("Section 40 requires 30 queries per regime")
    if int(config.get("k_repeats", 0)) != 5 or int(config.get("l_paraphrases", -1)) != 3:
        raise ValueError("Section 40 requires K=5 and L=3")
    if config.get("detector_strategy") != "proposal_mapping":
        raise ValueError("Section 40 requires proposal_mapping")

    seeds = config.get("seeds") or [config.get("seed")]
    models = config.get("models") or []
    windows = config.get("window_sizes") or []
    if seeds != [43] or windows != [10] or len(models) != 1:
        raise ValueError("Section 40 requires seed 43, W=10 and one model")
    model_name = str(models[0]["name"])
    if model_name != "llama3_8b":
        raise ValueError("Section 40 requires llama3_8b")

    expected_queries = 90
    expected_records = 720
    tag = "ST2_llama3_8b_seed43_W10"
    data_dir = result_dir / "data"
    records_path = data_dir / f"records_{tag}.jsonl"
    per_query_path = data_dir / f"per_query_{tag}.csv"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per_query = pd.read_csv(per_query_path)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)
    role_counts_ok = all(
        sum(record.get("role") == "run" for record in query_records) == 5
        and sum(record.get("role") == "paraphrase" for record in query_records) == 3
        for query_records in grouped.values()
    )
    step_band_pass = True
    for query_records in grouped.values():
        steps = {int(record.get("solution_steps") or 0) for record in query_records}
        difficulties = {str(record.get("difficulty")) for record in query_records}
        if len(steps) != 1 or len(difficulties) != 1:
            step_band_pass = False
            break
        step = next(iter(steps))
        difficulty = next(iter(difficulties))
        valid = (
            difficulty == "easy" and 1 <= step <= 2
            or difficulty == "medium" and 3 <= step <= 4
            or difficulty == "hard" and step >= 5
        )
        if not valid:
            step_band_pass = False
            break

    failed_calls = sum(not bool(record.get("ok")) for record in records)
    per_query_failed_calls = (
        sum(_is_true(value) for value in per_query["call_failed"])
        if "call_failed" in per_query
        else len(per_query)
    )
    diverse, complete, diversity_rate = repeat_diversity_rate(records, 5)
    metric_non_nan: dict[str, int] = {}
    metrics_in_range = True
    for metric in METRIC_ORDER:
        column = f"{metric}_raw"
        if column not in per_query:
            metric_non_nan[metric] = 0
            metrics_in_range = False
            continue
        values = [float(value) for value in per_query[column].dropna().tolist()]
        metric_non_nan[metric] = len(values)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            metrics_in_range = False
    metric_coverage_pass = all(
        count >= 20 if metric == "RS" else count == expected_queries
        for metric, count in metric_non_nan.items()
    ) and set(metric_non_nan) == set(METRIC_ORDER)

    cq_gate = cq_band_gate(per_query, 30)
    preflight_pass = not manifest.get("preflight", {}).get("problems")
    nondegenerate = tag not in (manifest.get("degenerate_metrics") or {})
    manifest_failed = int((manifest.get("failed_calls") or {}).get(tag, -1))
    metric_health_pass = all([
        len(records) == expected_records,
        len(grouped) == expected_queries,
        len(per_query) == expected_queries,
        role_counts_ok,
        step_band_pass,
        failed_calls == 0,
        per_query_failed_calls == 0,
        manifest_failed == 0,
        complete == expected_queries,
        diversity_rate >= 0.80,
        metric_non_nan.get("RS", 0) >= 20,
        metric_coverage_pass,
        metrics_in_range,
        nondegenerate,
        preflight_pass,
        cq_gate["phase_order_pass"],
    ])
    row = {
        "model": model_name,
        "n_records": len(records),
        "expected_records": expected_records,
        "n_queries": len(grouped),
        "n_per_query_rows": len(per_query),
        "expected_queries": expected_queries,
        "role_counts_ok": role_counts_ok,
        "step_band_pass": step_band_pass,
        "failed_calls": failed_calls,
        "per_query_failed_calls": per_query_failed_calls,
        "manifest_failed_calls": manifest_failed,
        "k_repeat_diverse_queries": diverse,
        "k_repeat_complete_queries": complete,
        "k_repeat_diversity_rate": diversity_rate,
        "k_repeat_diversity_pass": diversity_rate >= 0.80,
        "rs_evaluable_n": metric_non_nan.get("RS", 0),
        "metric_coverage_pass": metric_coverage_pass,
        "metrics_in_range": metrics_in_range,
        "nondegenerate": nondegenerate,
        "preflight_pass": preflight_pass,
        "metric_health_pass": metric_health_pass,
        **cq_gate,
    }
    row["overall_gate_pass"] = metric_health_pass and bool(cq_gate["cq_gate_pass"])
    gate = pd.DataFrame([row])
    gate.to_csv(data_dir / "e2h_kl_pilot_gate.csv", index=False)

    result = {
        "analysis": "e2h_solution_step_llama_kl_pilot_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_records_sha256": _sha256(records_path),
        "source_per_query_sha256": _sha256(per_query_path),
        "no_model_calls": True,
        "criteria": {
            "expected_records": expected_records,
            "min_k_repeat_diversity_rate": 0.80,
            "min_rs_evaluable": 20,
            "min_adjacent_cq_gap": 0.10,
            "require_strictly_monotonic_cq": True,
            "detector_alarms_used": False,
        },
        "overall_gate_pass": bool(row["overall_gate_pass"]),
    }
    (result_dir / "e2h_kl_pilot_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(result_dir / "E2H_KL_PILOT_ASSESSMENT.md", row)
    return result
