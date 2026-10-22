"""Frozen post-run assessment for the paid proposal-S1 K/L validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.types import METRIC_ORDER


def repeat_diversity_rate(records: list[dict], k_repeats: int) -> tuple[int, int, float]:
    """Fraction of queries with at least two distinct verbatim K completions."""
    runs_by_qid: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("role") == "run":
            runs_by_qid[str(record["qid"])].append(record)
    diverse = 0
    complete = 0
    for runs in runs_by_qid.values():
        if len(runs) != k_repeats:
            continue
        complete += 1
        if len({str(record.get("raw_text", "")) for record in runs}) >= 2:
            diverse += 1
    rate = diverse / complete if complete else 0.0
    return diverse, complete, rate


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _write_assessment(path: Path, gate: pd.DataFrame, overall_pass: bool) -> None:
    lines = [
        "# Proposal S1 K/L Metric-Health Assessment",
        "",
        "This report applies the criteria frozen in DEVIATIONS_FROM_PROPOSAL.md ",
        "Section 37. CQ is the only calibrated detector; other alarm times are not ",
        "used by this gate.",
        "",
        "| Model | Records | Queries | Failed | K-diversity | RS N | Coverage | Range | Nondegenerate | CQ delay | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in gate.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.n_records}/{row.expected_records} | "
            f"{row.n_per_query_rows}/{row.expected_queries} | {row.failed_calls} | "
            f"{row.k_repeat_diversity_rate:.3f} | {row.rs_evaluable_n} | "
            f"{'pass' if row.metric_coverage_pass else 'fail'} | "
            f"{'pass' if row.metrics_in_range else 'fail'} | "
            f"{'pass' if row.nondegenerate else 'fail'} | "
            f"{'pass' if row.cq_delay_gate_pass else 'fail'} | "
            f"{'PASS' if row.gate_pass else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        (
            "All frozen metric-health criteria pass. Saved records may now be used for "
            "call-free detector calibration; this does not make the run confirmatory."
            if overall_pass
            else
            "At least one frozen metric-health criterion failed. Stop progression and "
            "do not relax thresholds after inspecting these outcomes."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_s1_kl_validation(result_dir: str | Path) -> dict[str, Any]:
    """Assess a completed paid S1 K/L run without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if config.get("stream_design") != "proposal_multidataset":
        raise ValueError("result is not a proposal-multidataset run")
    if config.get("scenarios") != ["S1"]:
        raise ValueError("result must contain only S1")
    if int(config.get("k_repeats", 0)) != 5 or int(config.get("l_paraphrases", -1)) != 3:
        raise ValueError("Section 37 requires K=5 and L=3")
    cq_override = (config.get("detector_overrides") or {}).get("CQ", {})
    if float(cq_override.get("threshold", -1)) != 7.5:
        raise ValueError("Section 37 requires CQ-CUSUM threshold 7.5")
    if int(config.get("proposal_s1_max_cq_delay", -1)) != 10:
        raise ValueError("Section 37 requires proposal_s1_max_cq_delay=10")

    seeds = config.get("seeds") or [config.get("seed")]
    windows = config.get("window_sizes") or []
    models = config.get("models") or []
    if len(seeds) != 1 or len(windows) != 1 or len(models) != 2:
        raise ValueError("Section 37 requires one seed, one window and two models")
    seed, window = int(seeds[0]), int(windows[0])
    queries_per_model = 3 * int(config["proposal_queries_per_dataset"])
    calls_per_query = int(config["k_repeats"]) + int(config["l_paraphrases"])
    expected_records = queries_per_model * calls_per_query
    data_dir = result_dir / "data"
    delay_gate = pd.read_csv(data_dir / "proposal_s1_cq_delay_gate.csv")

    preflight_pass = not manifest.get("preflight", {}).get("problems")
    degenerate = manifest.get("degenerate_metrics") or {}
    failed_manifest = manifest.get("failed_calls") or {}
    gate_rows = []
    source_hashes = {}
    for model in models:
        model_name = str(model["name"])
        tag = f"S1_{model_name}_seed{seed}_W{window}"
        records_path = data_dir / f"records_{tag}.jsonl"
        per_query_path = data_dir / f"per_query_{tag}.csv"
        records = _read_jsonl(records_path)
        per_query = pd.read_csv(per_query_path)
        source_hashes[model_name] = {
            "records_sha256": _sha256(records_path),
            "per_query_sha256": _sha256(per_query_path),
        }

        role_counts_ok = True
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            grouped[str(record["qid"])].append(record)
        for query_records in grouped.values():
            runs = [record for record in query_records if record.get("role") == "run"]
            paraphrases = [
                record for record in query_records if record.get("role") == "paraphrase"
            ]
            if len(runs) != 5 or len(paraphrases) != 3:
                role_counts_ok = False
                break

        diverse, complete, diversity_rate = repeat_diversity_rate(records, 5)
        failed_calls = sum(not bool(record.get("ok")) for record in records)
        per_query_failed_calls = (
            sum(_is_true(value) for value in per_query["call_failed"])
            if "call_failed" in per_query
            else len(per_query)
        )
        metrics_in_range = True
        metric_non_nan: dict[str, int] = {}
        for metric in METRIC_ORDER:
            column = f"{metric}_raw"
            if column not in per_query:
                metrics_in_range = False
                metric_non_nan[metric] = 0
                continue
            values = [float(value) for value in per_query[column].dropna().tolist()]
            metric_non_nan[metric] = len(values)
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                metrics_in_range = False

        metric_coverage_pass = all(
            count >= 20 if metric == "RS" else count == queries_per_model
            for metric, count in metric_non_nan.items()
        ) and set(metric_non_nan) == set(METRIC_ORDER)

        delay_rows = delay_gate[delay_gate["model"] == model_name]
        cq_delay_gate_pass = bool(
            len(delay_rows) == 1 and _is_true(delay_rows.iloc[0]["gate_pass"])
        )
        nondegenerate = tag not in degenerate
        row = {
            "model": model_name,
            "n_records": len(records),
            "expected_records": expected_records,
            "n_queries": len(grouped),
            "expected_queries": queries_per_model,
            "n_per_query_rows": len(per_query),
            "role_counts_ok": role_counts_ok,
            "failed_calls": failed_calls,
            "per_query_failed_calls": per_query_failed_calls,
            "manifest_failed_calls": int(failed_manifest.get(tag, -1)),
            "k_repeat_diverse_queries": diverse,
            "k_repeat_complete_queries": complete,
            "k_repeat_diversity_rate": diversity_rate,
            "k_repeat_diversity_pass": diversity_rate >= 0.80,
            "rs_evaluable_n": metric_non_nan.get("RS", 0),
            "rs_coverage_pass": metric_non_nan.get("RS", 0) >= 20,
            "metric_coverage_pass": metric_coverage_pass,
            "metrics_in_range": metrics_in_range,
            "nondegenerate": nondegenerate,
            "preflight_pass": preflight_pass,
            "cq_delay_gate_pass": cq_delay_gate_pass,
        }
        row["gate_pass"] = all([
            row["n_records"] == expected_records,
            row["n_queries"] == queries_per_model,
            row["n_per_query_rows"] == queries_per_model,
            row["role_counts_ok"],
            row["failed_calls"] == 0,
            row["per_query_failed_calls"] == 0,
            row["manifest_failed_calls"] == 0,
            row["k_repeat_diversity_pass"],
            row["rs_coverage_pass"],
            row["metric_coverage_pass"],
            row["metrics_in_range"],
            row["nondegenerate"],
            row["preflight_pass"],
            row["cq_delay_gate_pass"],
        ])
        gate_rows.append(row)

    gate = pd.DataFrame(gate_rows)
    gate.to_csv(data_dir / "kl_validation_gate.csv", index=False)
    overall_pass = bool(not gate.empty and gate["gate_pass"].all())
    assessment_manifest = {
        "analysis": "proposal_s1_kl_metric_health_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_hashes": source_hashes,
        "no_model_calls": True,
        "criteria": {
            "expected_records": 2 * expected_records,
            "min_k_repeat_diversity_rate": 0.80,
            "min_rs_evaluable_per_model": 20,
            "required_non_rs_metric_coverage_per_model": queries_per_model,
            "max_cq_delay": 10,
            "require_zero_failed_calls": True,
            "require_no_degenerate_metrics": True,
        },
        "overall_gate_pass": overall_pass,
    }
    (result_dir / "kl_validation_manifest.json").write_text(
        json.dumps(assessment_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(
        result_dir / "KL_VALIDATION_ASSESSMENT.md", gate, overall_pass
    )
    return assessment_manifest
