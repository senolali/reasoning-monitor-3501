"""Frozen post-run gate for the official E2H Mistral holdout K/L run."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.data.loaders import e2h_partition
from reasoning_monitor.evaluation.e2h_benchmark_calibration import (
    permutation_slope_test,
)
from reasoning_monitor.evaluation.kl_validation import repeat_diversity_rate
from reasoning_monitor.types import METRIC_ORDER

BANDS = ("easy", "medium", "hard")
N_PER_BAND = 30
EXPECTED_QUERIES = 90
EXPECTED_RECORDS = 720
N_PERMUTATIONS = 5_000
PERMUTATION_SEED = 93_045
ALPHA = 0.05
MIN_EASY_HARD_GAP = 0.15
MIN_REPEAT_DIVERSITY = 0.80
MIN_RS_EVALUABLE = 20
TAG = "ST2_mistral_7b_seed45_W10"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def cq_holdout_gate(per_query: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen continuous-rating CQ generalisation gate."""
    ordered = per_query.sort_values("t").reset_index(drop=True)
    means = {
        band: float(ordered.loc[ordered["difficulty"] == band, "CQ_raw"].mean())
        for band in BANDS
    }
    slope, p_value = permutation_slope_test(
        ordered["difficulty_rating"].to_numpy(dtype=float),
        ordered["CQ_raw"].to_numpy(dtype=float),
        permutations=N_PERMUTATIONS,
        seed=PERMUTATION_SEED,
    )
    gap = means["easy"] - means["hard"]
    expected_order = [band for band in BANDS for _ in range(N_PER_BAND)]
    phase_order_pass = ordered["difficulty"].astype(str).tolist() == expected_order
    return {
        "phase_order_pass": phase_order_pass,
        "easy_cq": means["easy"],
        "medium_cq": means["medium"],
        "hard_cq": means["hard"],
        "easy_hard_gap": gap,
        "cq_rating_slope": slope,
        "permutation_p_one_sided": p_value,
        "negative_slope_pass": slope < 0 and p_value <= ALPHA,
        "easy_hard_gap_pass": gap >= MIN_EASY_HARD_GAP,
        "cq_holdout_gate_pass": (
            phase_order_pass
            and slope < 0
            and p_value <= ALPHA
            and gap >= MIN_EASY_HARD_GAP
        ),
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "holdout",
        "scenarios": ["ST2"],
        "seeds": [45],
        "window_sizes": [10],
        "queries_per_regime": 30,
        "k_repeats": 5,
        "l_paraphrases": 3,
        "use_rqeval": True,
        "coherence_scorer": "nli",
        "stability_scorer": "bertscore",
        "publication_mode": True,
        "detector_strategy": "proposal_mapping",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    names = [model.get("name") for model in config.get("models", [])]
    if names != ["mistral_7b"]:
        mismatches["models"] = (names, ["mistral_7b"])
    if mismatches:
        raise ValueError(f"result does not match frozen Section 42 config: {mismatches}")


def _metadata_pass(per_query: pd.DataFrame) -> bool:
    if per_query[["difficulty_rating", "difficulty_quantile"]].isna().any().any():
        return False
    for row in per_query.itertuples(index=False):
        quantile = float(row.difficulty_quantile)
        valid_band = (
            row.difficulty == "easy" and 0 <= quantile <= 1 / 3
            or row.difficulty == "medium" and 1 / 3 < quantile <= 2 / 3
            or row.difficulty == "hard" and 2 / 3 < quantile <= 1
        )
        if not valid_band or e2h_partition(str(row.qid)) != "holdout":
            return False
    return True


def _write_assessment(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# Official E2H Mistral Holdout K/L Assessment",
        "",
        "This call-free report applies the criteria frozen in Section 42. ",
        "Detector alarms are descriptive and are not used by this gate.",
        "",
        "## Completion And Metric Health",
        "",
        f"- Records: {row['n_records']}/{EXPECTED_RECORDS}",
        f"- Queries: {row['n_per_query_rows']}/{EXPECTED_QUERIES}",
        f"- Failed calls: {row['failed_calls']}",
        f"- K-repeat diversity: {row['k_repeat_diversity_rate']:.3f}",
        f"- RS evaluable queries: {row['rs_evaluable_n']}",
        f"- Metric-health gate: {'PASS' if row['metric_health_pass'] else 'FAIL'}",
        "",
        "## Frozen CQ Holdout Gate",
        "",
        "| Easy CQ | Medium CQ | Hard CQ | E-H gap | Slope | p | Gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {row['easy_cq']:.3f} | {row['medium_cq']:.3f} | "
            f"{row['hard_cq']:.3f} | {row['easy_hard_gap']:.3f} | "
            f"{row['cq_rating_slope']:.6f} | "
            f"{row['permutation_p_one_sided']:.6f} | "
            f"{'PASS' if row['cq_holdout_gate_pass'] else 'FAIL'} |"
        ),
        "",
        "## Decision",
        "",
        (
            "The holdout gate passes. The saved K/L records may proceed only to "
            "a separately preregistered call-free detector analysis."
            if row["overall_gate_pass"]
            else
            "The holdout gate fails. Stop this branch and do not tune labels, "
            "thresholds, metric rules, or test direction against these outcomes."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_mistral_holdout(result_dir: str | Path) -> dict[str, Any]:
    """Assess the completed holdout K/L run without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_config(manifest["config"])

    data_dir = result_dir / "data"
    records_path = data_dir / f"records_{TAG}.jsonl"
    per_query_path = data_dir / f"per_query_{TAG}.csv"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per_query = pd.read_csv(per_query_path)

    required = {
        "t", "qid", "difficulty", "difficulty_rating", "difficulty_quantile",
        "call_failed", *(f"{metric}_raw" for metric in METRIC_ORDER),
    }
    missing = required - set(per_query.columns)
    if missing:
        raise ValueError(f"holdout per-query file is missing columns: {sorted(missing)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)
    role_counts_ok = all(
        sum(record.get("role") == "run" for record in query_records) == 5
        and sum(record.get("role") == "paraphrase" for record in query_records) == 3
        for query_records in grouped.values()
    )
    raw_qids = set(grouped)
    per_query_qids = set(per_query["qid"].astype(str))
    qids_match = raw_qids == per_query_qids
    unique_queries = len(per_query) == len(per_query_qids) == EXPECTED_QUERIES
    holdout_metadata_pass = _metadata_pass(per_query)

    failed_calls = sum(not bool(record.get("ok")) for record in records)
    per_query_failed_calls = sum(_is_true(value) for value in per_query["call_failed"])
    manifest_failed = int((manifest.get("failed_calls") or {}).get(TAG, -1))
    diverse, complete, diversity_rate = repeat_diversity_rate(records, 5)

    metric_non_nan: dict[str, int] = {}
    metrics_in_range = True
    for metric in METRIC_ORDER:
        values = [float(value) for value in per_query[f"{metric}_raw"].dropna()]
        metric_non_nan[metric] = len(values)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            metrics_in_range = False
    metric_coverage_pass = all(
        count >= MIN_RS_EVALUABLE if metric == "RS" else count == EXPECTED_QUERIES
        for metric, count in metric_non_nan.items()
    )
    preflight_pass = not manifest.get("preflight", {}).get("problems")
    nondegenerate = TAG not in (manifest.get("degenerate_metrics") or {})
    cq_gate = cq_holdout_gate(per_query)

    metric_health_pass = all([
        len(records) == EXPECTED_RECORDS,
        len(grouped) == EXPECTED_QUERIES,
        len(per_query) == EXPECTED_QUERIES,
        unique_queries,
        qids_match,
        role_counts_ok,
        holdout_metadata_pass,
        failed_calls == 0,
        per_query_failed_calls == 0,
        manifest_failed == 0,
        complete == EXPECTED_QUERIES,
        diversity_rate >= MIN_REPEAT_DIVERSITY,
        metric_coverage_pass,
        metrics_in_range,
        nondegenerate,
        preflight_pass,
        cq_gate["phase_order_pass"],
    ])
    row = {
        "model": "mistral_7b",
        "n_records": len(records),
        "expected_records": EXPECTED_RECORDS,
        "n_queries": len(grouped),
        "n_per_query_rows": len(per_query),
        "expected_queries": EXPECTED_QUERIES,
        "unique_queries": unique_queries,
        "qids_match": qids_match,
        "role_counts_ok": role_counts_ok,
        "holdout_metadata_pass": holdout_metadata_pass,
        "failed_calls": failed_calls,
        "per_query_failed_calls": per_query_failed_calls,
        "manifest_failed_calls": manifest_failed,
        "k_repeat_diverse_queries": diverse,
        "k_repeat_complete_queries": complete,
        "k_repeat_diversity_rate": diversity_rate,
        "k_repeat_diversity_pass": diversity_rate >= MIN_REPEAT_DIVERSITY,
        "rs_evaluable_n": metric_non_nan.get("RS", 0),
        "metric_coverage_pass": metric_coverage_pass,
        "metrics_in_range": metrics_in_range,
        "nondegenerate": nondegenerate,
        "preflight_pass": preflight_pass,
        "metric_health_pass": metric_health_pass,
        **cq_gate,
    }
    row["overall_gate_pass"] = bool(
        metric_health_pass and cq_gate["cq_holdout_gate_pass"]
    )
    pd.DataFrame([row]).to_csv(
        data_dir / "e2h_mistral_holdout_gate.csv", index=False
    )

    result = {
        "analysis": "official_e2h_mistral_holdout_kl_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "criteria": {
            "expected_records": EXPECTED_RECORDS,
            "min_k_repeat_diversity_rate": MIN_REPEAT_DIVERSITY,
            "min_rs_evaluable": MIN_RS_EVALUABLE,
            "permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "one_sided_alpha": ALPHA,
            "min_easy_hard_cq_gap": MIN_EASY_HARD_GAP,
            "detector_alarms_used": False,
        },
        "overall_gate_pass": bool(row["overall_gate_pass"]),
        "source_sha256": {
            "manifest.json": _sha256(manifest_path),
            str(records_path.relative_to(result_dir)): _sha256(records_path),
            str(per_query_path.relative_to(result_dir)): _sha256(per_query_path),
        },
    }
    (result_dir / "e2h_mistral_holdout_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_assessment(result_dir / "E2H_MISTRAL_HOLDOUT_ASSESSMENT.md", row)
    return result
