"""Frozen Section 47 preflight and post-run gates for final E2H K/L tracks."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reasoning_monitor.data.loaders import (
    e2h_partition,
    e2h_refined_partition,
    load_all,
    tag_difficulty,
)
from reasoning_monitor.data.scenarios import SCENARIOS, build_stream, resize_regimes
from reasoning_monitor.evaluation.e2h_benchmark_calibration import (
    permutation_slope_test,
)
from reasoning_monitor.evaluation.kl_validation import repeat_diversity_rate
from reasoning_monitor.metrics.dimensions import answers_equal, require_amc_symbolic_matcher
from reasoning_monitor.types import METRIC_ORDER

BANDS = ("easy", "medium", "hard")
SEED = 48
WINDOW_SIZE = 10
N_PER_BAND = 30
EXPECTED_QUERIES = 90
K_REPEATS = 5
L_PARAPHRASES = 3
EXPECTED_RECORDS = EXPECTED_QUERIES * (K_REPEATS + L_PARAPHRASES)
MAX_TOKENS = 2048
N_PERMUTATIONS = 5_000
FAMILY_ALPHA = 0.05 / 3
MIN_EASY_HARD_GAP = 0.15
MIN_REPEAT_DIVERSITY = 0.80
MIN_RS_EVALUABLE = 20
TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens", "model_length"}


@dataclass(frozen=True)
class FinalHoldoutSpec:
    track: str
    config_name: str
    result_dir: str
    dataset: str
    model: str
    difficulty_scheme: str
    partition: str
    tag: str
    permutation_seed: int
    prior_result_dirs: tuple[str, ...]


FINAL_HOLDOUT_SPECS = {
    "llama3_8b": FinalHoldoutSpec(
        track="llama3_8b",
        config_name="e2h_final_holdout_llama3_8b_asu.yaml",
        result_dir="results/e2h_final_holdout_llama3_8b",
        dataset="e2h_gsm8k",
        model="llama3_8b",
        difficulty_scheme="e2h_extreme_quantiles",
        partition="final_holdout",
        tag="ST2_llama3_8b_seed48_W10",
        permutation_seed=96_048,
        prior_result_dirs=(
            "results/e2h_benchmark_calibration_main_models/data",
            "results/e2h_extreme_calibration_gpt_llama/data",
        ),
    ),
    "gpt4o_mini": FinalHoldoutSpec(
        track="gpt4o_mini",
        config_name="e2h_final_holdout_gpt4o_mini_asu.yaml",
        result_dir="results/e2h_final_holdout_gpt4o_mini",
        dataset="e2h_amc",
        model="gpt4o_mini",
        difficulty_scheme="e2h_rating_quantiles",
        partition="holdout",
        tag="ST2_gpt4o_mini_seed48_W10",
        permutation_seed=97_048,
        prior_result_dirs=(
            "results/e2h_amc_calibration_gpt4o_mini/data",
            "results/e2h_amc_calibration_gpt4o_mini_retry/data",
        ),
    ),
    "mistral_7b": FinalHoldoutSpec(
        track="mistral_7b",
        config_name="e2h_final_holdout_mistral_7b_asu.yaml",
        result_dir="results/e2h_final_holdout_mistral_7b",
        dataset="e2h_gsm8k",
        model="mistral_7b",
        difficulty_scheme="e2h_rating_quantiles",
        partition="final_holdout",
        tag="ST2_mistral_7b_seed48_W10",
        permutation_seed=98_048,
        prior_result_dirs=(
            "results/e2h_benchmark_calibration_main_models/data",
            "results/e2h_extreme_calibration_gpt_llama/data",
        ),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def validate_final_holdout_config(config: dict[str, Any], spec: FinalHoldoutSpec) -> None:
    """Reject result directories that do not match the frozen Section 47 arm."""
    expected = {
        "seed": SEED,
        "seeds": [SEED],
        "output_dir": spec.result_dir,
        "stream_design": (
            "e2h_benchmark_amc" if spec.dataset == "e2h_amc" else "e2h_benchmark_gsm8k"
        ),
        "datasets": [spec.dataset],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": spec.difficulty_scheme,
        "difficulty_partition": spec.partition,
        "scenarios": ["ST2"],
        "window_sizes": [WINDOW_SIZE],
        "queries_per_regime": N_PER_BAND,
        "k_repeats": K_REPEATS,
        "l_paraphrases": L_PARAPHRASES,
        "use_rqeval": True,
        "coherence_scorer": "nli",
        "stability_scorer": "bertscore",
        "publication_mode": True,
        "detector_strategy": "proposal_mapping",
        "resume": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    models = config.get("models") or []
    names = [model.get("name") for model in models]
    if names != [spec.model]:
        mismatches["models"] = (names, [spec.model])
    elif int(models[0].get("max_tokens") or 0) != MAX_TOKENS:
        mismatches["max_tokens"] = (models[0].get("max_tokens"), MAX_TOKENS)
    if mismatches:
        raise ValueError(
            f"result does not match frozen Section 47 {spec.track} config: {mismatches}"
        )


def _band_matches(quantile: float, band: str, scheme: str) -> bool:
    if scheme == "e2h_extreme_quantiles":
        return (
            band == "easy"
            and 0 <= quantile <= 0.20
            or band == "medium"
            and 0.40 <= quantile <= 0.60
            or band == "hard"
            and 0.80 <= quantile <= 1
        )
    return (
        band == "easy"
        and 0 <= quantile <= 1 / 3
        or band == "medium"
        and 1 / 3 < quantile <= 2 / 3
        or band == "hard"
        and 2 / 3 < quantile <= 1
    )


def _partition_matches(qid: str, partition: str) -> bool:
    if partition in {"calibration", "holdout"}:
        return e2h_partition(qid) == partition
    return e2h_refined_partition(qid) == partition


def _metadata_pass(frame: pd.DataFrame, spec: FinalHoldoutSpec) -> bool:
    columns = ["difficulty_rating", "difficulty_quantile"]
    if frame[columns].isna().any().any():
        return False
    for row in frame.itertuples(index=False):
        if str(row.dataset) != spec.dataset:
            return False
        quantile = float(row.difficulty_quantile)
        if quantile > 1:
            quantile /= 100
        if not _band_matches(quantile, str(row.difficulty), spec.difficulty_scheme):
            return False
        if not _partition_matches(str(row.qid), spec.partition):
            return False
    return True


def _completion_health(records: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    truncated = []
    missing_finish_reason = 0
    for record in records:
        finish_reason = str(record.get("finish_reason") or "").strip().lower()
        n_tokens = int(record.get("n_tokens") or 0)
        missing_finish_reason += not finish_reason
        legacy_at_cap = not finish_reason and n_tokens >= max_tokens
        truncated.append(finish_reason in TRUNCATION_REASONS or legacy_at_cap)
    return {
        "max_tokens": max_tokens,
        "n_records": len(records),
        "n_missing_finish_reason": int(missing_finish_reason),
        "n_truncated": int(sum(truncated)),
        "execution_valid": not any(truncated),
    }


def _rescore_primary_cq(
    frame: pd.DataFrame,
    records: list[dict[str, Any]],
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("role") == "run" and int(record.get("call_index", -1)) == 0:
            qid = str(record["qid"])
            if qid in primary:
                raise ValueError(f"duplicate primary record for {qid}")
            primary[qid] = record
    if len(primary) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} primary records, found {len(primary)}")

    rescored = []
    audit_rows = []
    for item in frame.itertuples(index=False):
        qid = str(item.qid)
        if qid not in primary:
            raise ValueError(f"primary record missing for {qid}")
        record = primary[qid]
        source = str(record.get("raw_text") or record.get("answer") or "")
        value = float(answers_equal(source, str(record.get("gold_answer") or ""), dataset))
        old_value = float(item.CQ_raw)
        rescored.append(value)
        audit_rows.append(
            {
                "qid": qid,
                "difficulty": str(item.difficulty),
                "gold_answer": record.get("gold_answer"),
                "extracted_answer": record.get("answer"),
                "cq_online": old_value,
                "cq_rescored": value,
                "changed": old_value != value,
            }
        )
    rescored_frame = frame.copy()
    rescored_frame["CQ_raw"] = rescored
    return rescored_frame, pd.DataFrame(audit_rows)


def final_holdout_cq_gate(per_query: pd.DataFrame, permutation_seed: int) -> dict[str, Any]:
    """Apply the frozen family-adjusted final-holdout CQ gate."""
    ordered = per_query.sort_values("t").reset_index(drop=True)
    means = {
        band: float(ordered.loc[ordered["difficulty"] == band, "CQ_raw"].mean())
        for band in BANDS
    }
    slope, p_value = permutation_slope_test(
        ordered["difficulty_rating"].to_numpy(dtype=float),
        ordered["CQ_raw"].to_numpy(dtype=float),
        permutations=N_PERMUTATIONS,
        seed=permutation_seed,
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
        "negative_slope_pass": slope < 0 and p_value <= FAMILY_ALPHA,
        "easy_hard_gap_pass": gap >= MIN_EASY_HARD_GAP,
        "cq_holdout_gate_pass": (
            phase_order_pass
            and slope < 0
            and p_value <= FAMILY_ALPHA
            and gap >= MIN_EASY_HARD_GAP
        ),
    }


def _write_assessment(path: Path, row: dict[str, Any], spec: FinalHoldoutSpec) -> None:
    lines = [
        f"# Section 47 Final Holdout Assessment: {spec.model}",
        "",
        "This call-free report applies the protocol frozen before Section 47 calls.",
        "Detector alarms are descriptive and are not used by this gate.",
        "",
        "## Completion And Metric Health",
        "",
        f"- Records: {row['n_records']}/{EXPECTED_RECORDS}",
        f"- Queries: {row['n_per_query_rows']}/{EXPECTED_QUERIES}",
        f"- Failed calls: {row['failed_calls']}",
        f"- Truncated calls: {row['n_truncated']}",
        f"- K-repeat diversity: {row['k_repeat_diversity_rate']:.3f}",
        f"- RS evaluable queries: {row['rs_evaluable_n']}",
        f"- Metric-health gate: {'PASS' if row['metric_health_pass'] else 'FAIL'}",
        f"- Execution gate: {'PASS' if row['execution_valid'] else 'FAIL'}",
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
        f"CQ was recomputed from primary raw records; {row['n_cq_changed_after_rescore']} "
        "of 90 online values changed.",
        "",
        "## Decision",
        "",
        (
            "This track passes. Its saved records may proceed only to the separately "
            "preregistered call-free normalized detector analysis."
            if row["overall_gate_pass"]
            else "This track fails. Do not tune bands, thresholds, metric rules, output "
            "limits, or test direction against these holdout outcomes."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_final_holdout(result_dir: str | Path, track: str) -> dict[str, Any]:
    """Assess one completed Section 47 track without issuing model calls."""
    if track not in FINAL_HOLDOUT_SPECS:
        raise ValueError(f"unknown final-holdout track {track!r}")
    spec = FINAL_HOLDOUT_SPECS[track]
    if spec.dataset == "e2h_amc":
        require_amc_symbolic_matcher()

    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    validate_final_holdout_config(config, spec)

    data_dir = result_dir / "data"
    records_path = data_dir / f"records_{spec.tag}.jsonl"
    per_query_path = data_dir / f"per_query_{spec.tag}.csv"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    per_query = pd.read_csv(per_query_path)
    required = {
        "t",
        "qid",
        "dataset",
        "difficulty",
        "difficulty_rating",
        "difficulty_quantile",
        "call_failed",
        *(f"{metric}_raw" for metric in METRIC_ORDER),
    }
    missing = required - set(per_query.columns)
    if missing:
        raise ValueError(f"final-holdout file is missing columns: {sorted(missing)}")

    per_query, cq_audit = _rescore_primary_cq(per_query, records, spec.dataset)
    cq_audit.to_csv(data_dir / "e2h_final_holdout_cq_rescore.csv", index=False)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)
    expected_roles = {
        *(("run", index) for index in range(K_REPEATS)),
        *(("paraphrase", index) for index in range(L_PARAPHRASES)),
    }
    role_counts_ok = all(
        len(query_records) == len(expected_roles)
        and {
            (str(record.get("role")), int(record.get("call_index", -1)))
            for record in query_records
        }
        == expected_roles
        for query_records in grouped.values()
    )
    raw_qids = set(grouped)
    per_query_qids = set(per_query["qid"].astype(str))
    qids_match = raw_qids == per_query_qids
    unique_queries = len(per_query) == len(per_query_qids) == EXPECTED_QUERIES
    metadata_pass = _metadata_pass(per_query, spec)

    failed_calls = sum(not bool(record.get("ok")) for record in records)
    per_query_failed_calls = sum(_is_true(value) for value in per_query["call_failed"])
    manifest_failed = int((manifest.get("failed_calls") or {}).get(spec.tag, -1))
    diverse, complete, diversity_rate = repeat_diversity_rate(records, K_REPEATS)
    completion = _completion_health(records, MAX_TOKENS)

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
    nondegenerate = spec.tag not in (manifest.get("degenerate_metrics") or {})
    cq_gate = final_holdout_cq_gate(per_query, spec.permutation_seed)

    metric_health_pass = all(
        [
            len(records) == EXPECTED_RECORDS,
            len(grouped) == EXPECTED_QUERIES,
            len(per_query) == EXPECTED_QUERIES,
            unique_queries,
            qids_match,
            role_counts_ok,
            metadata_pass,
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
        ]
    )
    row = {
        "track": spec.track,
        "model": spec.model,
        "dataset": spec.dataset,
        "n_records": len(records),
        "expected_records": EXPECTED_RECORDS,
        "n_queries": len(grouped),
        "n_per_query_rows": len(per_query),
        "expected_queries": EXPECTED_QUERIES,
        "unique_queries": unique_queries,
        "qids_match": qids_match,
        "role_counts_ok": role_counts_ok,
        "metadata_pass": metadata_pass,
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
        "n_cq_changed_after_rescore": int(cq_audit["changed"].sum()),
        **completion,
        **cq_gate,
    }
    row["overall_gate_pass"] = bool(
        metric_health_pass
        and completion["execution_valid"]
        and cq_gate["cq_holdout_gate_pass"]
    )
    pd.DataFrame([row]).to_csv(data_dir / "e2h_final_holdout_gate.csv", index=False)

    result = {
        "analysis": f"section47_{spec.track}_final_holdout_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "criteria": {
            "expected_records": EXPECTED_RECORDS,
            "min_k_repeat_diversity_rate": MIN_REPEAT_DIVERSITY,
            "min_rs_evaluable": MIN_RS_EVALUABLE,
            "permutations": N_PERMUTATIONS,
            "permutation_seed": spec.permutation_seed,
            "family_alpha": FAMILY_ALPHA,
            "min_easy_hard_cq_gap": MIN_EASY_HARD_GAP,
            "zero_truncation_required": True,
            "detector_alarms_used": False,
        },
        "overall_gate_pass": bool(row["overall_gate_pass"]),
        "source_sha256": {
            "manifest.json": _sha256(manifest_path),
            str(records_path.relative_to(result_dir)): _sha256(records_path),
            str(per_query_path.relative_to(result_dir)): _sha256(per_query_path),
        },
    }
    (result_dir / "e2h_final_holdout_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_assessment(result_dir / "E2H_FINAL_HOLDOUT_ASSESSMENT.md", row, spec)
    return result


def _prior_qids(project_root: Path, spec: FinalHoldoutSpec) -> set[str]:
    qids: set[str] = set()
    for relative_dir in spec.prior_result_dirs:
        directory = project_root / relative_dir
        for path in directory.glob("per_query_*.csv"):
            frame = pd.read_csv(path, usecols=["qid"])
            qids.update(frame["qid"].astype(str))
    return qids


def preflight_e2h_final_holdouts(
    project_root: str | Path,
    output_dir: str | Path = "results/e2h_final_holdout_preflight",
) -> dict[str, Any]:
    """Validate all frozen configs and selected queries without building clients."""
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tracks = {}
    selected_by_track: dict[str, set[str]] = {}
    for track, spec in FINAL_HOLDOUT_SPECS.items():
        config_path = project_root / "configs" / spec.config_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_final_holdout_config(config, spec)
        if spec.dataset == "e2h_amc":
            require_amc_symbolic_matcher()
        datasets = load_all(
            [spec.dataset],
            limit_per_dataset=int(config["items_per_dataset"]),
            cache_dir=project_root / "data" / "raw",
            seed=SEED,
            allow_synthetic=False,
        )
        tiers = tag_difficulty(
            datasets[spec.dataset].queries,
            seed=SEED,
            scheme=spec.difficulty_scheme,
            partition=spec.partition,
        )
        scenario = resize_regimes(SCENARIOS["ST2"], N_PER_BAND)
        stream = build_stream(
            scenario,
            datasets,
            seed=SEED,
            base_dataset=spec.dataset,
            difficulty_scheme=spec.difficulty_scheme,
            difficulty_partition=spec.partition,
        )
        frame = pd.DataFrame(
            [
                {
                    "track": track,
                    "model": spec.model,
                    "dataset": query.dataset,
                    "t": query.t,
                    "qid": query.qid,
                    "difficulty": query.difficulty,
                    "difficulty_rating": query.difficulty_rating,
                    "difficulty_quantile": query.difficulty_quantile,
                }
                for query in stream
            ]
        )
        qids = set(frame["qid"].astype(str))
        prior_overlap = qids & _prior_qids(project_root, spec)
        counts = frame["difficulty"].value_counts().to_dict()
        valid = all(
            [
                len(frame) == EXPECTED_QUERIES,
                len(qids) == EXPECTED_QUERIES,
                counts == {band: N_PER_BAND for band in BANDS},
                _metadata_pass(frame, spec),
                not prior_overlap,
            ]
        )
        if not valid:
            raise ValueError(
                f"Section 47 preflight failed for {track}: "
                f"rows={len(frame)}, unique={len(qids)}, counts={counts}, "
                f"prior_overlap={len(prior_overlap)}"
            )
        rows.extend(frame.to_dict(orient="records"))
        selected_by_track[track] = qids
        tracks[track] = {
            "config": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            "dataset": spec.dataset,
            "dataset_source": datasets[spec.dataset].source,
            "dataset_items": len(datasets[spec.dataset].queries),
            "dataset_content_hash": datasets[spec.dataset].content_hash,
            "difficulty_scheme": spec.difficulty_scheme,
            "partition": spec.partition,
            "available_by_band": {band: len(tiers[band]) for band in BANDS},
            "selected_by_band": counts,
            "selected_unique": len(qids),
            "prior_overlap": len(prior_overlap),
            "planned_calls": EXPECTED_RECORDS,
            "max_output_tokens": EXPECTED_RECORDS * MAX_TOKENS,
        }

    selected = pd.DataFrame(rows)
    selected_path = output_dir / "selected_queries.csv"
    selected.to_csv(selected_path, index=False)
    gsm_overlap = len(selected_by_track["llama3_8b"] & selected_by_track["mistral_7b"])
    result = {
        "analysis": "section47_final_holdout_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "protocol_frozen": True,
        "seed": SEED,
        "queries_per_track": EXPECTED_QUERIES,
        "calls_per_track": EXPECTED_RECORDS,
        "total_planned_calls": EXPECTED_RECORDS * len(FINAL_HOLDOUT_SPECS),
        "first_stage_planned_calls": EXPECTED_RECORDS * 2,
        "max_tokens_per_call": MAX_TOKENS,
        "family_alpha": FAMILY_ALPHA,
        "min_easy_hard_gap": MIN_EASY_HARD_GAP,
        "llama_mistral_selected_qid_overlap": gsm_overlap,
        "tracks": tracks,
        "selected_queries_sha256": _sha256(selected_path),
    }
    (output_dir / "preflight_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
