"""Frozen post-run gate for the GPT-4o-mini official E2H-AMC fallback."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.data.loaders import e2h_partition
from reasoning_monitor.evaluation.e2h_benchmark_calibration import (
    permutation_slope_test,
)
from reasoning_monitor.metrics.dimensions import (
    answers_equal,
    require_amc_symbolic_matcher,
)

BANDS = ("easy", "medium", "hard")
MODEL = "gpt4o_mini"
SCENARIOS = {"easy": "ST3", "medium": "ST4", "hard": "ST5"}
N_PER_BAND = 30
N_PERMUTATIONS = 5_000
PERMUTATION_SEED = 95_047
ALPHA = 0.05
MIN_EASY_HARD_GAP = 0.15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _band_matches(band: str, quantile: float) -> bool:
    return (
        band == "easy" and 0 <= quantile <= 1 / 3
        or band == "medium" and 1 / 3 < quantile <= 2 / 3
        or band == "hard" and 2 / 3 < quantile <= 1.0
    )


def evaluate_amc_model(frame: pd.DataFrame) -> dict[str, Any]:
    means = {
        band: float(frame.loc[frame["difficulty"] == band, "CQ_raw"].mean())
        for band in BANDS
    }
    slope, p_value = permutation_slope_test(
        frame["difficulty_rating"].to_numpy(dtype=float),
        frame["CQ_raw"].to_numpy(dtype=float),
        permutations=N_PERMUTATIONS,
        seed=PERMUTATION_SEED,
    )
    gap = means["easy"] - means["hard"]
    return {
        "easy_cq": means["easy"],
        "medium_cq": means["medium"],
        "hard_cq": means["hard"],
        "easy_hard_gap": gap,
        "cq_rating_slope": slope,
        "permutation_p_one_sided": p_value,
        "negative_slope_pass": slope < 0 and p_value <= ALPHA,
        "easy_hard_gap_pass": gap >= MIN_EASY_HARD_GAP,
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "stream_design": "e2h_benchmark_amc",
        "datasets": ["e2h_amc"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [47],
        "window_sizes": [5],
        "queries_per_regime": 30,
        "k_repeats": 1,
        "l_paraphrases": 0,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    names = tuple(model.get("name") for model in config.get("models", []))
    if names != (MODEL,):
        mismatches["models"] = (names, (MODEL,))
    if mismatches:
        raise ValueError(f"result does not match frozen Section 44 config: {mismatches}")


def _load_result(result_dir: Path) -> tuple[pd.DataFrame, list[Path], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    all_records: list[dict[str, Any]] = []
    for band in BANDS:
        scenario = SCENARIOS[band]
        tag = f"{scenario}_{MODEL}_seed47_W5"
        per_query_path = result_dir / "data" / f"per_query_{tag}.csv"
        records_path = result_dir / "data" / f"records_{tag}.jsonl"
        frame = pd.read_csv(per_query_path)
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(frame) != N_PER_BAND or len(records) != N_PER_BAND:
            raise ValueError(
                f"{MODEL}/{band} is incomplete: {len(frame)} rows, {len(records)} records"
            )
        if set(frame["difficulty"].astype(str)) != {band}:
            raise ValueError(f"{MODEL}/{band} has an unexpected difficulty label")
        if any(record.get("role") != "run" or not bool(record.get("ok")) for record in records):
            raise ValueError(f"{MODEL}/{band} contains an invalid raw record")
        frames.append(frame)
        paths.extend((per_query_path, records_path))
        all_records.extend(records)
    return pd.concat(frames, ignore_index=True), paths, all_records


def _completion_health(
    records: list[dict[str, Any]], max_tokens: int
) -> dict[str, Any]:
    explicit_truncation = {
        "length", "max_tokens", "max_output_tokens", "model_length"
    }
    truncated = []
    answer_markers = 0
    for record in records:
        finish_reason = str(record.get("finish_reason") or "").strip().lower()
        n_tokens = int(record.get("n_tokens") or 0)
        legacy_at_cap = not finish_reason and n_tokens >= max_tokens
        is_truncated = finish_reason in explicit_truncation or legacy_at_cap
        truncated.append(is_truncated)
        answer_markers += "Answer:" in str(record.get("raw_text") or "")
    return {
        "max_tokens": max_tokens,
        "n_records": len(records),
        "n_truncated": int(sum(truncated)),
        "n_answer_markers": int(answer_markers),
        "execution_valid": not any(truncated),
        "legacy_truncation_rule": (
            "missing finish_reason and n_tokens >= configured max_tokens"
        ),
    }


def _rescore_cq(
    frame: pd.DataFrame, records: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_qid = {str(record["qid"]): record for record in records}
    if len(by_qid) != len(records):
        raise ValueError("E2H-AMC raw records must have one unique primary record per qid")
    rescored = []
    audit_rows = []
    for item in frame.itertuples(index=False):
        qid = str(item.qid)
        if qid not in by_qid:
            raise ValueError(f"E2H-AMC raw record missing for {qid}")
        record = by_qid[qid]
        value = float(answers_equal(
            str(record.get("answer") or ""),
            str(record.get("gold_answer") or ""),
            "e2h_amc",
        ))
        old_value = float(item.CQ_raw)
        rescored.append(value)
        audit_rows.append({
            "qid": qid,
            "difficulty": str(item.difficulty),
            "gold_answer": record.get("gold_answer"),
            "extracted_answer": record.get("answer"),
            "cq_online": old_value,
            "cq_rescored": value,
            "changed": old_value != value,
        })
    rescored_frame = frame.copy()
    rescored_frame["CQ_raw"] = rescored
    return rescored_frame, pd.DataFrame(audit_rows)


def _write_assessment(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "# GPT-4o-mini E2H-AMC Calibration",
        "",
        "This call-free report applies the criteria frozen in Sections 44-45.",
        "",
        "| Model | Easy CQ | Medium CQ | Hard CQ | E-H gap | Slope | p | "
        "Statistical | Execution | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        f"| {row['model']} | {row['easy_cq']:.3f} | {row['medium_cq']:.3f} | "
        f"{row['hard_cq']:.3f} | {row['easy_hard_gap']:.3f} | "
        f"{row['cq_rating_slope']:.6f} | {row['permutation_p_one_sided']:.6f} | "
        f"{'PASS' if row['statistical_gate_pass'] else 'FAIL'} | "
        f"{'PASS' if row['execution_valid'] else 'INVALID'} | "
        f"{'PASS' if row['eligible_for_amc_holdout'] else 'FAIL'} |",
        "",
        f"CQ was recomputed from raw records; {row['n_cq_changed_after_rescore']} "
        "of 90 online CQ values changed after answer-extraction correction.",
        "",
        "## Decision",
        "",
        (
            "GPT-4o-mini is eligible for a separately preregistered E2H-AMC holdout "
            "K/L design."
            if row["eligible_for_amc_holdout"]
            else (
                f"This run is execution-invalid because {row['n_truncated']} of "
                f"{row['n_queries']} completions were truncated; its statistical "
                "gate result cannot establish eligibility."
                if not row["execution_valid"]
                else "GPT-4o-mini has no validated operating range in the declared "
                "E2H-GSM8K/E2H-AMC route. Do not tune bands or thresholds."
            )
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_amc_calibration(result_dir: str | Path) -> dict[str, Any]:
    """Assess the completed 90-call fallback without issuing model calls."""
    require_amc_symbolic_matcher()
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_config(manifest["config"])

    frame, paths, records = _load_result(result_dir)
    required = {
        "qid", "dataset", "difficulty", "difficulty_rating", "difficulty_quantile",
        "CQ_raw", "call_failed",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{MODEL} is missing columns: {sorted(missing)}")
    if frame["qid"].duplicated().any() or len(frame) != 3 * N_PER_BAND:
        raise ValueError(f"{MODEL} must contain 90 unique questions")
    if set(frame["dataset"].astype(str)) != {"e2h_amc"}:
        raise ValueError(f"{MODEL} contains a non-E2H-AMC item")
    if frame[["difficulty_rating", "difficulty_quantile", "CQ_raw"]].isna().any().any():
        raise ValueError(f"{MODEL} has missing difficulty metadata or CQ")
    if frame["call_failed"].astype(str).str.lower().isin({"true", "1"}).any():
        raise ValueError(f"{MODEL} contains failed calls")
    cq = frame["CQ_raw"].to_numpy(dtype=float)
    if any(not math.isfinite(value) or value not in {0.0, 1.0} for value in cq):
        raise ValueError(f"{MODEL} CQ values must be finite binary observations")
    for item in frame.itertuples(index=False):
        if e2h_partition(str(item.qid)) != "calibration":
            raise ValueError(f"{MODEL} contains a non-calibration item")
        if not _band_matches(str(item.difficulty), float(item.difficulty_quantile)):
            raise ValueError(f"{MODEL} contains an item outside its declared band")

    frame, cq_audit = _rescore_cq(frame, records)
    model_config = manifest["config"]["models"][0]
    max_tokens = int(model_config.get("max_tokens") or 0)
    if max_tokens < 1:
        raise ValueError("E2H-AMC result has no positive configured max_tokens")
    completion_health = _completion_health(records, max_tokens)
    row = {
        "model": MODEL,
        "n_queries": len(frame),
        **evaluate_amc_model(frame),
        **completion_health,
        "n_cq_changed_after_rescore": int(cq_audit["changed"].sum()),
    }
    row["statistical_gate_pass"] = bool(
        row["negative_slope_pass"] and row["easy_hard_gap_pass"]
    )
    row["eligible_for_amc_holdout"] = bool(
        row["statistical_gate_pass"] and row["execution_valid"]
    )

    data_dir = result_dir / "data"
    cq_audit.to_csv(data_dir / "e2h_amc_cq_rescore.csv", index=False)
    pd.DataFrame([row]).to_csv(data_dir / "e2h_amc_calibration_gate.csv", index=False)
    source_hashes = {"manifest.json": _sha256(manifest_path)}
    for source_path in paths:
        source_hashes[str(source_path.relative_to(result_dir))] = _sha256(source_path)
    result = {
        "analysis": "gpt4o_mini_e2h_amc_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "criteria": {
            "permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "one_sided_alpha": ALPHA,
            "min_easy_hard_cq_gap": MIN_EASY_HARD_GAP,
            "partition": "calibration",
            "execution_validity": "zero provider- or token-cap-detected truncations",
        },
        "completion_health": completion_health,
        "cq_rescore": {
            "source": "raw answer and gold_answer fields",
            "n_changed": int(cq_audit["changed"].sum()),
        },
        "statistical_gate_pass": row["statistical_gate_pass"],
        "eligible_for_amc_holdout": row["eligible_for_amc_holdout"],
        "source_sha256": source_hashes,
    }
    (result_dir / "e2h_amc_calibration_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_assessment(result_dir / "E2H_AMC_CALIBRATION_ASSESSMENT.md", row)
    return result
