"""Frozen post-run gate for the official Easy2Hard-Bench calibration."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reasoning_monitor.data.loaders import e2h_partition

BANDS = ("easy", "medium", "hard")
MODELS = ("gpt4o_mini", "llama3_8b", "mistral_7b")
SCENARIOS = {"easy": "ST3", "medium": "ST4", "hard": "ST5"}
N_PER_BAND = 30
N_PERMUTATIONS = 5_000
PERMUTATION_SEED = 92_044
ALPHA_PER_MODEL = 0.05 / len(MODELS)
MIN_EASY_HARD_GAP = 0.15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        raise ValueError("difficulty ratings have zero variance")
    return float(np.dot(centered, y - y.mean()) / denominator)


def permutation_slope_test(
    ratings: np.ndarray,
    cq: np.ndarray,
    permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, float]:
    """One-sided test for a more-negative-than-null CQ/difficulty slope."""
    x = np.asarray(ratings, dtype=float)
    y = np.asarray(cq, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("ratings and CQ must be equal-length one-dimensional arrays")
    observed = _slope(x, y)
    rng = np.random.default_rng(seed)
    at_least_as_negative = 0
    for _ in range(permutations):
        if _slope(x, rng.permutation(y)) <= observed:
            at_least_as_negative += 1
    p_value = (at_least_as_negative + 1) / (permutations + 1)
    return observed, p_value


def evaluate_model(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one model after structural completion checks have passed."""
    means = {
        band: float(frame.loc[frame["difficulty"] == band, "CQ_raw"].mean())
        for band in BANDS
    }
    slope, p_value = permutation_slope_test(
        frame["difficulty_rating"].to_numpy(dtype=float),
        frame["CQ_raw"].to_numpy(dtype=float),
    )
    gap = means["easy"] - means["hard"]
    return {
        "easy_cq": means["easy"],
        "medium_cq": means["medium"],
        "hard_cq": means["hard"],
        "easy_hard_gap": gap,
        "cq_rating_slope": slope,
        "permutation_p_one_sided": p_value,
        "negative_slope_pass": slope < 0 and p_value <= ALPHA_PER_MODEL,
        "easy_hard_gap_pass": gap >= MIN_EASY_HARD_GAP,
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [44],
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
    if names != MODELS:
        mismatches["models"] = (names, MODELS)
    if config.get("allow_synthetic_datasets") is not False:
        mismatches["allow_synthetic_datasets"] = (
            config.get("allow_synthetic_datasets"), False
        )
    if mismatches:
        raise ValueError(f"result does not match frozen Section 41 config: {mismatches}")


def _load_model(result_dir: Path, model: str) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for band in BANDS:
        scenario = SCENARIOS[band]
        tag = f"{scenario}_{model}_seed44_W5"
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
                f"{model}/{band} is incomplete: {len(frame)} rows, {len(records)} records"
            )
        if set(frame["difficulty"].astype(str)) != {band}:
            raise ValueError(f"{model}/{band} has an unexpected difficulty label")
        if any(record.get("role") != "run" for record in records):
            raise ValueError(f"{model}/{band} contains a non-primary call")
        if any(not bool(record.get("ok")) for record in records):
            raise ValueError(f"{model}/{band} contains a failed raw record")
        frames.append(frame)
        paths.extend((per_query_path, records_path))
    return pd.concat(frames, ignore_index=True), paths


def _write_assessment(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Official E2H Calibration Assessment",
        "",
        "This call-free report applies the criteria frozen in Section 41.",
        "Passing applies independently to each model.",
        "",
        "| Model | Easy CQ | Medium CQ | Hard CQ | E-H gap | Slope | p | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['easy_cq']:.3f} | "
            f"{row['medium_cq']:.3f} | {row['hard_cq']:.3f} | "
            f"{row['easy_hard_gap']:.3f} | {row['cq_rating_slope']:.6f} | "
            f"{row['permutation_p_one_sided']:.6f} | "
            f"{'PASS' if row['eligible_for_holdout'] else 'FAIL'} |"
        )
    eligible = [row["model"] for row in rows if row["eligible_for_holdout"]]
    lines.extend([
        "",
        "## Decision",
        "",
        "Eligible for the disjoint holdout K/L run: "
        + (", ".join(eligible) if eligible else "none"),
        "",
        "Failed models must not be carried into the holdout run or rescued by "
        "changing bands, thresholds, or test direction.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_benchmark_calibration(result_dir: str | Path) -> dict[str, Any]:
    """Assess the completed 270-call calibration without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_config(manifest["config"])

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {"manifest.json": _sha256(manifest_path)}
    for model in MODELS:
        frame, paths = _load_model(result_dir, model)
        required = {"qid", "difficulty", "difficulty_rating", "CQ_raw", "call_failed"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{model} is missing columns: {sorted(missing)}")
        if frame["qid"].duplicated().any() or len(frame) != 3 * N_PER_BAND:
            raise ValueError(f"{model} must contain 90 unique calibration questions")
        if frame[["difficulty_rating", "CQ_raw"]].isna().any().any():
            raise ValueError(f"{model} has missing rating or CQ values")
        cq = frame["CQ_raw"].to_numpy(dtype=float)
        if any(not math.isfinite(value) or value not in {0.0, 1.0} for value in cq):
            raise ValueError(f"{model} CQ values must be finite binary observations")
        failed = frame["call_failed"].astype(str).str.lower().isin({"true", "1"}).any()
        if failed:
            raise ValueError(f"{model} contains failed calls")
        if any(e2h_partition(str(qid)) != "calibration" for qid in frame["qid"]):
            raise ValueError(f"{model} contains a holdout question")
        row = {"model": model, "n_queries": len(frame), **evaluate_model(frame)}
        row["structural_pass"] = True
        row["eligible_for_holdout"] = bool(
            row["negative_slope_pass"] and row["easy_hard_gap_pass"]
        )
        rows.append(row)
        for source_path in paths:
            source_hashes[str(source_path.relative_to(result_dir))] = _sha256(source_path)

    gate_path = result_dir / "data" / "e2h_benchmark_calibration_gate.csv"
    pd.DataFrame(rows).to_csv(gate_path, index=False)
    eligible = [row["model"] for row in rows if row["eligible_for_holdout"]]
    result = {
        "analysis": "official_e2h_gsm8k_model_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "criteria": {
            "permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "one_sided_alpha_per_model": ALPHA_PER_MODEL,
            "min_easy_hard_cq_gap": MIN_EASY_HARD_GAP,
            "partition": "calibration",
        },
        "eligible_models": eligible,
        "any_model_eligible": bool(eligible),
        "source_sha256": source_hashes,
    }
    (result_dir / "e2h_benchmark_calibration_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_assessment(result_dir / "E2H_BENCHMARK_CALIBRATION_ASSESSMENT.md", rows)
    return result
