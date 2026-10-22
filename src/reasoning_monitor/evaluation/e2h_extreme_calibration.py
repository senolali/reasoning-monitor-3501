"""Frozen post-run gate for GPT/Llama official-E2H extreme bands."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.data.loaders import e2h_refined_partition
from reasoning_monitor.evaluation.e2h_benchmark_calibration import (
    permutation_slope_test,
)

BANDS = ("easy", "medium", "hard")
MODELS = ("gpt4o_mini", "llama3_8b")
SCENARIOS = {"easy": "ST3", "medium": "ST4", "hard": "ST5"}
N_PER_BAND = 30
N_PERMUTATIONS = 5_000
PERMUTATION_SEED = 94_046
ALPHA_PER_MODEL = 0.05 / len(MODELS)
MIN_EASY_HARD_GAP = 0.15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _band_matches(band: str, quantile: float) -> bool:
    return (
        band == "easy" and 0 <= quantile <= 0.20
        or band == "medium" and 0.40 <= quantile <= 0.60
        or band == "hard" and 0.80 <= quantile <= 1.0
    )


def evaluate_extreme_model(frame: pd.DataFrame) -> dict[str, Any]:
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
        "negative_slope_pass": slope < 0 and p_value <= ALPHA_PER_MODEL,
        "easy_hard_gap_pass": gap >= MIN_EASY_HARD_GAP,
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_extreme_quantiles",
        "difficulty_partition": "range_calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [46],
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
    if mismatches:
        raise ValueError(f"result does not match frozen Section 43 config: {mismatches}")


def _load_model(result_dir: Path, model: str) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for band in BANDS:
        scenario = SCENARIOS[band]
        tag = f"{scenario}_{model}_seed46_W5"
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
        if any(record.get("role") != "run" or not bool(record.get("ok")) for record in records):
            raise ValueError(f"{model}/{band} contains an invalid raw record")
        frames.append(frame)
        paths.extend((per_query_path, records_path))
    return pd.concat(frames, ignore_index=True), paths


def _write_assessment(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GPT/Llama E2H Extreme-Range Calibration",
        "",
        "This call-free report applies the criteria frozen in Section 43.",
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
            f"{'PASS' if row['eligible_for_final_holdout'] else 'FAIL'} |"
        )
    eligible = [row["model"] for row in rows if row["eligible_for_final_holdout"]]
    lines.extend([
        "",
        "## Decision",
        "",
        "Eligible for final-holdout K/L design: "
        + (", ".join(eligible) if eligible else "none"),
        "",
        "Failed models may proceed only to the preregistered E2H-AMC fallback; "
        "do not tune these bands or thresholds against observed outcomes.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_e2h_extreme_calibration(result_dir: str | Path) -> dict[str, Any]:
    """Assess the completed 180-call pilot without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_config(manifest["config"])

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {"manifest.json": _sha256(manifest_path)}
    for model in MODELS:
        frame, paths = _load_model(result_dir, model)
        required = {
            "qid", "difficulty", "difficulty_rating", "difficulty_quantile",
            "CQ_raw", "call_failed",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{model} is missing columns: {sorted(missing)}")
        if frame["qid"].duplicated().any() or len(frame) != 3 * N_PER_BAND:
            raise ValueError(f"{model} must contain 90 unique questions")
        if frame[["difficulty_rating", "difficulty_quantile", "CQ_raw"]].isna().any().any():
            raise ValueError(f"{model} has missing difficulty metadata or CQ")
        if frame["call_failed"].astype(str).str.lower().isin({"true", "1"}).any():
            raise ValueError(f"{model} contains failed calls")
        cq = frame["CQ_raw"].to_numpy(dtype=float)
        if any(not math.isfinite(value) or value not in {0.0, 1.0} for value in cq):
            raise ValueError(f"{model} CQ values must be finite binary observations")
        for item in frame.itertuples(index=False):
            if e2h_refined_partition(str(item.qid)) != "range_calibration":
                raise ValueError(f"{model} contains a non-range-calibration item")
            if not _band_matches(str(item.difficulty), float(item.difficulty_quantile)):
                raise ValueError(f"{model} contains an item outside its extreme band")

        row = {"model": model, "n_queries": len(frame), **evaluate_extreme_model(frame)}
        row["eligible_for_final_holdout"] = bool(
            row["negative_slope_pass"] and row["easy_hard_gap_pass"]
        )
        rows.append(row)
        for source_path in paths:
            source_hashes[str(source_path.relative_to(result_dir))] = _sha256(source_path)

    data_dir = result_dir / "data"
    pd.DataFrame(rows).to_csv(data_dir / "e2h_extreme_calibration_gate.csv", index=False)
    eligible = [row["model"] for row in rows if row["eligible_for_final_holdout"]]
    result = {
        "analysis": "gpt_llama_e2h_gsm8k_extreme_range_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "criteria": {
            "permutations": N_PERMUTATIONS,
            "permutation_seed": PERMUTATION_SEED,
            "one_sided_alpha_per_model": ALPHA_PER_MODEL,
            "min_easy_hard_cq_gap": MIN_EASY_HARD_GAP,
            "partition": "range_calibration",
        },
        "eligible_models": eligible,
        "source_sha256": source_hashes,
    }
    (result_dir / "e2h_extreme_calibration_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_assessment(result_dir / "E2H_EXTREME_CALIBRATION_ASSESSMENT.md", rows)
    return result
