"""Split-sample CUSUM threshold calibration for the accepted-proposal S1 pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.detectors.detectors import CUSUM
from reasoning_monitor.evaluation.permutation_controls import (
    cq_cusum_alarm_times,
    generate_matched_permutations,
)

THRESHOLD_GRID = (0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)


def wilson_interval(successes: int, trials: int, z: float = 1.9599639845):
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def select_common_threshold(
    calibration_grid: pd.DataFrame,
    model_names: Sequence[str],
) -> float | None:
    """Select the smallest threshold whose specificity criterion passes all models."""
    expected_models = set(model_names)
    for threshold in sorted(calibration_grid["threshold"].unique()):
        rows = calibration_grid[calibration_grid["threshold"] == threshold]
        if set(rows["model"]) == expected_models and bool(
            rows["specificity_pass"].all()
        ):
            return float(threshold)
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluate_null(
    values: Sequence[float],
    permutations: Sequence[Sequence[int]],
    window_size: int,
    threshold: float,
    drift: float,
    warmup: int,
) -> tuple[list[dict], dict[str, float | int | bool]]:
    detector = CUSUM(threshold=threshold, drift=drift, warmup=warmup)
    rows = []
    alarm_count = 0
    for permutation_id, indices in enumerate(permutations, start=1):
        permuted = [values[index] for index in indices]
        alarm_times = cq_cusum_alarm_times(permuted, window_size, detector)
        first_alarm = alarm_times[0] if alarm_times else None
        alarm_count += bool(alarm_times)
        rows.append({
            "permutation_id": permutation_id,
            "first_cusum_alarm": first_alarm,
            "n_cusum_alarms": len(alarm_times),
            "any_cusum_alarm": bool(alarm_times),
        })
    lower, upper = wilson_interval(alarm_count, len(permutations))
    return rows, {
        "null_any_alarm_count": alarm_count,
        "null_any_alarm_rate": alarm_count / len(permutations),
        "null_alarm_wilson_low": lower,
        "null_alarm_wilson_high": upper,
        "specificity_pass": upper <= 0.05,
    }


def _write_assessment(
    path: Path,
    grid: pd.DataFrame,
    validation: pd.DataFrame,
    selected_threshold: float | None,
    overall_pass: bool,
    calibration_seed: int,
    validation_seed: int,
    n_calibration: int,
    n_validation: int,
) -> None:
    lines = [
        "# S1 CQ-CUSUM Split-Sample Calibration",
        "",
        "This analysis makes zero model or network calls. It uses fresh, disjoint, ",
        f"matched permutation sets: `{n_calibration}` calibration streams with seed ",
        f"`{calibration_seed}` and `{n_validation}` validation streams with seed ",
        f"`{validation_seed}`.",
        "",
        "Window size W=10, drift allowance 0.01 and detector warm-up 20 are fixed. ",
        "The smallest common threshold whose calibration false-alarm Wilson upper ",
        "bound is at most 0.05 for both models is selected, then locked.",
        "",
        "## Calibration Grid",
        "",
        "| Threshold | GPT alarm rate | GPT upper | Llama alarm rate | Llama upper | Eligible |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    pivot = {}
    for row in grid.itertuples(index=False):
        pivot.setdefault(float(row.threshold), {})[row.model] = row
    for threshold in sorted(pivot):
        rows = pivot[threshold]
        gpt = rows.get("gpt4o_mini")
        llama = rows.get("llama3_8b")
        eligible = bool(gpt and llama and gpt.specificity_pass and llama.specificity_pass)
        lines.append(
            f"| {threshold:g} | {gpt.null_any_alarm_rate:.3f} | "
            f"{gpt.null_alarm_wilson_high:.3f} | {llama.null_any_alarm_rate:.3f} | "
            f"{llama.null_alarm_wilson_high:.3f} | {'yes' if eligible else 'no'} |"
        )

    selected = "none" if selected_threshold is None else f"{selected_threshold:g}"
    lines.extend([
        "",
        f"Selected common threshold: `{selected}`.",
        "",
        "## Locked Validation",
        "",
    ])
    if validation.empty:
        lines.append("No threshold qualified for locked validation.")
    else:
        lines.extend([
            "| Model | Null alarm rate | 95% CI | First alarm | Delay from t*=45 | Specificity | Sensitivity | Gate |",
            "| --- | ---: | --- | ---: | ---: | --- | --- | --- |",
        ])
        for row in validation.itertuples(index=False):
            first = "-" if pd.isna(row.ordered_first_alarm) else str(int(row.ordered_first_alarm))
            lines.append(
                f"| {row.model} | {row.null_any_alarm_rate:.3f} | "
                f"[{row.null_alarm_wilson_low:.3f}, {row.null_alarm_wilson_high:.3f}] | "
                f"{first} | {int(row.ordered_delay_from_midpoint)} | "
                f"{'pass' if row.specificity_pass else 'fail'} | "
                f"{'pass' if row.sensitivity_pass else 'fail'} | "
                f"{'PASS' if row.gate_pass else 'FAIL'} |"
            )

    if overall_pass:
        decision = (
            "The locked common threshold passes held-out specificity and detects ordered "
            "S1 for both models. This authorises planning, but not yet executing, the paid "
            "K/L validation under the locked detector configuration. The detections are "
            "late relative to the S1 midpoint; no delay criterion was part of this gate, so "
            "the next protocol must freeze one before execution."
        )
    elif selected_threshold is None:
        decision = (
            "No threshold in the frozen grid controlled calibration false alarms for both "
            "models. The paid K/L validation remains unauthorised."
        )
    elif bool(validation["specificity_pass"].all()) and not bool(
        validation["sensitivity_pass"].all()
    ):
        decision = (
            "The selected threshold controls held-out false alarms but loses ordered-S1 "
            "sensitivity for at least one model. The present CUSUM design cannot satisfy "
            "both requirements, so the paid K/L validation remains unauthorised."
        )
    else:
        decision = (
            "The selected threshold failed held-out specificity for at least one model. "
            "The paid K/L validation remains unauthorised."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        decision,
        "",
        "The earlier single-shuffle and 1,000-permutation results remain unchanged and ",
        "failed; this split-sample analysis is a separate prospective gate.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def calibrate_s1_cusum(
    result_dir: str | Path,
    n_calibration: int = 1000,
    n_validation: int = 1000,
    calibration_seed: int = 31042,
    validation_seed: int = 41042,
    threshold_grid: Sequence[float] = THRESHOLD_GRID,
) -> dict[str, Any]:
    """Select a common CQ-CUSUM threshold and test it on fresh permutations."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    permutation_dir = result_dir / "order_controls" / "permutation_test"
    permutation_manifest_path = permutation_dir / "manifest.json"
    permutation_summary_path = permutation_dir / "permutation_summary.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    permutation_manifest = json.loads(
        permutation_manifest_path.read_text(encoding="utf-8")
    )
    config = manifest["config"]
    if permutation_manifest.get("overall_gate_pass") is not False:
        raise ValueError("split calibration requires the failed permutation gate")
    if permutation_manifest.get("source_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("source manifest changed after permutation analysis")
    trend_summary = pd.read_csv(permutation_summary_path)
    if not bool(trend_summary["trend_pass"].all()):
        raise ValueError("the preregistered CQ trend prerequisite did not pass")
    if not threshold_grid or any(float(value) <= 0 for value in threshold_grid):
        raise ValueError("threshold grid must contain positive values")
    if len(set(float(value) for value in threshold_grid)) != len(threshold_grid):
        raise ValueError("threshold grid contains duplicates")

    seeds = config.get("seeds") or [config.get("seed")]
    windows = config.get("window_sizes") or []
    models = config.get("models") or []
    if len(seeds) != 1 or len(windows) != 1 or len(models) != 2:
        raise ValueError("split calibration requires one run seed, one window and two models")
    source_seed, window_size = int(seeds[0]), int(windows[0])
    if window_size != 10:
        raise ValueError("the preregistered split calibration fixes W=10")
    model_names = [str(model["name"]) for model in models]

    data_dir = result_dir / "data"
    output_dir = result_dir / "order_controls" / "cusum_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = data_dir / f"stream_S1_seed{source_seed}.csv"
    qids = pd.read_csv(stream_path).sort_values("t")["qid"].astype(str).tolist()
    detection = pd.read_csv(data_dir / "detection_results.csv")
    s1_detection = detection[detection["scenario"] == "S1"]
    transition_starts = s1_detection["transition_start"].dropna().astype(int).unique()
    effective_midpoints = s1_detection["effective_t_star"].dropna().astype(int).unique()
    if len(transition_starts) != 1 or len(effective_midpoints) != 1:
        raise ValueError("source S1 transition metadata is not unique")
    transition_start = int(transition_starts[0])
    effective_midpoint = int(effective_midpoints[0])
    calibration_permutations = generate_matched_permutations(
        len(qids), n_calibration, calibration_seed
    )
    validation_permutations = generate_matched_permutations(
        len(qids), n_validation, validation_seed
    )
    overlap = set(calibration_permutations) & set(validation_permutations)
    if overlap:
        raise ValueError("calibration and validation permutations overlap")

    values_by_model: dict[str, list[float]] = {}
    source_hashes: dict[str, str] = {}
    for model_name in model_names:
        per_query_path = (
            data_dir / f"per_query_S1_{model_name}_seed{source_seed}_W{window_size}.csv"
        )
        per_query = pd.read_csv(per_query_path).sort_values("t")
        if per_query["qid"].astype(str).tolist() != qids:
            raise ValueError(f"per-query order mismatch for {model_name}")
        values = [float(value) for value in per_query["CQ_raw"].tolist()]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"invalid CQ values for {model_name}")
        values_by_model[model_name] = values
        source_hashes[model_name] = _sha256(per_query_path)

    drift = 0.01
    warmup = 20
    grid_rows = []
    for threshold in sorted(float(value) for value in threshold_grid):
        for model_name in model_names:
            _rows, aggregate = _evaluate_null(
                values_by_model[model_name],
                calibration_permutations,
                window_size,
                threshold,
                drift,
                warmup,
            )
            grid_rows.append({
                "model": model_name,
                "threshold": threshold,
                "n_permutations": n_calibration,
                **aggregate,
            })
    calibration_grid = pd.DataFrame(grid_rows)
    calibration_grid.to_csv(output_dir / "calibration_grid.csv", index=False)
    selected_threshold = select_common_threshold(calibration_grid, model_names)

    validation_rows = []
    validation_null_rows = []
    if selected_threshold is not None:
        for model_name in model_names:
            null_rows, aggregate = _evaluate_null(
                values_by_model[model_name],
                validation_permutations,
                window_size,
                selected_threshold,
                drift,
                warmup,
            )
            for row in null_rows:
                validation_null_rows.append({"model": model_name, **row})
            detector = CUSUM(
                threshold=selected_threshold, drift=drift, warmup=warmup
            )
            ordered_alarms = cq_cusum_alarm_times(
                values_by_model[model_name], window_size, detector
            )
            first_alarm = ordered_alarms[0] if ordered_alarms else None
            sensitivity_pass = (
                first_alarm is not None
                and transition_start <= first_alarm <= len(qids)
            )
            validation_rows.append({
                "model": model_name,
                "threshold": selected_threshold,
                "n_permutations": n_validation,
                **aggregate,
                "ordered_first_alarm": first_alarm,
                "ordered_n_alarms": len(ordered_alarms),
                "ordered_delay_from_midpoint": (
                    first_alarm - effective_midpoint if first_alarm is not None else None
                ),
                "sensitivity_pass": sensitivity_pass,
                "gate_pass": bool(aggregate["specificity_pass"]) and sensitivity_pass,
            })

    validation_null = pd.DataFrame(validation_null_rows)
    validation_null.to_csv(output_dir / "validation_null.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    overall_pass = bool(
        selected_threshold is not None
        and not validation.empty
        and validation["gate_pass"].all()
    )

    result_manifest = {
        "analysis": "proposal_s1_cq_cusum_split_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(result_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_permutation_manifest_sha256": _sha256(permutation_manifest_path),
        "source_permutation_summary_sha256": _sha256(permutation_summary_path),
        "source_stream_sha256": _sha256(stream_path),
        "source_per_query_sha256": source_hashes,
        "source_seed": source_seed,
        "calibration_seed": calibration_seed,
        "validation_seed": validation_seed,
        "n_calibration": n_calibration,
        "n_validation": n_validation,
        "disjoint_permutation_sets": True,
        "matched_permutations_across_models": True,
        "threshold_grid": [float(value) for value in threshold_grid],
        "selection_rule": (
            "smallest common threshold with calibration 95% Wilson upper "
            "bound <= 0.05 for both models"
        ),
        "validation_specificity_rule": "95% Wilson upper bound <= 0.05 per model",
        "validation_sensitivity_rule": "at least one ordered alarm at t in [15, 90]",
        "transition_start": transition_start,
        "effective_midpoint": effective_midpoint,
        "window_size": window_size,
        "drift_allowance": drift,
        "warmup": warmup,
        "selected_threshold": selected_threshold,
        "no_model_calls": True,
        "overall_gate_pass": overall_pass,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(
        output_dir / "CUSUM_CALIBRATION_ASSESSMENT.md",
        calibration_grid,
        validation,
        selected_threshold,
        overall_pass,
        calibration_seed,
        validation_seed,
        n_calibration,
        n_validation,
    )
    return result_manifest
