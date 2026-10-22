"""Call-free proposal-mapped detector development on the paid S1 K/L records."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.detectors.detectors import build_detector
from reasoning_monitor.evaluation.cusum_calibration import wilson_interval
from reasoning_monitor.evaluation.permutation_controls import (
    generate_matched_permutations,
)

CELL_ALPHA = 0.005
CALIBRATION_SEED = 51042
VALIDATION_SEED = 61042
N_CALIBRATION = 1000
N_VALIDATION = 1000


@dataclass(frozen=True)
class DevelopmentSpec:
    detector: str
    parameter: str
    values: tuple[float, ...]


# Values are ordered from most to least sensitive. Selection uses calibration
# null streams only and takes the first value passing both models.
DEVELOPMENT_SPECS: dict[str, DevelopmentSpec] = {
    "CS": DevelopmentSpec(
        "adwin", "delta", (0.05, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001),
    ),
    "RS": DevelopmentSpec(
        "ddm", "drift_level", (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0),
    ),
    "LS": DevelopmentSpec(
        "page_hinkley",
        "threshold",
        (0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
    ),
    "ES": DevelopmentSpec(
        "ewma", "control_limit", (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
    ),
    "SS": DevelopmentSpec(
        "adwin", "delta", (0.05, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _efficiency_mean(cq_values: Sequence[float], token_counts: Sequence[int]) -> float:
    t_min = min(token_counts)
    t_max = max(token_counts)
    scores = []
    for cq, tokens in zip(cq_values, token_counts, strict=True):
        normalized = 0.0 if t_max == t_min else (tokens - t_min) / (t_max - t_min)
        conciseness = 1.0 - normalized
        denominator = cq + conciseness
        scores.append(0.0 if denominator == 0 else 2.0 * cq * conciseness / denominator)
    return sum(scores) / len(scores)


def build_window_signal(
    metric: str,
    metric_values: Sequence[float],
    cq_values: Sequence[float],
    token_counts: Sequence[int],
    indices: Sequence[int],
    window_size: int = 10,
) -> list[tuple[int, float]]:
    """Rebuild the exact W-step detector input after an item-order permutation."""
    if metric == "ES":
        cq_window: deque[float] = deque()
        token_window: deque[int] = deque()
        signal = []
        for t, index in enumerate(indices, start=1):
            cq_window.append(float(cq_values[index]))
            token_window.append(int(token_counts[index]))
            if len(cq_window) > window_size:
                cq_window.popleft()
                token_window.popleft()
            if len(cq_window) == window_size:
                signal.append(
                    (t, _efficiency_mean(list(cq_window), list(token_window)))
                )
        return signal

    window: deque[float] = deque()
    signal = []
    for t, index in enumerate(indices, start=1):
        value = float(metric_values[index])
        # This reproduces WindowMeanMetric's RS C-subset behavior: NaN items
        # neither enter nor advance the valid-value window.
        if math.isnan(value):
            continue
        window.append(value)
        if len(window) > window_size:
            window.popleft()
        if len(window) == window_size:
            signal.append((t, sum(window) / window_size))
    return signal


def _adwin_first_alarms(
    signal: Sequence[tuple[int, float]],
    deltas: Sequence[float],
    warmup: int = 20,
    max_window: int = 200,
) -> dict[float, int | None]:
    """Return exact first ADWIN alarms for a delta grid using shared split sums."""
    first: dict[float, int | None] = {float(delta): None for delta in deltas}
    window: list[float] = []
    update_count = 0
    for t, value in signal:
        update_count += 1
        window.append(float(value))
        if len(window) > max_window:
            window.pop(0)
        n_items = len(window)
        if n_items < 20 or update_count <= warmup:
            continue
        unresolved = [delta for delta, alarm in first.items() if alarm is None]
        if not unresolved:
            break
        prefix = [0.0]
        for item in window:
            prefix.append(prefix[-1] + item)
        total = prefix[-1]
        step = max(1, n_items // 20)
        splits = []
        for split in range(step, n_items - step + 1, step):
            n_left, n_right = split, n_items - split
            if n_left < 5 or n_right < 5:
                continue
            left_mean = prefix[split] / n_left
            right_mean = (total - prefix[split]) / n_right
            difference = left_mean - right_mean
            effective_n = 1.0 / (1.0 / n_left + 1.0 / n_right)
            splits.append((difference, effective_n))
        for delta in unresolved:
            if any(
                difference
                > math.sqrt((2.0 / effective_n) * math.log(2.0 * n_items / delta))
                for difference, effective_n in splits
            ):
                first[delta] = int(t)
    return first


def first_alarms_for_grid(
    signal: Sequence[tuple[int, float]],
    spec: DevelopmentSpec,
) -> dict[float, int | None]:
    if spec.detector == "adwin":
        return _adwin_first_alarms(signal, spec.values)
    first: dict[float, int | None] = {value: None for value in spec.values}
    detectors = {
        value: build_detector(spec.detector, **{spec.parameter: value})
        for value in spec.values
    }
    for t, value in signal:
        for parameter_value, detector in detectors.items():
            if first[parameter_value] is not None:
                continue
            alarm = detector.update(value, t, "development")
            if alarm is not None:
                first[parameter_value] = int(alarm.t)
    return first


def select_parameters(
    calibration: pd.DataFrame,
    model_names: Sequence[str],
) -> dict[str, float | None]:
    selected = {}
    expected_models = set(model_names)
    for metric, spec in DEVELOPMENT_SPECS.items():
        selected[metric] = None
        metric_rows = calibration[calibration["metric"] == metric]
        for value in spec.values:
            rows = metric_rows[metric_rows["parameter_value"] == value]
            if set(rows["model"]) == expected_models and bool(
                rows["specificity_pass"].all()
            ):
                selected[metric] = value
                break
    return selected


def _null_grid(
    frames: dict[str, pd.DataFrame],
    permutations: Sequence[Sequence[int]],
    window_size: int,
) -> pd.DataFrame:
    counts = {
        (metric, model, value): 0
        for metric, spec in DEVELOPMENT_SPECS.items()
        for model in frames
        for value in spec.values
    }
    for model, frame in frames.items():
        cq_values = frame["CQ_raw"].astype(float).tolist()
        token_counts = frame["n_tokens"].astype(int).tolist()
        for metric, spec in DEVELOPMENT_SPECS.items():
            metric_values = frame[f"{metric}_raw"].astype(float).tolist()
            for indices in permutations:
                signal = build_window_signal(
                    metric,
                    metric_values,
                    cq_values,
                    token_counts,
                    indices,
                    window_size,
                )
                alarms = first_alarms_for_grid(signal, spec)
                for value, first_alarm in alarms.items():
                    counts[(metric, model, value)] += first_alarm is not None

    rows = []
    n_permutations = len(permutations)
    for (metric, model, value), alarm_count in counts.items():
        low, high = wilson_interval(alarm_count, n_permutations)
        spec = DEVELOPMENT_SPECS[metric]
        rows.append({
            "metric": metric,
            "model": model,
            "detector": spec.detector,
            "parameter": spec.parameter,
            "parameter_value": value,
            "n_permutations": n_permutations,
            "null_any_alarm_count": alarm_count,
            "null_any_alarm_rate": alarm_count / n_permutations,
            "null_alarm_wilson_low": low,
            "null_alarm_wilson_high": high,
            "specificity_pass": high <= CELL_ALPHA,
        })
    return pd.DataFrame(rows)


def _validate_selected(
    frames: dict[str, pd.DataFrame],
    permutations: Sequence[Sequence[int]],
    selected: dict[str, float | None],
    window_size: int,
    transition_start: int,
    effective_midpoint: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    null_rows = []
    identity = tuple(range(len(next(iter(frames.values())))))
    for metric, parameter_value in selected.items():
        if parameter_value is None:
            continue
        spec = DEVELOPMENT_SPECS[metric]
        selected_spec = DevelopmentSpec(
            spec.detector, spec.parameter, (parameter_value,)
        )
        for model, frame in frames.items():
            cq_values = frame["CQ_raw"].astype(float).tolist()
            token_counts = frame["n_tokens"].astype(int).tolist()
            metric_values = frame[f"{metric}_raw"].astype(float).tolist()
            alarm_count = 0
            for permutation_id, indices in enumerate(permutations, start=1):
                signal = build_window_signal(
                    metric,
                    metric_values,
                    cq_values,
                    token_counts,
                    indices,
                    window_size,
                )
                first_alarm = first_alarms_for_grid(signal, selected_spec)[
                    parameter_value
                ]
                alarm_count += first_alarm is not None
                null_rows.append({
                    "metric": metric,
                    "model": model,
                    "permutation_id": permutation_id,
                    "first_alarm": first_alarm,
                    "any_alarm": first_alarm is not None,
                })
            low, high = wilson_interval(alarm_count, len(permutations))
            ordered_signal = build_window_signal(
                metric,
                metric_values,
                cq_values,
                token_counts,
                identity,
                window_size,
            )
            ordered_first = first_alarms_for_grid(ordered_signal, selected_spec)[
                parameter_value
            ]
            early_warning_pass = bool(
                ordered_first is not None
                and transition_start <= ordered_first < effective_midpoint
            )
            specificity_pass = high <= CELL_ALPHA
            summary_rows.append({
                "metric": metric,
                "model": model,
                "detector": spec.detector,
                "parameter": spec.parameter,
                "parameter_value": parameter_value,
                "n_permutations": len(permutations),
                "null_any_alarm_count": alarm_count,
                "null_any_alarm_rate": alarm_count / len(permutations),
                "null_alarm_wilson_low": low,
                "null_alarm_wilson_high": high,
                "specificity_pass": specificity_pass,
                "ordered_first_alarm": ordered_first,
                "ordered_delay_from_midpoint": (
                    ordered_first - effective_midpoint
                    if ordered_first is not None
                    else None
                ),
                "early_warning_pass": early_warning_pass,
                "cell_pass": specificity_pass and early_warning_pass,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def _write_assessment(
    path: Path,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    selected: dict[str, float | None],
    candidates: dict[str, bool],
) -> None:
    lines = [
        "# S1 Proposal-Mapped Detector Development",
        "",
        "This is a call-free development analysis, not a confirmatory H3 test. ",
        "Parameters were selected only from calibration null permutations. CQ ",
        "remains fixed at its separately calibrated threshold 7.5.",
        "",
        "## Calibration Selection",
        "",
        "| Metric | Detector | Parameter | Selected | Worst calibration upper |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for metric, spec in DEVELOPMENT_SPECS.items():
        value = selected[metric]
        if value is None:
            selected_text = "none"
            worst_text = "-"
        else:
            selected_text = f"{value:g}"
            rows = calibration[
                (calibration["metric"] == metric)
                & (calibration["parameter_value"] == value)
            ]
            worst_text = f"{rows['null_alarm_wilson_high'].max():.4f}"
        lines.append(
            f"| {metric} | {spec.detector} | {spec.parameter} | "
            f"{selected_text} | {worst_text} |"
        )

    lines.extend([
        "",
        "## Held-Out Validation And Ordered Stream",
        "",
        "| Metric | Model | Null alarms | Upper | First alarm | Delay | Specificity | Early |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in validation.itertuples(index=False):
        first = "-" if pd.isna(row.ordered_first_alarm) else str(int(row.ordered_first_alarm))
        delay = (
            "-"
            if pd.isna(row.ordered_delay_from_midpoint)
            else f"{int(row.ordered_delay_from_midpoint):+d}"
        )
        lines.append(
            f"| {row.metric} | {row.model} | {row.null_any_alarm_count}/"
            f"{row.n_permutations} | {row.null_alarm_wilson_high:.4f} | "
            f"{first} | {delay} | {'pass' if row.specificity_pass else 'fail'} | "
            f"{'pass' if row.early_warning_pass else 'fail'} |"
        )

    passing = [metric for metric, passed in candidates.items() if passed]
    decision = (
        "Development candidates meeting held-out specificity and ordered early-warning "
        f"criteria in both models: {', '.join(passing)}. They require a fresh held-out "
        "stream before confirmatory use."
        if passing
        else
        "No proposal-mapped metric met held-out specificity and ordered early-warning "
        "criteria in both models. Do not tune parameters against the ordered outcomes."
    )
    lines.extend(["", "## Decision", "", decision, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def develop_s1_detectors(
    result_dir: str | Path,
    n_calibration: int = N_CALIBRATION,
    n_validation: int = N_VALIDATION,
    calibration_seed: int = CALIBRATION_SEED,
    validation_seed: int = VALIDATION_SEED,
) -> dict[str, Any]:
    """Develop proposal-mapped candidates without issuing model calls."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    health_manifest_path = result_dir / "kl_validation_manifest.json"
    health_gate_path = result_dir / "data" / "kl_validation_gate.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    health_manifest = json.loads(health_manifest_path.read_text(encoding="utf-8"))
    health_gate = pd.read_csv(health_gate_path)
    config = manifest["config"]

    if config.get("stream_design") != "proposal_multidataset":
        raise ValueError("source is not a proposal-multidataset run")
    if config.get("scenarios") != ["S1"]:
        raise ValueError("source must contain only S1")
    if int(config.get("k_repeats", 0)) != 5 or int(config.get("l_paraphrases", -1)) != 3:
        raise ValueError("detector development requires the paid K=5/L=3 run")
    if health_manifest.get("overall_gate_pass") is not False:
        raise ValueError("Section 39 requires the failed Section 37 gate")
    if health_manifest.get("source_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("source manifest changed after the metric-health assessment")
    health_columns = [
        "role_counts_ok",
        "k_repeat_diversity_pass",
        "rs_coverage_pass",
        "metric_coverage_pass",
        "metrics_in_range",
        "nondegenerate",
        "preflight_pass",
    ]
    if any(not health_gate[column].map(_is_true).all() for column in health_columns):
        raise ValueError("a required metric-health criterion did not pass")
    if health_gate["cq_delay_gate_pass"].map(_is_true).any():
        raise ValueError("Section 39 expects both CQ-delay cells to have failed")

    seeds = config.get("seeds") or [config.get("seed")]
    windows = config.get("window_sizes") or []
    models = config.get("models") or []
    if len(seeds) != 1 or windows != [10] or len(models) != 2:
        raise ValueError("development requires one seed, W=10 and two models")
    source_seed = int(seeds[0])
    model_names = [str(model["name"]) for model in models]
    data_dir = result_dir / "data"
    stream_path = data_dir / f"stream_S1_seed{source_seed}.csv"
    qids = pd.read_csv(stream_path).sort_values("t")["qid"].astype(str).tolist()
    if len(qids) != 90:
        raise ValueError("development protocol requires exactly 90 S1 queries")

    frames = {}
    source_hashes = {}
    for model_name in model_names:
        path = data_dir / f"per_query_S1_{model_name}_seed{source_seed}_W10.csv"
        frame = pd.read_csv(path).sort_values("t").reset_index(drop=True)
        if frame["qid"].astype(str).tolist() != qids:
            raise ValueError(f"per-query order mismatch for {model_name}")
        required = {"CQ_raw", "CS_raw", "RS_raw", "LS_raw", "ES_raw", "SS_raw", "n_tokens"}
        if not required.issubset(frame.columns):
            raise ValueError(f"missing development columns for {model_name}")
        frames[model_name] = frame
        source_hashes[model_name] = _sha256(path)

    detection = pd.read_csv(data_dir / "detection_results.csv")
    s1 = detection[detection["scenario"] == "S1"]
    transition_starts = s1["transition_start"].dropna().astype(int).unique()
    midpoints = s1["effective_t_star"].dropna().astype(int).unique()
    if len(transition_starts) != 1 or len(midpoints) != 1:
        raise ValueError("S1 transition metadata is not unique")
    transition_start = int(transition_starts[0])
    effective_midpoint = int(midpoints[0])

    calibration_permutations = generate_matched_permutations(
        len(qids), n_calibration, calibration_seed
    )
    validation_permutations = generate_matched_permutations(
        len(qids), n_validation, validation_seed
    )
    if set(calibration_permutations) & set(validation_permutations):
        raise ValueError("calibration and validation permutations overlap")

    output_dir = result_dir / "detector_development"
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = _null_grid(frames, calibration_permutations, 10)
    calibration.to_csv(output_dir / "calibration_grid.csv", index=False)
    selected = select_parameters(calibration, model_names)
    validation, validation_null = _validate_selected(
        frames,
        validation_permutations,
        selected,
        10,
        transition_start,
        effective_midpoint,
    )
    validation.to_csv(output_dir / "validation_summary.csv", index=False)
    validation_null.to_csv(output_dir / "validation_null.csv", index=False)

    candidates = {}
    for metric in DEVELOPMENT_SPECS:
        rows = validation[validation["metric"] == metric]
        candidates[metric] = bool(
            selected[metric] is not None
            and set(rows["model"]) == set(model_names)
            and rows["cell_pass"].all()
        )

    result = {
        "analysis": "proposal_s1_call_free_detector_development",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(result_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_health_manifest_sha256": _sha256(health_manifest_path),
        "source_health_gate_sha256": _sha256(health_gate_path),
        "source_stream_sha256": _sha256(stream_path),
        "source_per_query_sha256": source_hashes,
        "no_model_calls": True,
        "development_only": True,
        "proposal_mapping": {
            metric: {
                "detector": spec.detector,
                "parameter": spec.parameter,
                "grid": list(spec.values),
            }
            for metric, spec in DEVELOPMENT_SPECS.items()
        },
        "selection_rule": (
            "first most-sensitive parameter whose calibration 95% Wilson upper "
            "bound is <=0.005 for both models"
        ),
        "validation_specificity_rule": (
            "95% Wilson upper bound <=0.005 for every metric-model cell; "
            "Bonferroni family of five metrics by two models"
        ),
        "ordered_early_warning_rule": "first alarm at t in [15, 44] in both models",
        "calibration_seed": calibration_seed,
        "validation_seed": validation_seed,
        "n_calibration": n_calibration,
        "n_validation": n_validation,
        "disjoint_permutation_sets": True,
        "matched_permutations_across_metrics_and_models": True,
        "selected_parameters": selected,
        "candidate_pass": candidates,
        "passing_candidates": [
            metric for metric, passed in candidates.items() if passed
        ],
        "requires_fresh_confirmatory_stream": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(
        output_dir / "DETECTOR_DEVELOPMENT_ASSESSMENT.md",
        calibration,
        validation,
        selected,
        candidates,
    )
    return result
