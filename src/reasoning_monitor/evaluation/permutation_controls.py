"""Permutation calibration for S1 CQ trend and CUSUM specificity."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import pandas as pd

from reasoning_monitor.detectors.assignment import assign_detector
from reasoning_monitor.metrics.dimensions import CorrectnessQuality


def generate_matched_permutations(
    n_items: int,
    n_permutations: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Return deterministic, unique, non-identity index permutations."""
    if n_items < 2:
        raise ValueError("at least two items are required")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    rng = random.Random(seed)
    identity = tuple(range(n_items))
    seen = {identity}
    permutations = []
    while len(permutations) < n_permutations:
        indices = list(identity)
        rng.shuffle(indices)
        candidate = tuple(indices)
        if candidate not in seen:
            seen.add(candidate)
            permutations.append(candidate)
    return permutations


def decline_score(values: Sequence[float]) -> float:
    """Negative OLS slope of CQ on normalized time; larger means more decline."""
    n_items = len(values)
    if n_items < 2:
        raise ValueError("at least two CQ values are required")
    x_values = [index / (n_items - 1) for index in range(n_items)]
    x_mean = 0.5
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    slope = sum(
        (x_value - x_mean) * float(value)
        for x_value, value in zip(x_values, values, strict=True)
    ) / denominator
    return -slope


def cq_cusum_alarm_times(
    values: Sequence[float],
    window_size: int,
    detector_template,
) -> list[int]:
    """Replay a raw CQ sequence through the source run's windowed detector."""
    if len(values) < window_size:
        raise ValueError("CQ sequence is shorter than the window")
    detector = copy.deepcopy(detector_template)
    detector.reset()
    window_sum = sum(float(value) for value in values[:window_size])
    alarm_times = []
    for end in range(window_size, len(values) + 1):
        if end > window_size:
            window_sum += float(values[end - 1]) - float(values[end - window_size - 1])
        window_mean = window_sum / window_size
        alarm = detector.update(window_mean, end, "CQ")
        if alarm is not None:
            alarm_times.append(int(alarm.t))
    return alarm_times


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson_interval(successes: int, trials: int, z: float = 1.9599639845):
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _permutation_p(observed: float, null_values: Sequence[float]) -> float:
    return (1 + sum(value >= observed for value in null_values)) / (len(null_values) + 1)


def _write_assessment(
    path: Path,
    summary: pd.DataFrame,
    overall_pass: bool,
    seed: int,
    n_permutations: int,
) -> None:
    lines = [
        "# S1 CQ Permutation Assessment",
        "",
        "This call-free analysis uses the exact 90 saved CQ outcomes per model and ",
        f"`{n_permutations}` matched, unique permutations with seed `{seed}`. The same ",
        "item-order permutations are applied to both models.",
        "",
        "## Frozen Criteria",
        "",
        "The temporal signal is the negative OLS slope of raw CQ on normalized time. ",
        "Larger values indicate a stronger decline. Its one-sided randomization p-value ",
        "must be at most 0.025 for each model, a Bonferroni correction for two models. ",
        "CUSUM specificity passes only when the upper bound of its two-sided 95% Wilson ",
        "interval for permutation-level false alarms is at most 0.05. Both criteria must ",
        "pass for both models.",
        "",
        "## Results",
        "",
        "| Model | Decline score | Trend p | First alarm | Alarm-time p | Null alarm rate | 95% CI | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        first = "-" if pd.isna(row.ordered_first_alarm) else str(int(row.ordered_first_alarm))
        lines.append(
            f"| {row.model} | {row.observed_decline_score:.4f} | "
            f"{row.decline_p_value:.4f} | {first} | "
            f"{row.alarm_earliness_p_value:.4f} | "
            f"{row.null_any_alarm_rate:.3f} | "
            f"[{row.null_alarm_wilson_low:.3f}, {row.null_alarm_wilson_high:.3f}] | "
            f"{'PASS' if row.gate_pass else 'FAIL'} |"
        )

    trend_pass = bool(summary["trend_pass"].all())
    specificity_pass = bool(summary["specificity_pass"].all())
    if overall_pass:
        interpretation = (
            "Both the temporal decline and detector-specificity criteria pass. This new "
            "gate authorises planning the paid K/L validation, while the earlier single-"
            "shuffle gate remains historically failed."
        )
    elif trend_pass and not specificity_pass:
        interpretation = (
            "The ordered CQ decline is stronger than the permutation null for both models, "
            "but the current CQ-CUSUM threshold does not control stream-level false alarms. "
            "The ordered first-alarm times are also not unusually early under that null. "
            "The S1 input trajectory is supported; the detector calibration is not."
        )
    elif specificity_pass and not trend_pass:
        interpretation = (
            "CUSUM specificity is acceptable, but the ordered CQ decline is not significant "
            "for both models under the corrected threshold."
        )
    else:
        interpretation = (
            "Neither the cross-model temporal-signal requirement nor CUSUM specificity is "
            "fully supported under this protocol."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        interpretation,
        "",
        "A failed specificity criterion does not authorise post-hoc threshold tuning on ",
        "these ordered outcomes. Any calibration step must freeze its threshold-selection ",
        "rule and validation split before execution.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_s1_permutations(
    result_dir: str | Path,
    n_permutations: int = 1000,
    permutation_seed: int = 21042,
) -> dict[str, Any]:
    """Evaluate S1 CQ trend and CUSUM false alarms under matched permutations."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    control_manifest_path = result_dir / "order_controls" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if config.get("stream_design") != "proposal_multidataset":
        raise ValueError("source run is not proposal_multidataset")
    if control_manifest.get("overall_gate_pass") is not False:
        raise ValueError("permutation reassessment requires the failed order-control gate")
    if control_manifest.get("source_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("source manifest changed after the order-control analysis")

    seeds = config.get("seeds") or [config.get("seed")]
    windows = config.get("window_sizes") or []
    if len(seeds) != 1 or len(windows) != 1:
        raise ValueError("permutation analysis requires one seed and one window size")
    source_seed, window_size = int(seeds[0]), int(windows[0])
    models = config.get("models") or []
    if len(models) != 2:
        raise ValueError("the preregistered correction assumes exactly two models")

    data_dir = result_dir / "data"
    output_dir = result_dir / "order_controls" / "permutation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = data_dir / f"stream_S1_seed{source_seed}.csv"
    stream = pd.read_csv(stream_path).sort_values("t")
    qids = stream["qid"].astype(str).tolist()
    permutations = generate_matched_permutations(
        len(qids), n_permutations, permutation_seed
    )

    detector_template = assign_detector(
        CorrectnessQuality(window_size),
        strategy=str(config.get("detector_strategy", "distribution_based")),
    )
    null_rows = []
    summary_rows = []
    source_hashes: dict[str, dict[str, str]] = {}
    for model in models:
        model_name = str(model["name"])
        per_query_path = (
            data_dir / f"per_query_S1_{model_name}_seed{source_seed}_W{window_size}.csv"
        )
        records_path = (
            data_dir / f"records_S1_{model_name}_seed{source_seed}_W{window_size}.jsonl"
        )
        per_query = pd.read_csv(per_query_path).sort_values("t")
        if per_query["qid"].astype(str).tolist() != qids:
            raise ValueError(f"per-query order mismatch for {model_name}")
        if per_query["call_failed"].astype(bool).any():
            raise ValueError(f"saved call failure encountered for {model_name}")
        values = [float(value) for value in per_query["CQ_raw"].tolist()]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"invalid CQ values for {model_name}")

        observed_decline = decline_score(values)
        ordered_alarms = cq_cusum_alarm_times(
            values, window_size, detector_template
        )
        original_alarms = pd.read_csv(
            data_dir / f"alarms_S1_{model_name}_seed{source_seed}_W{window_size}.csv"
        )
        original_times = original_alarms.loc[
            (original_alarms["metric"] == "CQ")
            & (original_alarms["detector"] == "cusum")
            & (original_alarms["level"] == "drift"),
            "t",
        ].astype(int).tolist()
        if ordered_alarms != original_times:
            raise ValueError(
                f"ordered CUSUM replay mismatch for {model_name}: "
                f"{ordered_alarms} != {original_times}"
            )

        null_declines = []
        null_alarm_scores = []
        null_any_alarm = 0
        for permutation_id, indices in enumerate(permutations, start=1):
            permuted = [values[index] for index in indices]
            null_decline = decline_score(permuted)
            alarm_times = cq_cusum_alarm_times(
                permuted, window_size, detector_template
            )
            first_alarm = alarm_times[0] if alarm_times else None
            alarm_score = len(values) + 1 - first_alarm if first_alarm is not None else 0
            null_declines.append(null_decline)
            null_alarm_scores.append(alarm_score)
            null_any_alarm += bool(alarm_times)
            null_rows.append({
                "model": model_name,
                "permutation_id": permutation_id,
                "decline_score": null_decline,
                "first_cusum_alarm": first_alarm,
                "n_cusum_alarms": len(alarm_times),
                "any_cusum_alarm": bool(alarm_times),
            })

        decline_p = _permutation_p(observed_decline, null_declines)
        ordered_first = ordered_alarms[0] if ordered_alarms else None
        ordered_alarm_score = (
            len(values) + 1 - ordered_first if ordered_first is not None else 0
        )
        alarm_earliness_p = _permutation_p(ordered_alarm_score, null_alarm_scores)
        false_alarm_rate = null_any_alarm / n_permutations
        wilson_low, wilson_high = _wilson_interval(
            null_any_alarm, n_permutations
        )
        trend_pass = decline_p <= 0.025
        specificity_pass = wilson_high <= 0.05
        summary_rows.append({
            "model": model_name,
            "n_queries": len(values),
            "n_permutations": n_permutations,
            "observed_decline_score": observed_decline,
            "null_decline_mean": fmean(null_declines),
            "null_decline_sd": pstdev(null_declines),
            "decline_p_value": decline_p,
            "trend_alpha_bonferroni": 0.025,
            "trend_pass": trend_pass,
            "ordered_first_alarm": ordered_first,
            "ordered_n_alarms": len(ordered_alarms),
            "alarm_earliness_p_value": alarm_earliness_p,
            "null_any_alarm_count": null_any_alarm,
            "null_any_alarm_rate": false_alarm_rate,
            "null_alarm_wilson_low": wilson_low,
            "null_alarm_wilson_high": wilson_high,
            "specificity_upper_limit": 0.05,
            "specificity_pass": specificity_pass,
            "gate_pass": trend_pass and specificity_pass,
        })
        source_hashes[model_name] = {
            "per_query_sha256": _sha256(per_query_path),
            "records_sha256": _sha256(records_path),
        }

    null_frame = pd.DataFrame(null_rows)
    null_frame.to_csv(output_dir / "permutation_null.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "permutation_summary.csv", index=False)
    overall_pass = bool(summary["gate_pass"].all())
    permutation_manifest = {
        "analysis": "proposal_s1_cq_permutation_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(result_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_order_control_manifest_sha256": _sha256(control_manifest_path),
        "source_stream_sha256": _sha256(stream_path),
        "source_hashes": source_hashes,
        "source_seed": source_seed,
        "permutation_seed": permutation_seed,
        "n_permutations": n_permutations,
        "matched_permutations_across_models": True,
        "unique_non_identity_permutations": True,
        "window_size": window_size,
        "detector_strategy": config.get("detector_strategy", "distribution_based"),
        "metric": "CQ",
        "no_model_calls": True,
        "trend_statistic": "negative OLS slope of raw CQ on normalized time",
        "trend_alpha_per_model": 0.025,
        "specificity_rule": "upper 95% Wilson bound for null any-alarm rate <= 0.05",
        "overall_gate_pass": overall_pass,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(permutation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(
        output_dir / "PERMUTATION_ASSESSMENT.md",
        summary,
        overall_pass,
        permutation_seed,
        n_permutations,
    )
    return permutation_manifest
