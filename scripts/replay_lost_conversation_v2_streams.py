"""Run the frozen call-free LIC-RQ-v2-SR session-stream detector replay.

Protocol: LIC_RQ_V2_STREAM_REPLAY_PROTOCOL.md. No model or network call.
Resumable: every (phase, model, W) unit is cached under the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation import stream_replay as sr  # noqa: E402
from reasoning_monitor.evaluation.cusum_calibration import wilson_interval  # noqa: E402
from reasoning_monitor.progress import format_duration  # noqa: E402

SCIENCE = ROOT / "results/lost_conversation_v2_science"
OUTPUT = ROOT / "results/lost_conversation_v2_stream_replay"
PROTOCOL = ROOT / "LIC_RQ_V2_STREAM_REPLAY_PROTOCOL.md"
FROZEN_INPUTS = {
    "records": (SCIENCE / "data/records.jsonl",
                "e313f1ce59854d2590efe5602cee4799a917664f6f389426412fe433ea1772ba"),
    "metrics": (SCIENCE / "metrics.csv",
                "7a9331e678b4f37b7fa9d62336a1634a3f1af42ce00c7e410324c68ba122c1c9"),
    "confirmatory": (SCIENCE / "confirmatory/assessment.json",
                     "1be56fb6bb98e6f8348202dfea9804d2c2ed57582fbf78a1586e07790589bcf1"),
}
MODES = ("session_mean", "final_turn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict:
    report = {}
    for name, (path, expected) in FROZEN_INPUTS.items():
        observed = sha256(path)
        report[name] = {"path": str(path.relative_to(ROOT)), "sha256": observed,
                        "expected": expected, "match": observed == expected}
        if observed != expected:
            raise SystemExit(f"Frozen input hash mismatch for {name}: {observed}")
    return report


def empty_primary_audit() -> list[dict]:
    rows = []
    with FROZEN_INPUTS["records"][0].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if (record.get("role") == "primary" and record.get("ok")
                    and not str(record.get("raw_text") or "").strip()):
                rows.append({key: record.get(key) for key in
                             ("model", "task_id", "condition", "turn", "n_tokens")})
    return rows


# ── worker units (top level for Windows spawn) ───────────────────────────────

def calibrate_unit(model, window, values, cells, streams, include_raw):
    counts = {cell: {v: 0 for v in sr.GRIDS[cell[1]].values} for cell in cells}
    raw_counts = {v: 0 for v in sr.GRIDS["ddm"].values} if include_raw else None
    for stream in streams:
        signals = {metric: sr.window_signal(values[metric], stream, window)
                   for metric in sr.METRICS}
        for metric, detector in cells:
            for value, first in sr.grid_first_alarms(signals[metric], detector).items():
                if first is not None:
                    counts[(metric, detector)][value] += 1
        if include_raw:
            raw = sr.raw_signal(values["CQ"], stream)
            for value, first in sr.grid_first_alarms(raw, "ddm").items():
                if first is not None:
                    raw_counts[value] += 1
    return {"model": model, "window": window, "counts": counts, "raw_counts": raw_counts}


def _any_alarm(signal, detector, value):
    return bool(sr.alarm_times(signal, detector, value, stop_at=0))


def validate_unit(model, window, values, locked, raw_value, streams, stationary):
    counts = {cell: 0 for cell in locked}
    raw_count = 0
    station = {label: {cell: 0 for cell in locked} for label in stationary}
    for stream in streams:
        for (metric, detector), value in locked.items():
            signal = sr.window_signal(values[metric], stream, window)
            counts[(metric, detector)] += _any_alarm(signal, detector, value)
        if raw_value is not None:
            raw_count += _any_alarm(sr.raw_signal(values["CQ"], stream), "ddm", raw_value)
    raw_station = {label: 0 for label in stationary}
    for label, group in stationary.items():
        for stream in group:
            for (metric, detector), value in locked.items():
                signal = sr.window_signal(values[metric], stream, window)
                station[label][(metric, detector)] += _any_alarm(signal, detector, value)
            if raw_value is not None:
                raw_station[label] += _any_alarm(
                    sr.raw_signal(values["CQ"], stream), "ddm", raw_value)
    return {"model": model, "window": window, "counts": counts, "raw_count": raw_count,
            "stationary": station, "raw_stationary": raw_station}


def ordered_unit(model, window, values, validated, raw_value, scenario_streams):
    from scipy.stats import mannwhitneyu

    rows, bands, h1 = [], [], []
    for label, streams in scenario_streams.items():
        base = sr.SCENARIOS[label.replace("_boot", "")]
        keep_detail = not label.endswith("_boot")
        trajectories = {metric: np.full((len(streams), 96), np.nan) for metric in sr.METRICS}
        for realization, stream in enumerate(streams):
            signals = {metric: sr.window_signal(values[metric], stream, window)
                       for metric in sr.METRICS}
            for (metric, detector), value in validated.items():
                times = sr.alarm_times(signals[metric], detector, value, stop_at=base.t_s)
                first = times[0] if times else None
                rows.append({"model": model, "scenario": label, "window": window,
                             "realization": realization, "metric": metric,
                             "detector": detector, "parameter": value,
                             "first_alarm": first,
                             "first_post_ts": sr.first_at_or_after(times, base.t_s),
                             "category": sr.categorize(first, base)})
            if raw_value is not None and window == sr.WINDOWS[0]:
                raw = sr.raw_signal(values["CQ"], stream)
                times = sr.alarm_times(raw, "ddm", raw_value, stop_at=base.t_s)
                first = times[0] if times else None
                rows.append({"model": model, "scenario": label, "window": 0,
                             "realization": realization, "metric": sr.RAW_CHANNEL,
                             "detector": "ddm", "parameter": raw_value,
                             "first_alarm": first,
                             "first_post_ts": sr.first_at_or_after(times, base.t_s),
                             "category": sr.categorize(first, base)})
            if not keep_detail:
                continue
            for metric in sr.METRICS:
                for t, v in signals[metric]:
                    trajectories[metric][realization, t - 1] = v
                ordered_values = [(t, float(values[metric][index]))
                                  for t, index in enumerate(stream, start=1)
                                  if math.isfinite(float(values[metric][index]))]
                pre = [v for t, v in ordered_values if t < base.t_s][-window:]
                post = [v for _, v in ordered_values][-window:]
                if len(pre) >= 2 and len(post) >= 2 and len(set(pre + post)) > 1:
                    p_value = float(mannwhitneyu(post, pre, alternative="two-sided").pvalue)
                else:
                    p_value = float("nan")
                h1.append({"model": model, "scenario": label, "window": window,
                           "metric": metric, "realization": realization,
                           "n_pre": len(pre), "n_post": len(post),
                           "pre_mean": float(np.mean(pre)) if pre else float("nan"),
                           "post_mean": float(np.mean(post)) if post else float("nan"),
                           "cliffs_delta": sr.cliffs_delta(post, pre),
                           "mann_whitney_p": p_value})
        if keep_detail:
            for metric, matrix in trajectories.items():
                support = np.isfinite(matrix).sum(axis=0)
                with warnings.catch_warnings():
                    # Columns before the window fills are all-NaN by design.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    quantiles = {q: np.nanquantile(matrix, q, axis=0)
                                 for q in (0.1, 0.5, 0.9)}
                for t in range(96):
                    bands.append({"model": model, "scenario": label, "window": window,
                                  "metric": metric, "t": t + 1,
                                  "support": int(support[t]),
                                  "p10": float(quantiles[0.1][t]) if support[t] else None,
                                  "median": float(quantiles[0.5][t]) if support[t] else None,
                                  "p90": float(quantiles[0.9][t]) if support[t] else None})
    return {"model": model, "window": window, "rows": rows, "bands": bands, "h1": h1}


# ── orchestration ───────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int, label: str):
        self.total, self.label, self.done = total, label, 0
        self.started = time.monotonic()

    def tick(self, cached: bool = False) -> None:
        self.done += 1
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed * 60 if elapsed > 0 else 0.0
        eta = (self.total - self.done) / (rate / 60) if rate > 0 else None
        note = " (cached)" if cached else ""
        print(f"{self.label}: {self.done}/{self.total} units "
              f"({self.done / self.total * 100:5.1f}%) | {rate:5.2f} units/min | "
              f"ETA {format_duration(eta)}{note}", flush=True)


def run_units(label, jobs, cache_dir: Path, workers: int):
    """jobs: list of (key, function, args). Returns {key: result}."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(len(jobs), label)
    results, pending = {}, []
    for key, function, args in jobs:
        path = cache_dir / f"{key}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                results[key] = pickle.load(handle)
            progress.tick(cached=True)
        else:
            pending.append((key, function, args))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(function, *args): key for key, function, args in pending}
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                with (cache_dir / f"{key}.pkl").open("wb") as handle:
                    pickle.dump(result, handle)
                results[key] = result
                progress.tick()
    return results


def run_mode(mode, tables, workers, cache_root, input_digest):
    cache = cache_root / f"{input_digest[:12]}_{mode}"
    models = sorted(tables)
    n_tasks = tables[models[0]].n_tasks
    n_items = 2 * n_tasks
    cells = sr.calibration_cells()

    calibration_streams = sr.null_streams(n_items, sr.N_CALIBRATION, sr.SEEDS["calibration"])
    validation_streams = sr.null_streams(n_items, sr.N_VALIDATION, sr.SEEDS["validation"],
                                         exclude=calibration_streams)
    stationary = sr.stationary_streams(n_tasks, sr.N_STATIONARY, sr.SEEDS["stationary"])
    scenario_streams = {
        "sudden": sr.ordered_streams("sudden", n_tasks, sr.N_ORDERED, sr.SEEDS["sudden"]),
        "gradual": sr.ordered_streams("gradual", n_tasks, sr.N_ORDERED, sr.SEEDS["gradual"]),
    }
    if mode == "session_mean":
        boot_rng_seed = sr.SEEDS["task_bootstrap"]
        scenario_streams["sudden_boot"] = sr.ordered_streams(
            "sudden", n_tasks, sr.N_BOOTSTRAP_STREAMS, boot_rng_seed, bootstrap=True)
        scenario_streams["gradual_boot"] = sr.ordered_streams(
            "gradual", n_tasks, sr.N_BOOTSTRAP_STREAMS, boot_rng_seed + 1, bootstrap=True)

    # Phase 1: calibration on null streams only.
    jobs = [(f"calibration_{model}_W{w}", calibrate_unit,
             (model, w, tables[model].values, cells, calibration_streams,
              w == sr.WINDOWS[0]))
            for model in models for w in sr.WINDOWS]
    calibration = run_units(f"[{mode}] calibration", jobs, cache, workers)

    locked, calibration_rows = {}, []
    for w in sr.WINDOWS:
        for cell in cells:
            per_model = {m: calibration[f"calibration_{m}_W{w}"]["counts"][cell] for m in models}
            values = sr.GRIDS[cell[1]].values
            locked[(w, cell)] = sr.select_parameter(per_model, sr.N_CALIBRATION, values,
                                                    sr.CELL_ALPHA)
            for model in models:
                for value in values:
                    count = per_model[model][value]
                    low, high = wilson_interval(count, sr.N_CALIBRATION)
                    calibration_rows.append({
                        "window": w, "metric": cell[0], "detector": cell[1], "model": model,
                        "parameter_value": value, "null_alarms": count,
                        "n_streams": sr.N_CALIBRATION, "wilson_low": low,
                        "wilson_high": high, "selected": value == locked[(w, cell)]})
    raw_per_model = {m: calibration[f"calibration_{m}_W{sr.WINDOWS[0]}"]["raw_counts"]
                     for m in models}
    raw_locked = sr.select_parameter(raw_per_model, sr.N_CALIBRATION,
                                     sr.GRIDS["ddm"].values, sr.RAW_CELL_ALPHA)
    for model in models:
        for value in sr.GRIDS["ddm"].values:
            count = raw_per_model[model][value]
            low, high = wilson_interval(count, sr.N_CALIBRATION)
            calibration_rows.append({
                "window": 0, "metric": sr.RAW_CHANNEL, "detector": "ddm", "model": model,
                "parameter_value": value, "null_alarms": count, "n_streams": sr.N_CALIBRATION,
                "wilson_low": low, "wilson_high": high, "selected": value == raw_locked})
    locked_summary = {f"W{w}|{c[0]}|{c[1]}": v for (w, c), v in locked.items()}
    locked_summary[f"raw|{sr.RAW_CHANNEL}|ddm"] = raw_locked
    print(f"[{mode}] locked parameters: {json.dumps(locked_summary)}", flush=True)

    # Phase 2: validation of locked parameters only.
    jobs = []
    for model in models:
        for w in sr.WINDOWS:
            unit_locked = {cell: locked[(w, cell)] for cell in cells
                           if locked[(w, cell)] is not None}
            jobs.append((f"validation_{model}_W{w}", validate_unit,
                         (model, w, tables[model].values, unit_locked,
                          raw_locked if w == sr.WINDOWS[0] else None,
                          validation_streams, stationary)))
    validation = run_units(f"[{mode}] validation", jobs, cache, workers)

    validated, validation_rows = {}, []
    for w in sr.WINDOWS:
        for cell in cells:
            value = locked[(w, cell)]
            if value is None:
                for model in models:
                    validation_rows.append({"window": w, "metric": cell[0],
                                            "detector": cell[1], "model": model,
                                            "parameter_value": None,
                                            "status": "no_qualifying_calibration_value"})
                continue
            ok = True
            for model in models:
                unit = validation[f"validation_{model}_W{w}"]
                count = unit["counts"][cell]
                low, high = wilson_interval(count, sr.N_VALIDATION)
                passed = sr.passes_bound(count, sr.N_VALIDATION, sr.CELL_ALPHA)
                ok &= passed
                validation_rows.append({
                    "window": w, "metric": cell[0], "detector": cell[1], "model": model,
                    "parameter_value": value, "null_alarms": count,
                    "n_streams": sr.N_VALIDATION, "wilson_low": low, "wilson_high": high,
                    "pass": passed,
                    "stationary_fresh_alarm_rate":
                        unit["stationary"]["fresh_only"][cell] / sr.N_STATIONARY,
                    "stationary_dialogue_alarm_rate":
                        unit["stationary"]["dialogue_only"][cell] / sr.N_STATIONARY,
                    "status": "validated" if passed else "validation_failed"})
            if ok:
                validated[(w, cell)] = value
    raw_validated = None
    if raw_locked is not None:
        raw_ok = True
        for model in models:
            unit = validation[f"validation_{model}_W{sr.WINDOWS[0]}"]
            passed = sr.passes_bound(unit["raw_count"], sr.N_VALIDATION, sr.RAW_CELL_ALPHA)
            raw_ok &= passed
            low, high = wilson_interval(unit["raw_count"], sr.N_VALIDATION)
            validation_rows.append({
                "window": 0, "metric": sr.RAW_CHANNEL, "detector": "ddm", "model": model,
                "parameter_value": raw_locked, "null_alarms": unit["raw_count"],
                "n_streams": sr.N_VALIDATION, "wilson_low": low, "wilson_high": high,
                "pass": passed,
                "stationary_fresh_alarm_rate":
                    unit["raw_stationary"]["fresh_only"] / sr.N_STATIONARY,
                "stationary_dialogue_alarm_rate":
                    unit["raw_stationary"]["dialogue_only"] / sr.N_STATIONARY,
                "status": "validated" if passed else "validation_failed"})
        raw_validated = raw_locked if raw_ok else None

    # Phase 3: ordered streams, validated cells only.
    jobs = []
    for model in models:
        for w in sr.WINDOWS:
            unit_cells = {cell: v for (ww, cell), v in validated.items() if ww == w}
            jobs.append((f"ordered_{model}_W{w}", ordered_unit,
                         (model, w, tables[model].values, unit_cells,
                          raw_validated if w == sr.WINDOWS[0] else None,
                          scenario_streams)))
    ordered = run_units(f"[{mode}] ordered streams", jobs, cache, workers)

    rows = pd.DataFrame([r for unit in ordered.values() for r in unit["rows"]])
    bands = pd.DataFrame([r for unit in ordered.values() for r in unit["bands"]])
    h1 = pd.DataFrame([r for unit in ordered.values() for r in unit["h1"]])
    return {
        "models": models,
        "calibration": pd.DataFrame(calibration_rows),
        "validation": pd.DataFrame(validation_rows),
        "locked": locked_summary,
        "validated": {f"W{w}|{c[0]}|{c[1]}": v for (w, c), v in validated.items()},
        "raw_validated": raw_validated,
        "rows": rows, "bands": bands, "h1": h1,
    }


# ── summaries ───────────────────────────────────────────────────────────────

def _strategy_rows(rows: pd.DataFrame, strategies: dict, validated: dict) -> pd.DataFrame:
    frames = []
    for strategy, mapping in strategies.items():
        for metric, detector in mapping.items():
            part = rows[(rows["metric"] == metric) & (rows["detector"] == detector)]
            if part.empty:
                continue
            part = part.copy()
            part["strategy"] = strategy
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(result: dict, include_bootstrap: bool) -> dict:
    strategies = sr.strategy_detectors()
    models = result["models"]
    rows = result["rows"]
    if rows.empty:
        return {"note": "no validated detector cell; no ordered outcomes"}
    windowed = rows[rows["metric"] != sr.RAW_CHANNEL]
    by_strategy = _strategy_rows(windowed, strategies, result["validated"])
    scenarios = ["sudden", "gradual"] + (["sudden_boot", "gradual_boot"]
                                         if include_bootstrap else [])

    detection = []
    for keys, part in by_strategy.groupby(["model", "scenario", "strategy", "metric", "window"]):
        model, scenario, strategy, metric, window = keys
        base = sr.SCENARIOS[scenario.replace("_boot", "")]
        detected = part["category"].isin(["early_detection", "post_threshold"])
        delays = (part.loc[detected, "first_alarm"] - base.t_star).astype(float)
        detection.append({
            "model": model, "scenario": scenario, "strategy": strategy, "metric": metric,
            "window": int(window), "detector": part["detector"].iloc[0],
            "parameter": float(part["parameter"].iloc[0]), "realizations": len(part),
            "detection_rate": float(detected.mean()),
            "pre_change_alarm_rate": float((part["category"] == "false_alarm").mean()),
            "early_detection_rate": float((part["category"] == "early_detection").mean()),
            "post_threshold_rate": float((part["category"] == "post_threshold").mean()),
            "no_alarm_rate": float((part["category"] == "no_alarm").mean()),
            "median_signed_delay": float(delays.median()) if len(delays) else None,
            "delay_q25": float(delays.quantile(0.25)) if len(delays) else None,
            "delay_q75": float(delays.quantile(0.75)) if len(delays) else None,
        })
    detection = pd.DataFrame(detection)

    raw = rows[rows["metric"] == sr.RAW_CHANNEL]
    raw_summary = []
    for (model, scenario), part in raw.groupby(["model", "scenario"]):
        base = sr.SCENARIOS[scenario.replace("_boot", "")]
        detected = part["category"].isin(["early_detection", "post_threshold"])
        delays = (part.loc[detected, "first_alarm"] - base.t_star).astype(float)
        raw_summary.append({"model": model, "scenario": scenario,
                            "detection_rate": float(detected.mean()),
                            "pre_change_alarm_rate":
                                float((part["category"] == "false_alarm").mean()),
                            "median_signed_delay":
                                float(delays.median()) if len(delays) else None})

    # H2: paired strategy contrasts.
    h2 = []
    seed = sr.SEEDS["h2_bootstrap"]
    pairs = [(a, b) for i, a in enumerate(sr.STRATEGIES) for b in sr.STRATEGIES[i + 1:]]
    for scenario in ("sudden", "gradual"):
        base = sr.SCENARIOS[scenario]
        for w in sr.WINDOWS:
            for metric in sr.METRICS:
                for a, b in pairs:
                    for model in models:
                        sel = by_strategy[(by_strategy["model"] == model)
                                          & (by_strategy["scenario"] == scenario)
                                          & (by_strategy["window"] == w)
                                          & (by_strategy["metric"] == metric)]
                        pa = sel[sel["strategy"] == a].sort_values("realization")
                        pb = sel[sel["strategy"] == b].sort_values("realization")
                        if pa.empty or pb.empty:
                            h2.append({"scenario": scenario, "window": w, "metric": metric,
                                       "strategy_a": a, "strategy_b": b, "model": model,
                                       "status": "not_comparable_unvalidated"})
                            continue
                        da = pa["category"].isin(["early_detection", "post_threshold"]).to_numpy()
                        db = pb["category"].isin(["early_detection", "post_threshold"]).to_numpy()
                        diff, low, high = sr.paired_bootstrap_difference(
                            da.astype(float), db.astype(float), sr.N_H2_BOOTSTRAP, seed)
                        delay_a = pa.loc[da, "first_alarm"].astype(float) - base.t_star
                        delay_b = pb.loc[db, "first_alarm"].astype(float) - base.t_star
                        verdict = "a_better" if low > 0 else "b_better" if high < 0 else "no_difference"
                        h2.append({"scenario": scenario, "window": w, "metric": metric,
                                   "strategy_a": a, "strategy_b": b, "model": model,
                                   "status": "compared",
                                   "same_detector": pa["detector"].iloc[0] == pb["detector"].iloc[0],
                                   "detection_rate_a": float(da.mean()),
                                   "detection_rate_b": float(db.mean()),
                                   "difference_a_minus_b": diff, "ci_low": low, "ci_high": high,
                                   "median_delay_a": float(delay_a.median()) if len(delay_a) else None,
                                   "median_delay_b": float(delay_b.median()) if len(delay_b) else None,
                                   "verdict": verdict})
    h2 = pd.DataFrame(h2)
    h2_cross = []
    if not h2.empty:
        compared = h2[h2["status"] == "compared"]
        for keys, part in compared.groupby(["scenario", "window", "metric",
                                            "strategy_a", "strategy_b"]):
            verdicts = set(part["verdict"])
            cross = "none"
            if len(part) == len(models) and len(verdicts) == 1:
                only = verdicts.pop()
                cross = only if only != "no_difference" else "none"
            h2_cross.append(dict(zip(["scenario", "window", "metric", "strategy_a",
                                      "strategy_b"], keys, strict=True), cross_model_verdict=cross))

    # H3: leads, candidates, first-to-alarm shares, Kendall's W.
    lead_rows, share_rows, kendall_rows = [], [], []
    for scenario in scenarios:
        for strategy in sr.STRATEGIES:
            for w in sr.WINDOWS:
                for model in models:
                    sel = by_strategy[(by_strategy["model"] == model)
                                      & (by_strategy["scenario"] == scenario)
                                      & (by_strategy["strategy"] == strategy)
                                      & (by_strategy["window"] == w)]
                    present = [m for m in sr.METRICS if (sel["metric"] == m).any()]
                    if not present:
                        continue
                    pivot = sel.pivot(index="realization", columns="metric",
                                      values="first_post_ts").reindex(columns=present)
                    times = pivot.to_numpy(dtype=float)
                    times = np.where(np.isnan(times), np.inf, times)
                    detect = {m: float(detection[(detection["model"] == model)
                                                 & (detection["scenario"] == scenario)
                                                 & (detection["strategy"] == strategy)
                                                 & (detection["window"] == w)
                                                 & (detection["metric"] == m)
                                                 ]["detection_rate"].iloc[0])
                              for m in present}
                    for metric in present:
                        if metric == "CQ":
                            continue
                        if "CQ" in present:
                            lead = float((times[:, present.index(metric)]
                                          < times[:, present.index("CQ")]).mean())
                        else:
                            lead = None
                        lead_rows.append({"scenario": scenario, "strategy": strategy,
                                          "window": w, "model": model, "metric": metric,
                                          "lead_over_cq": lead,
                                          "detection_rate": detect[metric]})
                    shares = sr.first_to_alarm_share(times)
                    share_map = dict(zip(present, map(float, shares), strict=True))
                    base_name = scenario.replace("_boot", "")
                    share_rows.append({"scenario": scenario, "strategy": strategy,
                                       "window": w, "model": model,
                                       "first_to_alarm_share": share_map,
                                       "predicted": list(sr.PREDICTIONS[base_name]),
                                       "prediction_holds": sr.prediction_holds(
                                           share_map, sr.PREDICTIONS[base_name])})
                    kendall_rows.append({"scenario": scenario, "strategy": strategy,
                                         "window": w, "model": model, "metrics": present,
                                         "kendalls_w": sr.kendalls_w(sr.rank_rows(times))
                                         if len(present) >= 2 else None})
    leads = pd.DataFrame(lead_rows)
    candidates = []
    if not leads.empty:
        for keys, part in leads.groupby(["scenario", "strategy", "window", "metric"]):
            qualifies = (len(part) == len(models)
                         and part["lead_over_cq"].notna().all()
                         and (part["detection_rate"] >= 0.5).all()
                         and (part["lead_over_cq"] >= 0.5).all())
            candidates.append(dict(zip(["scenario", "strategy", "window", "metric"], keys, strict=True),
                                   replay_early_warning_candidate=bool(qualifies)))

    h1 = result["h1"]
    h1_summary = []
    if not h1.empty:
        for keys, part in h1.groupby(["model", "scenario", "window", "metric"]):
            h1_summary.append(dict(zip(["model", "scenario", "window", "metric"], keys, strict=True),
                                   median_pre_mean=float(part["pre_mean"].median()),
                                   median_post_mean=float(part["post_mean"].median()),
                                   median_cliffs_delta=float(part["cliffs_delta"].median()),
                                   median_mann_whitney_p=float(part["mann_whitney_p"].median()),
                                   fraction_p_below_05=float(
                                       (part["mann_whitney_p"] < 0.05).mean()),
                                   median_n_pre=float(part["n_pre"].median())))
    return {
        "detection": detection, "raw_cq_ddm": raw_summary, "h2": h2,
        "h2_cross_model": h2_cross, "leads": leads, "candidates": candidates,
        "first_to_alarm": share_rows, "kendalls_w": kendall_rows,
        "h1": pd.DataFrame(h1_summary),
    }


def _json_ready(value):
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records"))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--figures-only", action="store_true",
                        help="Redraw primary figures from saved outputs; no recomputation.")
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    print("Verifying frozen inputs...", flush=True)
    inputs = verify_inputs()
    digest = hashlib.sha256("".join(v["sha256"] for v in inputs.values()).encode()).hexdigest()
    metrics = pd.read_csv(FROZEN_INPUTS["metrics"][0])

    from reasoning_monitor.evaluation.stream_replay_figures import write_figures

    if args.figures_only:
        tables = sr.session_values(metrics, "session_mean")
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        result = {"models": sorted(tables),
                  "rows": pd.read_csv(output / "ordered_outcomes.csv"),
                  "bands": pd.read_csv(output / "trajectory_bands.csv")}
        names = write_figures(result, {"detection": pd.DataFrame(saved["detection"])},
                              tables, output)
        print(f"Redrew {len(names)} figure files.", flush=True)
        return

    empty = empty_primary_audit()

    verification = {"inputs": inputs, "protocol_sha256": sha256(PROTOCOL),
                    "empty_primary_records": empty, "modes": {}}
    for mode in MODES:
        tables = sr.session_values(metrics, mode)
        verification["modes"][mode] = {
            "sessions_per_model": {m: 2 * t.n_tasks for m, t in tables.items()},
            "missingness": sr.missingness(tables)}
        result = run_mode(mode, tables, args.workers, output / "cache", digest)
        summary = summarize(result, include_bootstrap=mode == "session_mean")
        target = output if mode == "session_mean" else output / "final_turn_sensitivity"
        target.mkdir(parents=True, exist_ok=True)
        with (target / "calibration.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_ready({
                "family_alpha": sr.FAMILY_ALPHA, "cell_alpha": sr.CELL_ALPHA,
                "raw_cell_alpha": sr.RAW_CELL_ALPHA, "seeds": sr.SEEDS,
                "strategies": sr.strategy_detectors(), "locked": result["locked"],
                "validated": result["validated"], "raw_validated": result["raw_validated"],
                "calibration_grid": result["calibration"],
                "validation": result["validation"]}), handle, indent=2)
        result["rows"].to_csv(target / "ordered_outcomes.csv", index=False)
        result["bands"].to_csv(target / "trajectory_bands.csv", index=False)
        with (target / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_ready({
                "protocol": "LIC-RQ-v2-SR", "mode": mode,
                "label": "post-hoc replay / method demonstration; not confirmatory",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                **summary}), handle, indent=2)
        if mode == "session_mean":
            figures = write_figures(result, summary, tables, target)
            verification["figures"] = figures
        print(f"[{mode}] written to {target}", flush=True)
    with (output / "input_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(verification), handle, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
