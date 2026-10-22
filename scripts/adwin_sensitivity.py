"""Call-free ADWIN-defect sensitivity re-analysis (ADWIN_DEFECT_AUDIT.md section 4).

Re-runs ONLY the cells whose detector was the defective historical ADWIN (CS and
SS), from saved records, with two substitutes: the corrected ``adwin_v`` and
CUSUM. Every other choice is the frozen one. Outputs go to
results/adwin_sensitivity/ and never replace frozen results.

  section39   S1 K/L detector development (permutation calibration/validation)
  section49   E2H GPT alarm timing (Section 47 trace), CS and SS
  section51   E2H GPT corrected-SS alarm timing (Section 51 per-query values)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.detectors.detectors import build_detector  # noqa: E402
from reasoning_monitor.evaluation import detector_development as dd  # noqa: E402
from reasoning_monitor.evaluation import stream_replay as sr  # noqa: E402
from reasoning_monitor.evaluation.permutation_controls import (  # noqa: E402
    generate_matched_permutations,
)

OUT = ROOT / "results/adwin_sensitivity"
SUBSTITUTES = {
    "adwin_v": dd.DevelopmentSpec("adwin_v", "delta", sr.GRIDS["adwin"].values),
    "cusum": sr.GRIDS["cusum"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Section 39 ──────────────────────────────────────────────────────────────

def section39() -> dict:
    src = ROOT / "results/proposal_s1_kl_validation_gpt_llama"
    frozen = json.loads((src / "detector_development/manifest.json").read_text(encoding="utf-8"))
    data = src / "data"
    qids = pd.read_csv(data / "stream_S1_seed42.csv").sort_values("t")["qid"].astype(str).tolist()
    frames = {}
    for model, expected in frozen["source_per_query_sha256"].items():
        path = data / f"per_query_S1_{model}_seed42_W10.csv"
        if sha(path) != expected:
            raise SystemExit(f"Section 39 source changed: {path}")
        frame = pd.read_csv(path).sort_values("t").reset_index(drop=True)
        if frame["qid"].astype(str).tolist() != qids:
            raise SystemExit("Section 39 order mismatch")
        frames[model] = frame
    detection = pd.read_csv(data / "detection_results.csv")
    s1 = detection[detection["scenario"] == "S1"]
    start = int(s1["transition_start"].dropna().astype(int).unique()[0])
    midpoint = int(s1["effective_t_star"].dropna().astype(int).unique()[0])
    calib = generate_matched_permutations(len(qids), frozen["n_calibration"],
                                          frozen["calibration_seed"])
    valid = generate_matched_permutations(len(qids), frozen["n_validation"],
                                          frozen["validation_seed"])
    original = dict(dd.DEVELOPMENT_SPECS)
    result = {}
    try:
        for name, spec in SUBSTITUTES.items():
            dd.DEVELOPMENT_SPECS.clear()
            dd.DEVELOPMENT_SPECS.update({"CS": spec, "SS": spec})
            out = OUT / "section39" / name
            out.mkdir(parents=True, exist_ok=True)
            calibration = dd._null_grid(frames, calib, 10)
            calibration.to_csv(out / "calibration_grid.csv", index=False)
            selected = dd.select_parameters(calibration, list(frames))
            validation, _ = dd._validate_selected(frames, valid, selected, 10, start, midpoint)
            validation.to_csv(out / "validation_summary.csv", index=False)
            result[name] = {
                "selected": selected,
                "cells": validation.to_dict(orient="records"),
                "candidate_pass": {m: bool(selected[m] is not None
                                           and validation[validation["metric"] == m][
                                               "cell_pass"].all()
                                           and len(validation[validation["metric"] == m]) == 2)
                                   for m in ("CS", "SS")}}
    finally:
        dd.DEVELOPMENT_SPECS.clear()
        dd.DEVELOPMENT_SPECS.update(original)
    result["rule"] = ("frozen Section 39: calibration Wilson upper <= 0.005 in both models; "
                      "held-out specificity; ordered first alarm in [15, 44] in both models")
    return result


# ── Sections 49 / 51 ────────────────────────────────────────────────────────

TRANSITIONS = (("easy_to_medium", 30, 60), ("medium_to_hard", 60, 90))


def replay(values: pd.Series, detector: str, metric: str, window: int = 10) -> list[int]:
    """Monitor semantics: W-mean once the window is full, default parameters, reset."""
    det = build_detector(detector)
    buffer: deque[float] = deque(maxlen=window)
    alarms = []
    for t, value in values.items():
        buffer.append(float(value))
        if len(buffer) < window:
            continue
        alarm = det.update(sum(buffer) / window, int(t), metric)
        if alarm is not None:
            alarms.append(int(alarm.t))
    return alarms


def timing(alarms: list[int], cq_alarms: list[int]) -> dict:
    rows, leads_all, detected_all = [], True, True
    for name, cp, end in TRANSITIONS:
        first = next((t for t in alarms if cp < t <= end), None)
        cq = next((t for t in cq_alarms if cp < t <= end), None)
        lead = None if first is None or cq is None else cq - first
        rows.append({"transition": name, "alarm_t": first, "cq_alarm_t": cq, "lead": lead})
        detected_all &= first is not None
        leads_all &= lead is not None and lead >= 1
    baseline_clean = not any(t <= 30 for t in alarms)
    return {"transitions": rows, "baseline_alarms_t_le_30": [t for t in alarms if t <= 30],
            "all_alarms": alarms,
            "early_warning_gate_pass": bool(baseline_clean and detected_all and leads_all)}


def section49_51() -> dict:
    data = ROOT / "results/e2h_final_holdout_gpt4o_mini/data"
    trace = pd.read_csv(data / "trace_ST2_gpt4o_mini_seed48_W10.csv").set_index("t")
    per_query = pd.read_csv(data / "per_query_ST2_gpt4o_mini_seed48_W10.csv").set_index("t")
    alarms = pd.read_csv(data / "alarms_ST2_gpt4o_mini_seed48_W10.csv")
    cq = sorted(alarms[(alarms["metric"] == "CQ") & (alarms["level"] == "drift")]["t"].astype(int))
    s51 = pd.read_csv(data / "section51_trace_reparse_per_query.csv").set_index("t")
    result = {"cq_drift_alarms": cq, "hashes": {
        "trace": sha(data / "trace_ST2_gpt4o_mini_seed48_W10.csv"),
        "per_query": sha(data / "per_query_ST2_gpt4o_mini_seed48_W10.csv"),
        "section51_per_query": sha(data / "section51_trace_reparse_per_query.csv")}}
    # Check the replay reproduces the trace's window means from per-query values.
    for metric in ("CS", "SS"):
        rebuilt = per_query[f"{metric}_raw"].rolling(10).mean().dropna()
        diff = (rebuilt - trace[metric].reindex(rebuilt.index)).abs().max()
        result[f"{metric}_trace_reproduced_max_abs_diff"] = float(diff)
    for name in ("adwin_v", "cusum"):
        result[f"section49|{name}"] = {
            m: timing(replay(per_query[f"{m}_raw"], name, m), cq) for m in ("CS", "SS")}
        result[f"section51|{name}"] = {
            "SS_corrected": timing(replay(s51["SS_corrected"], name, "SS"), cq)}
    result["historical_adwin_check"] = {
        m: replay(per_query[f"{m}_raw"], "adwin", m) for m in ("CS", "SS")}
    result["rule"] = ("frozen Section 49: alarms after the change and up to the interval end, "
                      "no alarm at t <= 30, detected in both transitions, leads CQ by >= 1 "
                      "query in both; default detector parameters as in Section 47")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="+", choices=("section39", "section49_51"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        result = section39() if part == "section39" else section49_51()
        result["sensitivity_only"] = True
        (OUT / f"{part}.json").write_text(json.dumps(result, indent=2, default=str) + "\n",
                                          encoding="utf-8")
        print(json.dumps(result, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
