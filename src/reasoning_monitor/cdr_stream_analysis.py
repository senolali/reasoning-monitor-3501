"""Call-free analysis of CDR-Stream main-run records (CDR_STREAM_ANALYSIS_PROTOCOL.md).

Stages: per-turn scoring -> H1 window contrasts -> null-calibrated monitoring
(H2/H3) -> descriptive H4/H5 -> figures. Detectors, grids, window signals and
strategy assignments are reused unchanged from the existing evaluation layer.
No model, gateway or network call is made; NLI/BERTScore scorers are injected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from reasoning_monitor.detectors.assignment import DETECTOR_STRATEGIES, assign_detector
from reasoning_monitor.evaluation import stream_replay as sr
from reasoning_monitor.evaluation.cusum_calibration import wilson_interval
from reasoning_monitor.evaluation.detector_development import (
    DEVELOPMENT_SPECS,
    DevelopmentSpec,
    build_window_signal,
    first_alarms_for_grid,
)
from reasoning_monitor.metrics.dimensions import build_metric
from reasoning_monitor.metrics.rqeval_adapters import _ESMetric
from reasoning_monitor.models.clients import reasoning_steps_from_text

METRICS: tuple[str, ...] = ("CQ", "CS", "RS", "LS", "ES", "SS")
STRATEGIES: tuple[str, ...] = tuple(DETECTOR_STRATEGIES)
WINDOWS: tuple[int, ...] = (10, 20, 30)
PRIMARY_WINDOW = 20
ES_WINDOW = 5
H1_HALF_WINDOW = 20
ALPHA_CELL = 0.05
DETECTION_SPAN = 33
N_SIGN_FLIP = 20_000
N_BOOTSTRAP = 2_000
SEEDS = {"sign_flip": 20261004, "bootstrap": 20261005}
GRIDS = {**sr.GRIDS, "ewma": DEVELOPMENT_SPECS["ES"],
         # opt-in corrected ADWIN; same delta grid as the historical detector
         "adwin_v": DevelopmentSpec("adwin_v", "delta", sr.GRIDS["adwin"].values)}

CHANGE_POINTS: dict[str, tuple[int, ...]] = {
    "S0-E": (), "S0-H": (), "S1": (34, 68), "S2": (51,), "S3": (34, 67)}
NULL_SCENARIOS = ("S0-E", "S0-H")
DEGRADATIONS: tuple[tuple[str, str, int], ...] = (
    ("S1-a", "S1", 34), ("S1-b", "S1", 68), ("S2", "S2", 51), ("S3-a", "S3", 34))
RECOVERIES: tuple[tuple[str, str, int], ...] = (("S3-b", "S3", 67),)
H3_PREDICTIONS = sr.PREDICTIONS          # gradual: CS, LS lead; sudden: CQ, RS lead
H3_KIND = {"S1": "gradual", "S2": "sudden"}
TWO_SIDED_METRICS = ("LS", "SS")


# ── scoring ─────────────────────────────────────────────────────────────────

def is_valid(record: Mapping) -> bool:
    return bool(record["ok"]) and not bool(record["truncated"])


def score_turns(records: Sequence[Mapping], coherence: Callable[[list[str]], float],
                similarity: Callable[[list[str], list[str]], list[float]], *,
                k: int = 3, n_paraphrases: int = 2,
                truncated_as_incorrect: bool = False) -> pd.DataFrame:
    """One row per (model, dialogue, turn): six dialogue metrics plus CQ_ctrl.

    ``truncated_as_incorrect`` is the prespecified CQ sensitivity: a truncated
    primary scores CQ=0 (and enters ES) instead of being missing.
    """
    cells: dict[tuple, list[Mapping]] = defaultdict(list)
    for record in records:
        if record["ok"]:
            cells[(record["model"], record["dialogue_id"], int(record["turn"]))].append(record)
    es_history: dict[tuple, tuple[list[float], list[int]]] = {}
    rows = []
    for key in sorted(cells):
        cell = cells[key]
        dialogue = [r for r in cell if r["arm"] == "dialogue"]
        runs = sorted((r for r in dialogue if r["role"] in ("primary", "repeat")),
                      key=lambda r: (r["role"] != "primary", r["call_index"]))
        paras = [r for r in dialogue if r["role"] == "paraphrase"]
        control = [r for r in cell if r["arm"] == "control"]
        meta = cell[0]
        row = {"model": key[0], "dialogue_id": key[1], "turn": key[2],
               "scenario": meta["scenario"], "template": meta["template"],
               "regime": meta["regime"], "level": meta["level"], "family": meta["family"],
               **{m: np.nan for m in METRICS}, "CQ_ctrl": np.nan,
               "primary_truncated": False, "primary_tokens": np.nan,
               "any_truncated": any(r["truncated"] for r in cell)}
        primary = runs[0] if runs and runs[0]["role"] == "primary" else None
        if control and (is_valid(control[0]) or truncated_as_incorrect):
            row["CQ_ctrl"] = float(bool(control[0]["correct"]) and is_valid(control[0]))
        if primary is not None:
            row["primary_truncated"] = bool(primary["truncated"])
            row["primary_tokens"] = int(primary["n_tokens"])
            history = es_history.setdefault(key[:2], ([], []))
            if is_valid(primary) or truncated_as_incorrect:
                cq = float(bool(primary["correct"]) and is_valid(primary))
                row["CQ"] = cq
                history[0].append(cq)
                history[1].append(int(primary["n_tokens"]))
                del history[0][:-ES_WINDOW], history[1][:-ES_WINDOW]
                if len(history[0]) == ES_WINDOW:
                    row["ES"] = float(_ESMetric().compute_per_instance(
                        list(history[0]), list(history[1]))[-1])
            if is_valid(primary):
                steps = reasoning_steps_from_text(primary["raw_text"])
                row["LS"] = coherence(steps) if len(steps) >= 2 else np.nan
                if row["CQ"] == 1.0 and len(paras) == n_paraphrases and all(
                        is_valid(r) for r in paras):
                    row["RS"] = sum(bool(r["correct"]) for r in paras) / n_paraphrases
            if len(runs) == k and all(is_valid(r) for r in runs):
                pairs = list(combinations(runs, 2))
                row["CS"] = sum(a["answer"] is not None and a["answer"] == b["answer"]
                                for a, b in pairs) / len(pairs)
                traces = [" ".join(reasoning_steps_from_text(r["raw_text"])) for r in runs]
                if all(traces):
                    trace_pairs = list(combinations(traces, 2))
                    row["SS"] = float(np.mean(similarity([a for a, _ in trace_pairs],
                                                         [b for _, b in trace_pairs])))
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame) -> dict:
    out = {}
    for model, part in frame.groupby("model"):
        out[model] = {m: round(float(part[m].notna().mean()), 4) for m in (*METRICS, "CQ_ctrl")}
        out[model]["primary_truncated"] = int(part["primary_truncated"].sum())
        out[model]["cells_with_truncation"] = int(part["any_truncated"].sum())
    return out


# ── H1: window contrasts ────────────────────────────────────────────────────

def window_delta(values: pd.Series, turns: pd.Series, change: int,
                 half: int = H1_HALF_WINDOW) -> float:
    pre = values[(turns >= change - half) & (turns < change)].dropna()
    post = values[(turns >= change) & (turns < change + half)].dropna()
    if pre.empty or post.empty:
        return float("nan")
    return float(post.mean() - pre.mean())


def dialogue_deltas(frame: pd.DataFrame, transitions=DEGRADATIONS + RECOVERIES
                    ) -> pd.DataFrame:
    rows = []
    for (model, dialogue), part in frame.groupby(["model", "dialogue_id"]):
        scenario = part["scenario"].iloc[0]
        for label, scen, change in transitions:
            if scen != scenario:
                continue
            row = {"model": model, "dialogue_id": dialogue, "transition": label}
            for metric in (*METRICS, "CQ_ctrl"):
                row[metric] = window_delta(part[metric], part["turn"], change)
            row["CQ_net"] = row["CQ"] - row["CQ_ctrl"]
            rows.append(row)
    return pd.DataFrame(rows)


def stratified_sign_flip(values: Sequence[float], strata: Sequence[str], *,
                         alternative: str = "less", n: int = N_SIGN_FLIP,
                         seed: int = SEEDS["sign_flip"]) -> dict:
    """Sign-flip test of the mean, flipping signs per dialogue within strata.

    Exact when 2**n_dialogues <= n, otherwise Monte Carlo with the observed
    statistic counted once (p >= 1/(n+1)).
    """
    x = np.asarray(values, float)
    keep = np.isfinite(x)
    x = x[keep]
    support = int(len(x))
    if support == 0:
        return {"mean": float("nan"), "p": float("nan"), "support": 0}
    observed = float(x.mean())
    if 2 ** support <= n:
        signs = np.array([[1 if (i >> j) & 1 else -1 for j in range(support)]
                          for i in range(2 ** support)])
        stats = (signs * x).mean(axis=1)
        extra = 0
    else:
        rng = np.random.default_rng(seed)
        stats = (rng.choice((-1.0, 1.0), size=(n, support)) * x).mean(axis=1)
        extra = 1
    if alternative == "less":
        hits = np.sum(stats <= observed + 1e-12)
    elif alternative == "greater":
        hits = np.sum(stats >= observed - 1e-12)
    else:
        hits = np.sum(np.abs(stats) >= abs(observed) - 1e-12)
    p = float((hits + extra) / (len(stats) + extra))
    return {"mean": observed, "p": p, "support": support,
            "per_stratum_support": pd.Series(np.asarray(strata)[keep]).value_counts().to_dict()}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    items = sorted(((p, k) for k, p in pvalues.items() if math.isfinite(p)))
    adjusted, running, m = {}, 0.0, len(items)
    for rank, (p, key) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return {k: adjusted.get(k, float("nan")) for k in pvalues}


def level_matched_contrast(part: pd.DataFrame, split: int = 50) -> float:
    early, late = part[part["turn"] <= split], part[part["turn"] > split]
    diffs, weights = [], []
    for level in set(early["level"]) & set(late["level"]):
        a = early.loc[early["level"] == level, "CQ"].dropna()
        b = late.loc[late["level"] == level, "CQ"].dropna()
        if len(a) and len(b):
            diffs.append(b.mean() - a.mean())
            weights.append(min(len(a), len(b)))
    return float(np.average(diffs, weights=weights)) if diffs else float("nan")


def analyze_h1(frame: pd.DataFrame) -> dict:
    deltas = dialogue_deltas(frame)
    h1a, h1b = {}, {}
    for label, _, _ in DEGRADATIONS:
        part = deltas[deltas["transition"] == label]
        for metric in METRICS:
            h1a[f"{label}|{metric}"] = stratified_sign_flip(part[metric], part["model"])
        h1b[label] = stratified_sign_flip(part["CQ_net"], part["model"])
    for family in (h1a, h1b):
        adjusted = holm({k: v["p"] for k, v in family.items()})
        for key, value in family.items():
            value["p_holm"] = adjusted[key]
    per_model = {}
    for (model, label), part in deltas.groupby(["model", "transition"]):
        alternative = "greater" if label in {r[0] for r in RECOVERIES} else "less"
        per_model[f"{model}|{label}"] = {
            m: stratified_sign_flip(part[m], part["model"], alternative=alternative)
            for m in (*METRICS, "CQ_net")}
    level = []
    for (model, dialogue), part in frame.groupby(["model", "dialogue_id"]):
        level.append({"model": model, "dialogue_id": dialogue,
                      "scenario": part["scenario"].iloc[0],
                      "delta": level_matched_contrast(part)})
    level = pd.DataFrame(level)
    return {"h1a": h1a, "h1b": h1b, "per_model_descriptive": per_model,
            "level_matched_context": stratified_sign_flip(level["delta"], level["model"]),
            "level_matched_by_scenario": {
                s: stratified_sign_flip(p["delta"], p["model"])
                for s, p in level.groupby("scenario")},
            "deltas": deltas}


# ── monitoring ──────────────────────────────────────────────────────────────

def strategy_detectors() -> dict[str, dict[str, str]]:
    return {s: {m: assign_detector(build_metric(m, window_size=10), strategy=s).name
                for m in METRICS} for s in STRATEGIES}


def dialogue_signal(part: pd.DataFrame, metric: str, window: int, *,
                    increase: bool = False) -> list[tuple[int, float]]:
    part = part.sort_values("turn")
    turns = part["turn"].to_numpy()
    if metric == "ES":
        valid = part["CQ"].notna().to_numpy()
        tokens = part["primary_tokens"].to_numpy()
        indices = [i for i in range(len(part)) if valid[i]]
        raw = build_window_signal("ES", [], part["CQ"].to_numpy(), tokens, indices, window)
        # build_window_signal numbers items 1..n over valid turns; map back to turns.
        signal = [(int(turns[indices[t - 1]]), v) for t, v in raw]
    else:
        values = part[metric].to_numpy(float)
        raw = build_window_signal(metric, values, [], [], list(range(len(part))), window)
        signal = [(int(turns[t - 1]), v) for t, v in raw]
    if increase:
        signal = [(t, 1.0 - v) for t, v in signal]
    return signal


def first_alarm_grid(signal, detector: str) -> dict[float, int | None]:
    return first_alarms_for_grid(signal, GRIDS[detector])


def calibrate(null_grids: Sequence[dict[float, int | None]], detector: str,
              alpha: float = ALPHA_CELL) -> float | None:
    for value in GRIDS[detector].values:
        rate = sum(g[value] is not None for g in null_grids) / max(len(null_grids), 1)
        if rate <= alpha:
            return value
    return None


def categorize(first: int | None, change: int) -> str:
    if first is None:
        return "miss"
    if first < change:
        return "false_alarm"
    if first <= change + DETECTION_SPAN:
        return "detection"
    return "late"


def run_monitoring(frame: pd.DataFrame, *, windows=WINDOWS,
                   available: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Null-calibrated monitoring for every strategy, window and metric.

    ``available`` (model -> acquired metrics) restricts every metric to the
    models that acquired it; other dialogues are excluded, never counted quiet.
    """
    detectors = strategy_detectors()
    all_dialogues = {key: part for key, part in frame.groupby(["model", "dialogue_id"])}
    scenario = {key: part["scenario"].iloc[0] for key, part in all_dialogues.items()}

    def acquired(metric):
        return {k: p for k, p in all_dialogues.items()
                if available is None or metric in available.get(k[0], ())}
    arms = [(m, False) for m in METRICS] + [(m, True) for m in TWO_SIDED_METRICS]
    grids: dict[tuple, dict] = {}
    for window in windows:
        for metric, increase in arms:
            for detector in sorted({d[metric] for d in detectors.values()}):
                for key, part in acquired(metric).items():
                    signal = dialogue_signal(part, metric, window, increase=increase)
                    grids[(window, metric, increase, detector, key)] = first_alarm_grid(
                        signal, detector)
    cells, outcomes = {}, []
    for window in windows:
        for metric, increase in arms:
            for detector in sorted({d[metric] for d in detectors.values()}):
                dialogues = acquired(metric)
                nulls = [k for k in dialogues if scenario[k] in NULL_SCENARIOS]
                null_grids = [grids[(window, metric, increase, detector, k)] for k in nulls]
                chosen = calibrate(null_grids, detector)
                loo_alarms = 0
                for i in range(len(nulls)):
                    others = null_grids[:i] + null_grids[i + 1:]
                    value = calibrate(others, detector)
                    if value is not None and null_grids[i][value] is not None:
                        loo_alarms += 1
                low, high = wilson_interval(loo_alarms, len(nulls)) if nulls else (0, 0)
                cell = f"W{window}|{metric}{'+' if increase else ''}|{detector}"
                cells[cell] = {"value": chosen, "null_dialogues": len(nulls),
                               "loo_far": loo_alarms / max(len(nulls), 1),
                               "loo_far_wilson": [low, high]}
                if chosen is None:
                    continue
                for key in dialogues:
                    changes = CHANGE_POINTS[scenario[key]]
                    first = grids[(window, metric, increase, detector, key)][chosen]
                    outcomes.append({
                        "window": window, "metric": metric, "increase": increase,
                        "detector": detector, "model": key[0], "dialogue_id": key[1],
                        "scenario": scenario[key], "first_alarm": first,
                        "category": categorize(first, changes[0]) if changes else (
                            "null_alarm" if first is not None else "null_quiet"),
                        "delay": (first - changes[0]) if changes and first is not None
                        else None})
    outcomes = pd.DataFrame(outcomes)
    return {"cells": cells, "outcomes": outcomes, "detectors": detectors,
            "null_dialogues": {m: [list(k) for k in acquired(m)
                                   if scenario[k] in NULL_SCENARIOS] for m in METRICS}}


def summarize_strategies(monitoring: Mapping, window: int = PRIMARY_WINDOW) -> dict:
    out = monitoring["outcomes"]
    summary = {}
    for strategy, mapping in monitoring["detectors"].items():
        per_metric = {}
        for metric, detector in mapping.items():
            rows = out[(out["window"] == window) & (out["metric"] == metric)
                       & (~out["increase"]) & (out["detector"] == detector)]
            drift = rows[~rows["scenario"].isin(NULL_SCENARIOS)]
            cell = monitoring["cells"].get(f"W{window}|{metric}|{detector}", {})
            per_metric[metric] = {
                "detector": detector, "parameter": cell.get("value"),
                "loo_far": cell.get("loo_far"),
                "detection_rate": float((drift["category"] == "detection").mean())
                if len(drift) else None,
                "false_alarm_rate_drift": float((drift["category"] == "false_alarm").mean())
                if len(drift) else None,
                "median_delay": float(drift.loc[drift["category"] == "detection",
                                                "delay"].median())
                if (drift["category"] == "detection").any() else None,
                "n_drift_dialogues": int(len(drift))}
        summary[strategy] = per_metric
    return summary


def analyze_h3(monitoring: Mapping, strategy: str = "proposal_mapping",
               window: int = PRIMARY_WINDOW) -> dict:
    out = monitoring["outcomes"]
    mapping = monitoring["detectors"][strategy]
    result = {}
    for scenario, kind in H3_KIND.items():
        change = CHANGE_POINTS[scenario][0]
        table = {}
        for metric, detector in mapping.items():
            rows = out[(out["window"] == window) & (out["metric"] == metric)
                       & (~out["increase"]) & (out["detector"] == detector)
                       & (out["scenario"] == scenario)]
            for _, r in rows.iterrows():
                first = r["first_alarm"]
                valid = first is not None and not pd.isna(first) and first >= change
                table.setdefault((r["model"], r["dialogue_id"]), {})[metric] = (
                    float(first) if valid else math.inf)
        if not table:
            result[scenario] = {"n_dialogues": 0}
            continue
        keys = sorted(table)
        metrics = [m for m in METRICS if all(m in table[k] for k in keys)]
        times = np.array([[table[k][m] for m in metrics] for k in keys])
        shares = dict(zip(metrics, sr.first_to_alarm_share(times), strict=True))
        cq = metrics.index("CQ") if "CQ" in metrics else None
        before_cq = {m: float(np.mean(times[:, i] < times[:, cq])) if cq is not None else None
                     for i, m in enumerate(metrics) if m != "CQ"}
        result[scenario] = {
            "kind": kind, "n_dialogues": len(keys),
            "cq_alarm_dialogues": int(np.isfinite(times[:, cq]).sum()) if cq is not None else 0,
            "first_to_alarm_share": shares, "alarms_before_cq_share": before_cq,
            "kendalls_w": sr.kendalls_w(sr.rank_rows(times)),
            "predicted_leaders": list(H3_PREDICTIONS[kind]),
            "prediction_holds": sr.prediction_holds(shares, H3_PREDICTIONS[kind])}
    return result


# ── H4 / H5 ─────────────────────────────────────────────────────────────────

def analyze_descriptive(deltas: pd.DataFrame, seed: int = SEEDS["bootstrap"]) -> dict:
    degr = deltas[deltas["transition"].isin([d[0] for d in DEGRADATIONS])]
    rng = np.random.default_rng(seed)
    h4 = {}
    for model, part in degr.groupby("model"):
        h4[model] = {}
        for metric in METRICS:
            x = part[metric].dropna().to_numpy()
            if not len(x):
                h4[model][metric] = None
                continue
            boots = x[rng.integers(0, len(x), size=(N_BOOTSTRAP, len(x)))].mean(axis=1)
            h4[model][metric] = {"mean": float(x.mean()), "ci95": [
                float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                "support": int(len(x))}
    both = degr[["CQ", "ES"]].dropna()
    h5 = {"support": int(len(both)),
          "spearman": float(both["CQ"].rank().corr(both["ES"].rank())) if len(both) > 2
          else None,
          "share_cq_and_es_decline": float(((both["CQ"] < 0) & (both["ES"] < 0)).mean())
          if len(both) else None}
    return {"h4": h4, "h5": h5}


def restrict_to_available(frame: pd.DataFrame, available: Mapping[str, Sequence[str]]
                          ) -> pd.DataFrame:
    """Blank model-metric pairs that were not acquired (Stage A, protocol §12)."""
    out = frame.copy()
    for metric in (*METRICS, "CQ_ctrl"):
        allowed = [m for m, metrics in available.items() if metric in metrics]
        out.loc[~out["model"].isin(allowed), metric] = np.nan
    return out


# ── control-arm monitoring (CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md) ──────

CONTROL_METRICS: tuple[str, ...] = ("CQ", "LS", "ES")


def score_control_turns(records: Sequence[Mapping], coherence: Callable[[list[str]], float]
                        ) -> pd.DataFrame:
    """Per-turn CQ/LS/ES of the control (state_summary) arm, in the dialogue-frame layout."""
    control = sorted((r for r in records if r["ok"] and r["arm"] == "control"),
                     key=lambda r: (r["model"], r["dialogue_id"], int(r["turn"])))
    history: dict[tuple, tuple[list[float], list[int]]] = {}
    rows = []
    for r in control:
        row = {"model": r["model"], "dialogue_id": r["dialogue_id"], "turn": int(r["turn"]),
               "scenario": r["scenario"], "template": r["template"], "regime": r["regime"],
               "level": r["level"], "family": r["family"],
               **{m: np.nan for m in METRICS}, "CQ_ctrl": np.nan,
               "primary_truncated": bool(r["truncated"]), "primary_tokens": int(r["n_tokens"]),
               "any_truncated": bool(r["truncated"])}
        if is_valid(r):
            cq = float(bool(r["correct"]))
            row["CQ"] = cq
            steps = reasoning_steps_from_text(r["raw_text"])
            row["LS"] = coherence(steps) if len(steps) >= 2 else np.nan
            cqs, tokens = history.setdefault((r["model"], r["dialogue_id"]), ([], []))
            cqs.append(cq)
            tokens.append(int(r["n_tokens"]))
            del cqs[:-ES_WINDOW], tokens[:-ES_WINDOW]
            if len(cqs) == ES_WINDOW:
                row["ES"] = float(_ESMetric().compute_per_instance(list(cqs), list(tokens))[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def h2_paired_bootstrap(monitoring: Mapping, window: int = PRIMARY_WINDOW,
                        metrics: Sequence[str] = CONTROL_METRICS, n: int = N_BOOTSTRAP,
                        seed: int = 20261006) -> dict:
    """Per metric: detection(strategy) - detection(adwin_all) over drift dialogues."""
    out = monitoring["outcomes"]
    rng = np.random.default_rng(seed)
    result = {}
    for metric in metrics:
        per_strategy = {}
        for strategy, mapping in monitoring["detectors"].items():
            rows = out[(out["window"] == window) & (out["metric"] == metric)
                       & (~out["increase"]) & (out["detector"] == mapping[metric])
                       & (~out["scenario"].isin(NULL_SCENARIOS))]
            per_strategy[strategy] = rows.set_index(["model", "dialogue_id"])["category"].eq(
                "detection").astype(float)
        base = per_strategy.get("adwin_all")
        result[metric] = {}
        for strategy, series in per_strategy.items():
            if strategy == "adwin_all" or base is None or series.empty:
                continue
            joined = pd.concat([series, base], axis=1, join="inner").to_numpy()
            if not len(joined):
                continue
            diff = joined[:, 0] - joined[:, 1]
            boots = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
            result[metric][f"{strategy}-adwin_all"] = {
                "mean": float(diff.mean()), "ci95": [float(np.quantile(boots, 0.025)),
                                                     float(np.quantile(boots, 0.975))],
                "support": int(len(diff))}
    return result


def control_screen(monitoring: Mapping, window: int = PRIMARY_WINDOW) -> dict:
    """Protocol section 5 screening criterion on CQ."""
    summary = summarize_strategies(monitoring, window=window)
    checks = {}
    for strategy in ("distribution_based", "proposal_mapping"):
        cq = summary[strategy]["CQ"]
        checks[strategy] = {
            **cq,
            "pass": bool(cq["detection_rate"] is not None and cq["detection_rate"] >= 0.5
                         and cq["loo_far"] is not None and cq["loo_far"] <= 1 / 12 + 1e-9
                         and cq["median_delay"] is not None and cq["median_delay"] <= 25)}
    return {"window": window, "checks": checks,
            "promising": any(c["pass"] for c in checks.values())}
