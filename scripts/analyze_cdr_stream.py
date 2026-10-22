"""Call-free CDR-Stream main-run analysis (CDR_STREAM_ANALYSIS_PROTOCOL.md).

  default       completeness and hash check only (no metric, accuracy or alarm shown)
  --synthetic   run the whole pipeline on synthetic records (demonstration/plumbing)
  --analyze     real analysis: needs all 54,000 records AND protocol_frozen: true;
                loads the real local NLI and BERTScore scorers (no fallback)
  --stage-a     protocol §12 Stage A on the pinned checkpoint prefix (budget pause)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reasoning_monitor import cdr_stream_analysis as ca  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/cdr_stream_v1_analysis.yaml"
COLORS = {"CQ": "#0072B2", "CS": "#E69F00", "RS": "#009E73", "LS": "#CC79A7",
          "ES": "#56B4E9", "SS": "#D55E00", "CQ_ctrl": "#555555"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.DataFrame):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2) + "\n", encoding="utf-8")


def plot_trajectories(frame: pd.DataFrame, output: Path, label: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    scenarios = list(ca.CHANGE_POINTS)
    panels = (*ca.METRICS, "CQ_ctrl")
    for model, part in frame.groupby("model"):
        fig, axes = plt.subplots(len(panels), len(scenarios), figsize=(4 * len(scenarios),
                                 2.0 * len(panels)), sharex=True, squeeze=False,
                                 constrained_layout=True)
        for c, scenario in enumerate(scenarios):
            sub = part[part["scenario"] == scenario]
            n = sub["dialogue_id"].nunique()
            for r, metric in enumerate(panels):
                ax = axes[r, c]
                if sub.empty:
                    ax.set_axis_off()
                    continue
                by_turn = sub.groupby("turn")[metric]
                mean, cover = by_turn.mean(), by_turn.apply(lambda s: s.notna().mean())
                ax.plot(mean.index, mean.values, color=COLORS[metric], lw=1.2)
                ax.fill_between(cover.index, 0, cover.values, color="#BBBBBB", alpha=0.25,
                                step="mid", label="coverage")
                for change in ca.CHANGE_POINTS[scenario]:
                    ax.axvline(change, color="#333333", ls="--", lw=0.8)
                ax.set_ylim(-0.02, 1.02)
                if c == 0:
                    ax.set_ylabel(metric)
                if r == 0:
                    ax.set_title(f"{scenario} (n={n} dialogues)")
                if r == len(panels) - 1:
                    ax.set_xlabel("turn")
        fig.suptitle(f"{label}: {model} - mean per turn; grey = share of dialogues "
                     "with a measured value; dashed = change points")
        for ext in ("png", "pdf"):
            path = output / f"trajectories_{model}.{ext}"
            fig.savefig(path, dpi=130)
            files.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    return files


def plot_alarms(monitoring, output: Path, label: str, window: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = monitoring["outcomes"]
    mapping = monitoring["detectors"]["proposal_mapping"]
    rows = out[(out["window"] == window) & (~out["increase"])]
    rows = pd.concat([rows[(rows["metric"] == m) & (rows["detector"] == d)]
                      for m, d in mapping.items()])
    keys = sorted({(r.model, r.scenario, r.dialogue_id) for r in rows.itertuples()})
    fig, ax = plt.subplots(figsize=(10, max(4, 0.12 * len(keys))), constrained_layout=True)
    index = {k: i for i, k in enumerate(keys)}
    for metric in ca.METRICS:
        sub = rows[(rows["metric"] == metric) & rows["first_alarm"].notna()]
        ax.scatter(sub["first_alarm"], [index[(r.model, r.scenario, r.dialogue_id)]
                                        for r in sub.itertuples()],
                   s=14, color=COLORS[metric], label=metric)
    for i, (_, scenario, _) in enumerate(keys):
        for change in ca.CHANGE_POINTS[scenario]:
            ax.plot([change, change], [i - 0.4, i + 0.4], color="#333333", lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{m} {s}" for m, s, _ in keys], fontsize=5)
    ax.set_xlim(0, 101)
    ax.set_xlabel("first alarm turn (black ticks = change points)")
    ax.legend(ncol=6, fontsize=7, loc="upper left")
    ax.set_title(f"{label}: first alarms, proposal_mapping, W={window}")
    files = []
    for ext in ("png", "pdf"):
        path = output / f"alarms_W{window}.{ext}"
        fig.savefig(path, dpi=130)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def memoized(function):
    cache = {}

    def wrapper(*args):
        key = tuple(tuple(a) for a in args)
        if key not in cache:
            cache[key] = function(*args)
        return cache[key]
    return wrapper


def run_pipeline(frame: pd.DataFrame, sensitivity: pd.DataFrame | None, output: Path,
                 label: str, windows=ca.WINDOWS, available=None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "turn_metrics.csv", index=False)
    h1 = ca.analyze_h1(frame)
    h1["deltas"].to_csv(output / "h1_dialogue_deltas.csv", index=False)
    write_json(output / "h1.json", {k: v for k, v in h1.items() if k != "deltas"})
    monitoring = ca.run_monitoring(frame, windows=windows, available=available)
    monitoring["outcomes"].to_csv(output / "monitoring_outcomes.csv", index=False)
    primary_window = ca.PRIMARY_WINDOW if ca.PRIMARY_WINDOW in windows else windows[0]
    write_json(output / "monitoring.json", {
        "cells": monitoring["cells"], "detectors": monitoring["detectors"],
        "null_dialogues": monitoring["null_dialogues"],
        "strategies": {f"W{w}": ca.summarize_strategies(monitoring, window=w)
                       for w in windows}})
    write_json(output / "h3.json", ca.analyze_h3(monitoring, window=primary_window))
    write_json(output / "descriptive.json", ca.analyze_descriptive(h1["deltas"]))
    report = {"coverage": ca.coverage(frame)}
    if sensitivity is not None:
        sens = ca.analyze_h1(sensitivity)
        report["truncated_as_incorrect_h1a_cq"] = {
            k: v for k, v in sens["h1a"].items() if k.endswith("|CQ")}
        report["truncated_as_incorrect_h1b"] = sens["h1b"]
    figures = plot_trajectories(frame, output, label)
    figures += plot_alarms(monitoring, output, label, primary_window)
    report["figures"] = figures
    return report


def stage_a(config: dict, records_path: Path) -> int:
    if not config["protocol_frozen"]:
        raise SystemExit("The analysis protocol is not frozen.")
    if sha256_file(ROOT / "CDR_STREAM_ANALYSIS_PROTOCOL.md") != config["protocol_sha256"]:
        raise SystemExit("Protocol file differs from the frozen hash.")
    with records_path.open("rb") as handle:
        prefix = b"".join(handle.readline() for _ in range(int(config["stage_a_lines"])))
    if hashlib.sha256(prefix).hexdigest() != config["stage_a_prefix_sha256"]:
        raise SystemExit("Stage A checkpoint prefix differs from the pinned hash.")
    for key in ("manifest", "mappings"):
        if sha256_file(ROOT / config[f"{key}_path"]) != config[f"{key}_sha256"]:
            raise SystemExit(f"Frozen {key} hash mismatch")
    records = [json.loads(line) for line in prefix.decode("utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["ok"] and r["fingerprint"] == config["fingerprint"]]
    available = {m: tuple(v) for m, v in config["stage_a_available"].items()}
    from reasoning_monitor.dialogue_six_metrics import StrictScorers

    print(f"Stage A: {len(records)} successful calls. Loading real NLI and BERTScore...",
          flush=True)
    scorers = StrictScorers(with_ss=True)
    coherence, similarity = memoized(scorers.coherence), memoized(scorers.similarity)
    frame = ca.restrict_to_available(ca.score_turns(records, coherence, similarity), available)
    print("Scored main frame; scoring truncation sensitivity...", flush=True)
    sensitivity = ca.restrict_to_available(ca.score_turns(
        records, coherence, similarity, truncated_as_incorrect=True), available)
    output = ROOT / config["stage_a_output_dir"]
    report = run_pipeline(frame, sensitivity, output, "CDR-Stream main run, STAGE A",
                          available=available)
    report.update({"stage": "A", "available": available,
                   "records_prefix_lines": config["stage_a_lines"],
                   "records_prefix_sha256": config["stage_a_prefix_sha256"],
                   "successful_calls": len(records),
                   "manifest_sha256": config["manifest_sha256"],
                   "mappings_sha256": config["mappings_sha256"],
                   "protocol_sha256": config["protocol_sha256"],
                   "nli_model": "cross-encoder/nli-deberta-v3-small",
                   "bertscore_model": "distilbert-base-uncased",
                   "note": "Stage A of protocol section 12; Stage B pending"})
    write_json(output / "manifest.json", report)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "coverage": report["coverage"]},
                     indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--stage-a", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    if args.synthetic:
        from test_cdr_stream_analysis import coherence, similarity, synthetic_records

        output = ROOT / config["synthetic_output_dir"]
        records = synthetic_records(seed=1, per_scenario=6, models=("model_a", "model_b",
                                                                     "model_c"))
        frame = ca.score_turns(records, coherence, similarity)
        report = run_pipeline(frame, None, output, "SYNTHETIC (not data)")
        report["note"] = "synthetic records only; plumbing demonstration, not evidence"
        write_json(output / "manifest.json", report)
        print(json.dumps({"output": str(output.relative_to(ROOT)),
                          "figures": report["figures"]}, indent=2))
        return 0

    records_path = ROOT / config["records_path"]
    if args.stage_a:
        return stage_a(config, records_path)
    lines = records_path.read_text(encoding="utf-8").splitlines() if records_path.exists() else []
    ok_keys = set()
    fingerprints = set()
    for line in lines:
        if line.strip():
            row = json.loads(line)
            fingerprints.add(row["fingerprint"])
            if row["ok"]:
                ok_keys.add((row["model"], row["dialogue_id"], row["turn"], row["arm"],
                             row["role"], row["call_index"]))
    status = {"records_path": config["records_path"], "successful_calls": len(ok_keys),
              "planned_calls": config["planned_calls"],
              "fingerprint_ok": fingerprints <= {config["fingerprint"]},
              "complete": len(ok_keys) == config["planned_calls"],
              "protocol_frozen": bool(config["protocol_frozen"])}
    print(json.dumps(status, indent=2))
    if not args.analyze:
        print("Check only: no metric, accuracy or alarm was computed.")
        return 0
    if not (status["complete"] and status["fingerprint_ok"]):
        raise SystemExit("Acquisition is not complete; resume the main run first.")
    if not status["protocol_frozen"]:
        raise SystemExit("The analysis protocol is not frozen; it needs investigator approval.")
    for key in ("manifest", "mappings"):
        if sha256_file(ROOT / config[f"{key}_path"]) != config[f"{key}_sha256"]:
            raise SystemExit(f"Frozen {key} hash mismatch")
    from reasoning_monitor.dialogue_six_metrics import StrictScorers

    print("Loading real local NLI and BERTScore scorers...", flush=True)
    scorers = StrictScorers(with_ss=True)
    records = [json.loads(line) for line in lines if line.strip()]
    records = [r for r in records if r["ok"]]
    frame = ca.score_turns(records, scorers.coherence, scorers.similarity)
    sensitivity = ca.score_turns(records, scorers.coherence, scorers.similarity,
                                 truncated_as_incorrect=True)
    output = ROOT / config["output_dir"]
    report = run_pipeline(frame, sensitivity, output, "CDR-Stream main run")
    report.update({"records_sha256": sha256_file(records_path),
                   "manifest_sha256": config["manifest_sha256"],
                   "mappings_sha256": config["mappings_sha256"],
                   "protocol": "CDR_STREAM_ANALYSIS_PROTOCOL.md",
                   "protocol_sha256": sha256_file(ROOT / "CDR_STREAM_ANALYSIS_PROTOCOL.md"),
                   "nli_model": "cross-encoder/nli-deberta-v3-small",
                   "bertscore_model": "distilbert-base-uncased"})
    write_json(output / "manifest.json", report)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "coverage": report["coverage"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
