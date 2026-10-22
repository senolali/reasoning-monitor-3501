"""Control-arm H2 monitoring (CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md), call-free.

  --screen-gpt   step 1: exploratory GPT-4o-mini screen on the pinned Stage A prefix
  --validate     step 2: Llama-3 70B and Gemma 3 12B (needs their control calls acquired)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_cdr_stream import (  # noqa: E402
    memoized,
    plot_alarms,
    plot_trajectories,
    sha256_file,
    write_json,
)

from reasoning_monitor import cdr_stream_analysis as ca  # noqa: E402

PROTOCOL = ROOT / "CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md"
PROTOCOL_SHA256 = "f7783a2c26a434d2742ac0e9937074b655627d25ab4f39d3070df149d0a0a667"
CONFIG = ROOT / "configs/cdr_stream_v1_analysis.yaml"
OUTPUT = ROOT / "results/cdr_stream_v1_control_monitoring"


def analyze_model(frame, model: str, output: Path, label: str) -> dict:
    part = frame[frame["model"] == model]
    available = {model: ca.CONTROL_METRICS}
    monitoring = ca.run_monitoring(part, available=available)
    output.mkdir(parents=True, exist_ok=True)
    part.to_csv(output / "control_turn_metrics.csv", index=False)
    monitoring["outcomes"].to_csv(output / "monitoring_outcomes.csv", index=False)
    result = {
        "model": model, "dialogues": int(part["dialogue_id"].nunique()),
        "coverage": {m: round(float(part[m].notna().mean()), 4) for m in ca.CONTROL_METRICS},
        "truncated": int(part["primary_truncated"].sum()),
        "screen": ca.control_screen(monitoring),
        "strategies": {f"W{w}": {s: {m: v for m, v in per.items() if m in ca.CONTROL_METRICS}
                                 for s, per in ca.summarize_strategies(monitoring, w).items()}
                       for w in ca.WINDOWS},
        "h2_paired_bootstrap_W20": ca.h2_paired_bootstrap(monitoring),
        "order_descriptive_W20": ca.analyze_h3(monitoring),
        "cells": {k: v for k, v in monitoring["cells"].items()
                  if k.split("|")[1].rstrip("+") in ca.CONTROL_METRICS and "+" not in k},
        "null_dialogues": len(monitoring["null_dialogues"]["CQ"])}
    result["figures"] = plot_trajectories(part, output, label) + plot_alarms(
        monitoring, output, label, ca.PRIMARY_WINDOW)
    write_json(output / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--screen-gpt", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise SystemExit("Control monitoring protocol differs from its frozen hash.")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records_path = ROOT / config["records_path"]
    if args.screen_gpt:
        with records_path.open("rb") as handle:
            prefix = b"".join(handle.readline() for _ in range(int(config["stage_a_lines"])))
        if hashlib.sha256(prefix).hexdigest() != config["stage_a_prefix_sha256"]:
            raise SystemExit("Stage A prefix differs from the pinned hash.")
        lines = prefix.decode("utf-8").splitlines()
        models, label, sub = ["gpt4o_mini"], "CONTROL ARM, GPT exploratory screen", "gpt_screen"
    else:
        lines = records_path.read_text(encoding="utf-8").splitlines()
        models, label, sub = ["llama3_70b", "gemma3_12b_it"], "CONTROL ARM, validation", \
            "validation"
    records = [json.loads(line) for line in lines if line.strip()]
    records = [r for r in records if r["ok"] and r["fingerprint"] == config["fingerprint"]
               and r["arm"] == "control" and r["model"] in models]
    if args.validate:
        for model in models:
            n = sum(r["model"] == model for r in records)
            if n < 3000:
                raise SystemExit(f"{model}: only {n}/3000 control calls acquired.")
    from reasoning_monitor.dialogue_six_metrics import StrictScorers

    print(f"{len(records)} control calls; loading real NLI...", flush=True)
    scorers = StrictScorers(with_ss=False)
    frame = ca.score_control_turns(records, memoized(scorers.coherence))
    results = {m: analyze_model(frame, m, OUTPUT / sub / m, label) for m in models}
    summary = {"protocol_sha256": PROTOCOL_SHA256, "records_source": config["records_path"],
               "results": {m: {"screen": r["screen"], "coverage": r["coverage"],
                               "null_dialogues": r["null_dialogues"]}
                           for m, r in results.items()}}
    write_json(OUTPUT / sub / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
