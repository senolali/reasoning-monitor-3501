"""Assess completed RAT-v1 calibration records without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (  # noqa: E402
    build_risk_set,
)
from reasoning_monitor.evaluation.risk_adjusted_trajectory_calibration import (  # noqa: E402
    assess_calibration_model,
)

CONFIG_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_calibration_asu.yaml"
GATE_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_calibration_analysis_gate.yaml"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    seed = int((config.get("seeds") or [config["seed"]])[0])
    window = int(config["window_sizes"][0])
    results = {}
    combined = []
    problems: list[str] = []

    for model in gate["expected_models"]:
        tag = f"RAT_{model}_seed{seed}_W{window}"
        records_path = output / f"records_{tag}.jsonl"
        per_query_path = output / f"per_query_{tag}.csv"
        if not records_path.exists() or not per_query_path.exists():
            problems.append(f"missing completed outputs for {model}")
            continue
        records = _read_jsonl(records_path)
        per_query = pd.read_csv(per_query_path)
        result = assess_calibration_model(model, records, per_query, gate)
        results[model] = result
        model_frame = per_query.copy()
        model_frame["model"] = model
        combined.append(model_frame)

    passing_models = sorted(model for model, result in results.items() if result["pass"])
    if len(passing_models) < int(gate["minimum_passing_models"]):
        problems.append("fewer than two preregistered models passed calibration")

    risk_rows = 0
    if combined:
        try:
            risk = build_risk_set(pd.concat(combined, ignore_index=True))
            risk_rows = len(risk)
            risk.to_csv(output / "rat_v1_calibration_risk_set.csv", index=False)
        except ValueError as exc:
            problems.append(str(exc))

    report = {
        "analysis": "rat_v1_calibration_gate",
        "no_model_calls": True,
        "models": results,
        "passing_models": passing_models,
        "risk_set_rows": int(risk_rows),
        "problems": problems,
        "pass": not problems,
        "gate": gate,
    }
    report_path = output / "rat_v1_calibration_assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
