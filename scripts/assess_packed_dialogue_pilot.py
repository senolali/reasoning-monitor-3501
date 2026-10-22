"""Assess completed CDR-v1.1 pilot records without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_experiment import load_dialogue_checkpoint  # noqa: E402
from reasoning_monitor.evaluation.packed_dialogue_pilot import (  # noqa: E402
    assess_packed_dialogue_pilot,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_asu.yaml"
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_analysis_gate.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    records_by_model = {
        model: load_dialogue_checkpoint(output / f"records_CDR11_{model}.jsonl")
        for model in gate["expected_models"]
    }
    report = assess_packed_dialogue_pilot(records_by_model, gate)
    report_path = output / "controlled_dialogue_v1_1_pilot_assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

