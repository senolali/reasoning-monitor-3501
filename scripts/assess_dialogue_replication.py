"""Assess CDR-v1.4 saved records without making model calls."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_experiment import load_dialogue_checkpoint  # noqa: E402
from reasoning_monitor.evaluation.primary_dialogue_validation import (  # noqa: E402
    assess_primary_validation,
)

CONFIG = ROOT / "configs" / "controlled_dialogue_v1_4_replication_asu.yaml"
GATE = ROOT / "configs" / "controlled_dialogue_v1_4_replication_analysis_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    paths = {
        str(model["name"]): output / f"records_CDR14_{model['name']}.jsonl"
        for model in config["models"]
    }
    records = {
        model: load_dialogue_checkpoint(path) for model, path in paths.items()
    }
    report = assess_primary_validation(records, gate)
    report["analysis"] = "cdr_v1_4_fresh_calibration_replication"
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): _sha256(path) for path in paths.values()
    }
    report_path = output / "replication_assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
