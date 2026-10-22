"""Diagnose the failed CDR-v1.3 gate using saved records only."""

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
from reasoning_monitor.evaluation.primary_validation_sensitivity import (  # noqa: E402
    diagnose_primary_validation_sensitivity,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_3_sensitivity.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assessment_spec = config["source_assessment"]
    assessment_path = ROOT / assessment_spec["path"]
    if _sha256(assessment_path) != assessment_spec["sha256"]:
        raise ValueError("frozen CDR-v1.3 assessment hash differs")
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if assessment.get("pass") is not False:
        raise ValueError("sensitivity analysis requires the preserved failed gate")

    records_by_model = {}
    for model, spec in config["source_records"].items():
        path = ROOT / spec["path"]
        if _sha256(path) != spec["sha256"]:
            raise ValueError(f"source hash differs: {spec['path']}")
        records_by_model[model] = load_dialogue_checkpoint(path)

    gate = yaml.safe_load(
        (ROOT / config["analysis_gate_path"]).read_text(encoding="utf-8")
    )
    report = diagnose_primary_validation_sensitivity(records_by_model, gate)
    report["source_assessment_sha256"] = assessment_spec["sha256"]
    report["source_record_sha256"] = {
        spec["path"]: spec["sha256"] for spec in config["source_records"].values()
    }
    output_path = ROOT / config["output_path"]
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

