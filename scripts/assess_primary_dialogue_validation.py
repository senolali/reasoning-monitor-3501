"""Assess the cluster-aware CDR-v1.3 validation without model calls."""

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

CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_3_primary_validation_asu.yaml"
)
GATE_PATH = (
    ROOT
    / "configs"
    / "controlled_dialogue_v1_3_primary_validation_analysis_gate.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    records_by_model = {}
    source_sha256 = {}
    for model in gate["expected_models"]:
        path = output / f"records_CDR13_{model}.jsonl"
        records_by_model[model] = load_dialogue_checkpoint(path)
        source_sha256[str(path.relative_to(ROOT))] = _sha256(path)
    report = assess_primary_validation(records_by_model, gate)
    report["source_sha256"] = source_sha256
    report_path = output / "primary_validation_assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

