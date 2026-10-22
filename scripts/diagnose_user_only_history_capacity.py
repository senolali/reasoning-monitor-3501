"""Run the frozen-output CDR-v1.2 capacity diagnostic without model calls."""

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
from reasoning_monitor.evaluation.user_only_capacity import (  # noqa: E402
    diagnose_user_only_capacity,
)

CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_capacity_diagnostic.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_records(spec: dict[str, str]) -> list[dict]:
    path = ROOT / spec["path"]
    digest = _sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"source hash differs: {spec['path']}")
    return load_dialogue_checkpoint(path)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assessment_spec = config["source_assessment"]
    assessment_path = ROOT / assessment_spec["path"]
    if _sha256(assessment_path) != assessment_spec["sha256"]:
        raise ValueError("frozen CDR-v1.2 assessment hash differs")
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if assessment.get("pass") is not False:
        raise ValueError("capacity diagnostic requires the preserved failed gate")

    user_records = {
        model: _verified_records(config["user_only_records"][model])
        for model in config["expected_models"]
    }
    existing_records = {
        model: _verified_records(config["existing_packed_pilot_records"][model])
        for model in config["expected_models"]
    }
    report = diagnose_user_only_capacity(user_records, existing_records, config)
    report["source_assessment_sha256"] = assessment_spec["sha256"]
    report["source_record_sha256"] = {
        spec["path"]: spec["sha256"]
        for group in ("user_only_records", "existing_packed_pilot_records")
        for spec in config[group].values()
    }
    output_path = ROOT / config["output_path"]
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
