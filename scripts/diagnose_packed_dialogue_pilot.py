"""Run the frozen call-free CDR-v1.1 pilot diagnostic."""

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
from reasoning_monitor.evaluation.packed_dialogue_diagnostic import (  # noqa: E402
    diagnose_packed_dialogue_pilot,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_diagnostic.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    records_by_model = {}
    source_sha256 = {}
    for model, source in config["sources"].items():
        path = ROOT / source["path"]
        digest = _sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"frozen packed-pilot source hash differs for {model}")
        source_sha256[source["path"]] = digest
        records_by_model[model] = load_dialogue_checkpoint(path)
    assessment_path = ROOT / config["assessment"]["path"]
    if _sha256(assessment_path) != config["assessment"]["sha256"]:
        raise ValueError("frozen packed-pilot assessment hash differs")
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if assessment.get("pass") is not False:
        raise ValueError("diagnostic requires the preserved failed pilot gate")

    report = diagnose_packed_dialogue_pilot(records_by_model)
    report["source_sha256"] = source_sha256
    report["assessment_sha256"] = config["assessment"]["sha256"]
    report["decision_scope"] = config["decision_scope"]
    output = assessment_path.with_name("controlled_dialogue_v1_1_diagnostic.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(report["decision"] != "freeze_format_matched_successor_design")


if __name__ == "__main__":
    raise SystemExit(main())

