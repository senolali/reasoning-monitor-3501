"""Assess the CDR-v1.2 user-only mechanism arm without model calls."""

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
from reasoning_monitor.evaluation.user_only_mechanism import (  # noqa: E402
    assess_user_only_mechanism,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_asu.yaml"
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_user_only_analysis_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    user_records = {}
    existing_records = {}
    source_sha256 = {}
    for model in gate["expected_models"]:
        user_records[model] = load_dialogue_checkpoint(
            output / f"records_CDR12_user_only_{model}.jsonl"
        )
        source = config["existing_packed_pilot_records"][model]
        source_path = ROOT / source["path"]
        digest = _sha256(source_path)
        if digest != source["sha256"]:
            raise ValueError(f"frozen packed-pilot record hash differs for {model}")
        source_sha256[source["path"]] = digest
        existing_records[model] = load_dialogue_checkpoint(source_path)
    report = assess_user_only_mechanism(user_records, existing_records, gate)
    report["source_sha256"] = source_sha256
    report_path = output / "user_only_history_assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

