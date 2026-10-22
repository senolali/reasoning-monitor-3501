"""Assess the CDR-v1.2a capacity retry without model calls."""

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
from reasoning_monitor.evaluation.user_only_capacity_retry import (  # noqa: E402
    assess_capacity_retry,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_asu.yaml"
GATE_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_analysis_gate.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    retry_path = ROOT / config["output_dir"] / "data" / "records_CDR12a_capacity_retry.jsonl"
    retries = load_dialogue_checkpoint(retry_path)
    retries_by_model = {
        model: [record for record in retries if record.get("model") == model]
        for model in gate["expected_models"]
    }
    original_by_model = {}
    existing_by_model = {}
    packed_config = yaml.safe_load(
        (
            ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_asu.yaml"
        ).read_text(encoding="utf-8")
    )
    for model in gate["expected_models"]:
        source = config["models"][model]
        source_path = ROOT / source["source_path"]
        if _sha256(source_path) != source["source_sha256"]:
            raise ValueError(f"original source hash differs for {model}")
        original_by_model[model] = load_dialogue_checkpoint(source_path)
        existing = packed_config["existing_packed_pilot_records"][model]
        existing_path = ROOT / existing["path"]
        if _sha256(existing_path) != existing["sha256"]:
            raise ValueError(f"packed-pilot source hash differs for {model}")
        existing_by_model[model] = load_dialogue_checkpoint(existing_path)

    report = assess_capacity_retry(
        retries_by_model,
        original_by_model,
        existing_by_model,
        gate,
    )
    report["retry_record_sha256"] = _sha256(retry_path)
    report_path = retry_path.with_name("capacity_retry_assessment.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
