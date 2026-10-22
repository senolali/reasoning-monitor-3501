"""Call-free preflight for the one-cell CDR-v1.1a capacity retry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_retry_asu.yaml"
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_retry_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_probe_path"]
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    target = config["target"]
    matches = [
        row
        for row in rows
        if row.get("model") == target["model"]
        and row.get("dialogue_id") == target["dialogue_id"]
        and row.get("condition") == target["condition"]
        and int(row.get("turn", 0)) == int(target["turn"])
    ]
    original = matches[0] if len(matches) == 1 else {}
    probe_spec = config["production_probe"]
    probe_path = ROOT / probe_spec["path"]
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    checks = {
        "protocol": config.get("protocol_id") == gate.get("protocol_id") == "CDR-v1.1a",
        "source_hash": _sha256(source_path) == config["source_probe_sha256"],
        "single_target": len(matches) == 1,
        "original_ceiling_hit": int(original.get("n_tokens", 0))
        == int(target["original_max_tokens"]),
        "original_failed_marker": not bool(original.get("strict_final_marker")),
        "original_failed_accuracy": not bool(original.get("correct")),
        "same_messages": len(original.get("messages") or []) == 2,
        "one_call": int(config["expected_calls"]) == 1,
        "larger_ceiling": int(config["retry_max_tokens"])
        > int(target["original_max_tokens"]),
        "model_options": config["model"].get("request_options") == expected_options
        and config["model"].get("temperature") == 0.0
        and int(config["model"].get("max_tokens"))
        == int(config["retry_max_tokens"]),
        "production_probe": _sha256(probe_path) == probe_spec["sha256"]
        and probe.get("model") == probe_spec["provider_model"]
        and bool(probe.get("results"))
        and all(result.get("accepted") for result in probe["results"]),
    }
    source_checks = {
        relative: _sha256(ROOT / relative) == expected
        for relative, expected in gate["expected_sha256"].items()
    }
    checks["frozen_sources"] = all(source_checks.values())
    checks = {name: bool(passed) for name, passed in checks.items()}
    report = {
        "analysis": "cdr_v1_1a_transport_retry_preflight",
        "no_model_calls": True,
        "checks": checks,
        "frozen_source_checks": source_checks,
        "expected_calls": 1,
        "maximum_output_tokens": int(config["retry_max_tokens"]),
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

