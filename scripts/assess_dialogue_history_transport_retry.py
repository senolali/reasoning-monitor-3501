"""Assess the one-cell CDR-v1.1a capacity retry without model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_retry_asu.yaml"
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_retry_gate.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    result_path = ROOT / config["output_path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "model": result.get("model") == gate["expected_model"],
        "dialogue": result.get("dialogue_id") == gate["expected_dialogue_id"],
        "condition": result.get("condition") == gate["expected_condition"],
        "turn": int(result.get("turn", 0)) == int(gate["expected_turn"]),
        "gold": str(result.get("gold_answer")) == str(gate["expected_gold_answer"]),
        "success": float(bool(result.get("ok")))
        >= float(gate["minimum_success_rate"]),
        "accuracy": float(bool(result.get("correct")))
        >= float(gate["minimum_accuracy"]),
        "strict_final_marker": float(bool(result.get("strict_final_marker")))
        >= float(gate["minimum_strict_final_marker_rate"]),
        "no_truncation": int(
            str(result.get("finish_reason") or "").lower() == "length"
        )
        <= int(gate["maximum_truncations"]),
        "no_token_ceiling_hit": int(result.get("n_tokens", 0))
        < int(result.get("max_tokens", 0)),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    report = {
        "analysis": "cdr_v1_1a_transport_capacity_retry_gate",
        "no_model_calls": True,
        "checks": checks,
        "n_tokens": int(result.get("n_tokens", 0)),
        "max_tokens": int(result.get("max_tokens", 0)),
        "pass": all(checks.values()),
        "pass_authorizes": gate["pass_authorizes"],
        "gate": gate,
    }
    report_path = result_path.with_name("transport_retry_assessment.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

