"""Call-free preflight for the 12-call CDR-v1.1 transport probe."""

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
from reasoning_monitor.dialogue_transport import build_packed_turn_messages  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_probe_asu.yaml"
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_transport_probe_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    checks = {
        "protocol": config.get("protocol_id") == gate.get("protocol_id") == "CDR-v1.1",
        "bank_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "call_count": int(config["expected_total_calls"]) == 12,
        "conditions": config["conditions"] == ["self_history", "oracle_history"],
        "transport": config.get("history_transport") == "packed_system_context",
        "selected_dialogues": all(
            dialogue_id in by_id for dialogue_id in config["selected_dialogue_ids"]
        ),
    }

    source_checks = {}
    packed_checks = {}
    for model, source_spec in config["source_v1_records"].items():
        source_path = ROOT / source_spec["path"]
        source_checks[model] = _sha256(source_path) == source_spec["sha256"]
        records = load_dialogue_checkpoint(source_path)
        first_by_dialogue = {
            record["dialogue_id"]: record
            for record in records
            if record.get("condition") == "self_history"
            and record.get("role") == "primary"
            and int(record.get("turn", 0)) == 1
        }
        model_checks = []
        for dialogue_id in config["selected_dialogue_ids"]:
            dialogue = by_id[dialogue_id]
            first_turn = dialogue["turns"][0]
            second_turn = dialogue["turns"][1]
            messages = build_packed_turn_messages(
                current_user_prompt=second_turn["user_prompt"],
                history=[
                    {"role": "user", "content": first_turn["user_prompt"]},
                    {
                        "role": "assistant",
                        "content": first_by_dialogue[dialogue_id]["raw_text"],
                    },
                ],
                system_prompt=config["system_prompt"],
            )
            model_checks.append(
                len(messages) == 2
                and messages[0]["role"] == "system"
                and "[PRIOR USER TURN 1]" in messages[0]["content"]
                and first_turn["user_prompt"] in messages[0]["content"]
                and messages[1]["content"] == second_turn["user_prompt"]
            )
        packed_checks[model] = all(model_checks)
    checks["source_records"] = all(source_checks.values())
    checks["packed_context"] = all(packed_checks.values())

    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    model_checks = {}
    probe_checks = {}
    for model in config["models"]:
        name = model["name"]
        model_checks[name] = bool(
            model.get("request_options") == expected_options
            and model.get("temperature") == 0.0
            and int(model.get("max_tokens")) == 256
        )
        probe_spec = config["production_probes"][name]
        probe_path = ROOT / probe_spec["path"]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_checks[name] = bool(
            _sha256(probe_path) == probe_spec["sha256"]
            and probe.get("model") == probe_spec["provider_model"] == model["model"]
            and probe.get("results")
            and all(result.get("accepted") for result in probe["results"])
        )
    checks["model_options"] = all(model_checks.values())
    checks["production_probes"] = all(probe_checks.values())

    source_hash_checks = {}
    for relative, expected in gate["expected_sha256"].items():
        source_hash_checks[relative] = _sha256(ROOT / relative) == expected
    checks["frozen_sources"] = all(source_hash_checks.values())
    checks = {name: bool(passed) for name, passed in checks.items()}
    report = {
        "analysis": "cdr_v1_1_transport_probe_preflight",
        "no_model_calls": True,
        "checks": checks,
        "source_record_checks": source_checks,
        "packed_context_checks": packed_checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": source_hash_checks,
        "expected_calls": int(config["expected_total_calls"]),
        "maximum_output_tokens": int(config["expected_total_calls"])
        * max(int(model["max_tokens"]) for model in config["models"]),
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

