"""Call-free preflight for the CDR-v1.2 user-only history mechanism arm."""

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

from reasoning_monitor.data.dialogue_streams import validate_dialogue_bank  # noqa: E402
from reasoning_monitor.dialogue_experiment import load_dialogue_checkpoint  # noqa: E402
from reasoning_monitor.user_only_dialogue_experiment import (  # noqa: E402
    build_user_only_messages,
)

CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_asu.yaml"
)
ANALYSIS_GATE_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_user_only_analysis_gate.yaml"
)
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    analysis_gate = yaml.safe_load(ANALYSIS_GATE_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    selected_ids = list(config["selected_dialogue_ids"])

    expected_calls = (
        len(selected_ids)
        * int(config["turns_per_dialogue"])
        * len(config["models"])
    )
    checks = {
        "protocol": config.get("protocol_id")
        == analysis_gate.get("protocol_id")
        == gate.get("protocol_id")
        == "CDR-v1.2",
        "stage": config.get("stage") == "user_only_history_mechanism_probe"
        and analysis_gate.get("stage") == "user_only_history_mechanism_analysis"
        and gate.get("stage") == "user_only_history_mechanism_probe_preflight",
        "bank_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "bank_validation": not validate_dialogue_bank(bank),
        "selected_dialogues": len(selected_ids) == 3
        and len(set(selected_ids)) == 3
        and all(dialogue_id in by_id for dialogue_id in selected_ids),
        "calibration_only": all(
            by_id[dialogue_id]["split"] == "calibration"
            for dialogue_id in selected_ids
        ),
        "turn_counts": all(
            len(by_id[dialogue_id]["turns"])
            == int(config["turns_per_dialogue"])
            for dialogue_id in selected_ids
        ),
        "single_new_condition": config.get("condition") == "user_only_history",
        "transport_mode": config.get("history_transport")
        == "packed_user_only_context",
        "counts": expected_calls == int(config["expected_total_calls"]) == 72
        and int(config["expected_calls_per_model"]) == 36,
        "token_ceiling": int(config["maximum_output_token_ceiling"])
        == expected_calls * int(config["max_output_tokens_per_call"]),
        "resume": bool(config["resume"]),
        "paid_calls_locked": gate.get("paid_calls_authorized") is False,
    }

    transport_checks = []
    for dialogue_id in selected_ids:
        dialogue = by_id[dialogue_id]
        prior_user_turns: list[str] = []
        prior_oracle_responses: list[str] = []
        for turn in dialogue["turns"]:
            messages = build_user_only_messages(
                current_user_prompt=str(turn["user_prompt"]),
                prior_user_turns=prior_user_turns,
                system_prompt=str(config["system_prompt"]),
            )
            packed_system = messages[0]["content"]
            transport_checks.append(
                len(messages) == 2
                and [message["role"] for message in messages] == ["system", "user"]
                and messages[1]["content"] == str(turn["user_prompt"])
                and all(prompt in packed_system for prompt in prior_user_turns)
                and all(response not in packed_system for response in prior_oracle_responses)
                and (
                    int(turn["turn"]) == 1
                    or "CALLER-MANAGED PRIOR USER HISTORY" in packed_system
                )
                and "[PRIOR ASSISTANT" not in packed_system
            )
            prior_user_turns.append(str(turn["user_prompt"]))
            prior_oracle_responses.append(str(turn["oracle_response"]))
    checks["user_only_transport"] = all(transport_checks)

    existing_checks = {}
    for model, source in config["existing_packed_pilot_records"].items():
        source_path = ROOT / source["path"]
        records = load_dialogue_checkpoint(source_path)
        conditions = {record.get("condition") for record in records}
        existing_checks[model] = bool(
            _sha256(source_path) == source["sha256"]
            and len(records) == 108
            and {"self_history", "oracle_history", "state_summary"} <= conditions
            and all(record.get("dialogue_id") in selected_ids for record in records)
        )
    checks["existing_records_frozen"] = all(existing_checks.values())

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
            and int(model.get("max_tokens"))
            == int(config["max_output_tokens_per_call"])
        )
        probe_spec = config["production_probes"][name]
        probe_path = ROOT / probe_spec["path"]
        saved_probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_checks[name] = bool(
            _sha256(probe_path) == probe_spec["sha256"]
            and saved_probe.get("model")
            == probe_spec["provider_model"]
            == model["model"]
            and saved_probe.get("results")
            and all(result.get("accepted") for result in saved_probe["results"])
        )
    checks["model_options"] = all(model_checks.values())
    checks["production_probes"] = all(probe_checks.values())

    frozen_checks = {
        relative: _sha256(ROOT / relative) == expected
        for relative, expected in gate["expected_sha256"].items()
    }
    checks["frozen_sources"] = all(frozen_checks.values())
    checks = {name: bool(passed) for name, passed in checks.items()}
    report = {
        "analysis": "cdr_v1_2_user_only_history_probe_preflight",
        "no_model_calls": True,
        "checks": checks,
        "existing_record_checks": existing_checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": frozen_checks,
        "expected_calls": expected_calls,
        "reused_existing_calls": 144,
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
