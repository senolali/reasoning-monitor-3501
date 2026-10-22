"""Call-free preflight for the fresh CDR-v1.1 packed-history pilot."""

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
from reasoning_monitor.packed_dialogue_experiment import packed_pilot_messages  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_asu.yaml"
ANALYSIS_GATE_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_1_pilot_analysis_gate.yaml"
)
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    analysis_gate = yaml.safe_load(ANALYSIS_GATE_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}

    evidence = {
        name: json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for name, relative in config["transport_evidence"].items()
    }
    probe = evidence["probe_assessment"]
    retry_result = evidence["retry_result"]
    retry_assessment = evidence["retry_assessment"]
    checks = {
        "protocol": config.get("protocol_id")
        == analysis_gate.get("protocol_id")
        == gate.get("protocol_id")
        == "CDR-v1.1",
        "bank_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "bank_validation": not validate_dialogue_bank(bank),
        "selected_dialogues": all(
            dialogue_id in by_id for dialogue_id in config["selected_dialogue_ids"]
        ),
        "calibration_only": all(
            by_id[dialogue_id]["split"] == "calibration"
            for dialogue_id in config["selected_dialogue_ids"]
        ),
        "counts": int(config["expected_total_calls"]) == 216
        and int(config["expected_calls_per_model"]) == 108,
        "token_ceiling": int(config["maximum_output_token_ceiling"])
        == int(config["expected_total_calls"])
        * int(config["max_output_tokens_per_call"]),
        "resume": bool(config["resume"]),
        "stage_one_only": int(config["k_repeats"]) == 1
        and int(config["l_paraphrases"]) == 0,
        "transport_mode": config.get("history_transport")
        == "packed_system_context",
        "original_probe_preserved": probe.get("pass") is False
        and probe["models"]["gpt4o_mini"]["pass"] is True
        and probe["models"]["llama4_scout"]["pass"] is False,
        "capacity_retry_pass": retry_assessment.get("pass") is True
        and retry_result.get("correct") is True
        and retry_result.get("strict_final_marker") is True
        and int(retry_result.get("n_tokens", 0))
        < int(retry_result.get("max_tokens", 0)),
    }

    first_dialogue = by_id[config["selected_dialogue_ids"][0]]
    first_turn, second_turn = first_dialogue["turns"][:2]
    history = [
        {"role": "user", "content": first_turn["user_prompt"]},
        {"role": "assistant", "content": first_turn["oracle_response"]},
    ]
    packed = packed_pilot_messages(
        turn=second_turn,
        condition="oracle_history",
        history=history,
        system_prompt=config["system_prompt"],
        user_prompt=second_turn["user_prompt"],
    )
    summary = packed_pilot_messages(
        turn=second_turn,
        condition="state_summary",
        history=[],
        system_prompt=config["system_prompt"],
        user_prompt=second_turn["user_prompt"],
    )
    checks["two_message_transport"] = len(packed) == len(summary) == 2
    checks["packed_history_present"] = (
        "CALLER-MANAGED PRIOR DIALOGUE TRANSCRIPT" in packed[0]["content"]
        and first_turn["user_prompt"] in packed[0]["content"]
    )
    checks["summary_isolation"] = (
        "VERIFIED STATE BEFORE THIS TURN" in summary[0]["content"]
        and first_turn["user_prompt"] not in summary[0]["content"]
    )

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
            and int(model.get("max_tokens")) == 512
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
        "analysis": "cdr_v1_1_packed_pilot_preflight",
        "no_model_calls": True,
        "checks": checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": frozen_checks,
        "expected_calls": int(config["expected_total_calls"]),
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

