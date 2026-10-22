"""Call-free preflight for the CDR-v1.3 primary dialogue validation."""

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
from reasoning_monitor.primary_dialogue_validation import (  # noqa: E402
    build_primary_validation_messages,
)

CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_3_primary_validation_asu.yaml"
)
ANALYSIS_GATE_PATH = (
    ROOT
    / "configs"
    / "controlled_dialogue_v1_3_primary_validation_analysis_gate.yaml"
)
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_3_primary_validation_gate.yaml"


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
    calibration_ids = {
        dialogue["dialogue_id"]
        for dialogue in bank["dialogues"]
        if dialogue["split"] == "calibration"
    }
    feasibility_ids = {
        "cdr-v1-budget-001",
        "cdr-v1-incident-001",
        "cdr-v1-cohort-001",
    }
    expected_remaining = calibration_ids - feasibility_ids
    expected_calls = (
        len(selected_ids)
        * int(config["turns_per_dialogue"])
        * len(config["conditions"])
        * len(config["models"])
    )
    checks = {
        "protocol": config.get("protocol_id")
        == analysis_gate.get("protocol_id")
        == gate.get("protocol_id")
        == "CDR-v1.3",
        "stage": config.get("stage") == "nine_dialogue_primary_validation"
        and analysis_gate.get("stage")
        == "nine_dialogue_primary_validation_analysis"
        and gate.get("stage") == "nine_dialogue_primary_validation_preflight",
        "bank_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "bank_validation": not validate_dialogue_bank(bank),
        "all_remaining_calibration_dialogues": set(selected_ids)
        == expected_remaining
        and len(selected_ids) == 9,
        "no_feasibility_reuse": not (set(selected_ids) & feasibility_ids),
        "turn_counts": all(
            len(by_id[dialogue_id]["turns"])
            == int(config["turns_per_dialogue"])
            for dialogue_id in selected_ids
        ),
        "two_primary_conditions": config.get("conditions")
        == ["user_only_history", "state_summary"],
        "stage_one_only": int(config["k_repeats"]) == 1
        and int(config["l_paraphrases"]) == 0,
        "counts": expected_calls == int(config["expected_total_calls"]) == 432
        and int(config["expected_calls_per_model"]) == 216,
        "token_ceiling": int(config["maximum_output_token_ceiling"])
        == expected_calls * int(config["max_output_tokens_per_call"]),
        "resume": bool(config["resume"]),
        "cluster_primary_analysis": int(
            analysis_gate["expected_post_baseline_pairs_per_model"]
        )
        == 9 * 11
        and int(analysis_gate["minimum_positive_dialogue_gaps"]) == 7
        and float(analysis_gate["maximum_one_sided_sign_flip_p"]) == 0.05,
        "longitudinal_windows": (
            int(analysis_gate["early_turn_start"]),
            int(analysis_gate["early_turn_end"]),
            int(analysis_gate["late_turn_start"]),
            int(analysis_gate["late_turn_end"]),
        )
        == (2, 6, 7, 12),
        "paid_calls_locked": gate.get("paid_calls_authorized") is False,
    }

    transport_checks = []
    for dialogue_id in selected_ids:
        dialogue = by_id[dialogue_id]
        prior_user_turns: list[str] = []
        prior_oracle_responses: list[str] = []
        for turn in dialogue["turns"]:
            user_messages = build_primary_validation_messages(
                turn=turn,
                condition="user_only_history",
                prior_user_turns=prior_user_turns,
                system_prompt=str(config["system_prompt"]),
            )
            summary_messages = build_primary_validation_messages(
                turn=turn,
                condition="state_summary",
                prior_user_turns=prior_user_turns,
                system_prompt=str(config["system_prompt"]),
            )
            user_system = user_messages[0]["content"]
            summary_system = summary_messages[0]["content"]
            transport_checks.append(
                len(user_messages) == len(summary_messages) == 2
                and all(prompt in user_system for prompt in prior_user_turns)
                and all(
                    response not in user_system for response in prior_oracle_responses
                )
                and (
                    int(turn["turn"]) == 1
                    or "CALLER-MANAGED PRIOR USER HISTORY" in user_system
                )
                and "[PRIOR ASSISTANT" not in user_system
                and "VERIFIED STATE BEFORE THIS TURN" in summary_system
                and all(prompt not in summary_system for prompt in prior_user_turns)
                and user_messages[1]["content"] == summary_messages[1]["content"]
                == str(turn["user_prompt"])
            )
            prior_user_turns.append(str(turn["user_prompt"]))
            prior_oracle_responses.append(str(turn["oracle_response"]))
    checks["transport_isolation"] = all(transport_checks)

    evidence = config["mechanism_evidence"]
    assessment_path = ROOT / evidence["assessment_path"]
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    retry_path = ROOT / evidence["retry_records_path"]
    checks["mechanism_evidence"] = bool(
        _sha256(assessment_path) == evidence["assessment_sha256"]
        and _sha256(retry_path) == evidence["retry_records_sha256"]
        and assessment.get("pass") is True
        and assessment.get("capacity_adjusted_all_models_qualify") is True
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
            and int(model.get("max_tokens"))
            == int(config["max_output_tokens_per_call"])
        )
        probe_spec = config["production_probes"][name]
        probe_path = ROOT / probe_spec["path"]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_checks[name] = bool(
            _sha256(probe_path) == probe_spec["sha256"]
            and probe.get("model")
            == probe_spec["provider_model"]
            == model["model"]
            and probe.get("results")
            and all(result.get("accepted") for result in probe["results"])
        )
    checks["model_options"] = all(model_checks.values())
    checks["production_probes"] = all(probe_checks.values())

    frozen_checks = {
        relative: _sha256(ROOT / relative) == expected
        for relative, expected in gate["expected_sha256"].items()
    }
    checks["frozen_sources"] = all(frozen_checks.values())
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "analysis": "cdr_v1_3_primary_validation_preflight",
        "no_model_calls": True,
        "checks": checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": frozen_checks,
        "expected_calls": expected_calls,
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "conditional_self_history_calls_deferred": 216,
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())

