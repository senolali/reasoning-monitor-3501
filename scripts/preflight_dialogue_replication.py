"""Call-free preflight for the frozen CDR-v1.4 replication."""

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

from reasoning_monitor.dialogue_replication import (  # noqa: E402
    generate_replication_bank,
    validate_replication_bank,
)
from reasoning_monitor.primary_dialogue_validation import (  # noqa: E402
    build_primary_validation_messages,
)

CONFIG = ROOT / "configs" / "controlled_dialogue_v1_4_replication_asu.yaml"
ANALYSIS_GATE = (
    ROOT / "configs" / "controlled_dialogue_v1_4_replication_analysis_gate.yaml"
)
PREFLIGHT_GATE = (
    ROOT / "configs" / "controlled_dialogue_v1_4_replication_preflight_gate.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    analysis_gate = yaml.safe_load(ANALYSIS_GATE.read_text(encoding="utf-8"))
    preflight_gate = yaml.safe_load(PREFLIGHT_GATE.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    regenerated = generate_replication_bank(seed=int(config["seed"]))
    selected = list(config["selected_dialogue_ids"])
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    expected_calls = (
        len(selected)
        * int(config["turns_per_dialogue"])
        * len(config["conditions"])
        * len(config["models"])
    )
    checks = {
        "protocol": config.get("protocol_id")
        == analysis_gate.get("protocol_id")
        == preflight_gate.get("protocol_id")
        == "CDR-v1.4",
        "stage": config.get("stage") == "fresh_calibration_replication"
        and analysis_gate.get("stage")
        == "fresh_calibration_replication_analysis"
        and preflight_gate.get("stage")
        == "fresh_calibration_replication_preflight",
        "bank_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "bank_reproducible": bank == regenerated,
        "bank_valid": not validate_replication_bank(bank),
        "fresh_namespace": all(item.startswith("cdr-v14-") for item in selected),
        "all_bank_dialogues_selected": set(selected) == set(by_id) and len(selected) == 9,
        "turn_counts": all(
            len(by_id[item]["turns"]) == int(config["turns_per_dialogue"])
            for item in selected
        ),
        "conditions": config.get("conditions")
        == ["user_only_history", "state_summary"],
        "single_samples": int(config["k_repeats"]) == 1
        and int(config["l_paraphrases"]) == 0,
        "counts": expected_calls == int(config["expected_total_calls"]) == 432
        and int(config["expected_calls_per_model"]) == 216,
        "token_ceiling": int(config["max_output_tokens_per_call"]) == 2048
        and int(config["maximum_output_token_ceiling"])
        == expected_calls * int(config["max_output_tokens_per_call"]),
        "resume": bool(config["resume"]),
        "assay_validity_frozen": float(
            analysis_gate["minimum_state_summary_accuracy"]
        )
        == 0.90,
        "cluster_inference": int(
            analysis_gate["expected_post_baseline_pairs_per_model"]
        )
        == 99
        and float(analysis_gate["maximum_one_sided_sign_flip_p"]) == 0.05,
        "paid_calls_locked": preflight_gate.get("paid_calls_authorized") is False,
    }

    transport_checks = []
    for dialogue_id in selected:
        prior_user_turns: list[str] = []
        for turn in by_id[dialogue_id]["turns"]:
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
                and "[PRIOR ASSISTANT" not in user_system
                and "VERIFIED STATE BEFORE THIS TURN" in summary_system
                and all(prompt not in summary_system for prompt in prior_user_turns)
                and user_messages[1]["content"] == summary_messages[1]["content"]
            )
            prior_user_turns.append(str(turn["user_prompt"]))
    checks["transport_isolation"] = all(transport_checks)

    evidence = config["source_evidence"]
    assessment_path = ROOT / evidence["assessment_path"]
    sensitivity_path = ROOT / evidence["sensitivity_path"]
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    checks["source_evidence"] = bool(
        _sha256(assessment_path) == evidence["assessment_sha256"]
        and _sha256(sensitivity_path) == evidence["sensitivity_sha256"]
        and assessment.get("pass") is False
        and sensitivity.get("frozen_gate_preserved_as_failed") is True
        and sensitivity.get("cross_model_primary_effect_robust") is True
    )

    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    model_checks = {}
    probe_checks = {}
    for model in config["models"]:
        name = str(model["name"])
        model_checks[name] = bool(
            model.get("request_options") == expected_options
            and model.get("temperature") == 0.0
            and int(model.get("max_tokens")) == 2048
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

    frozen_checks = {
        relative: _sha256(ROOT / relative) == expected
        for relative, expected in preflight_gate["expected_sha256"].items()
    }
    checks["frozen_sources"] = bool(frozen_checks) and all(frozen_checks.values())
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "analysis": "cdr_v1_4_fresh_calibration_replication_preflight",
        "no_model_calls": True,
        "checks": checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": frozen_checks,
        "expected_calls": expected_calls,
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "live_progress": "completed/total, percent, calls/min, ETA after each call",
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
