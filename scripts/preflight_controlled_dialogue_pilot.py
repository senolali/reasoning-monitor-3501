"""Call-free structural and integration preflight for the CDR-v1 pilot."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.dialogue_streams import (  # noqa: E402
    CONDITIONS,
    dialogue_bank_sha256,
    generate_dialogue_bank,
    validate_dialogue_bank,
)
from reasoning_monitor.dialogue_experiment import build_turn_messages  # noqa: E402
from reasoning_monitor.models.clients import OpenAICompatibleClient  # noqa: E402

PROTOCOL_PATH = ROOT / "configs" / "controlled_dialogue_v1.yaml"
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_pilot_asu.yaml"
ANALYSIS_GATE_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_pilot_analysis_gate.yaml"
)
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_pilot_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    regenerated = generate_dialogue_bank()
    problems = []

    source_sha256 = {}
    for relative, expected in gate["expected_sha256"].items():
        digest = _sha256(ROOT / relative)
        source_sha256[relative] = digest
        if digest != expected:
            problems.append(f"hash mismatch: {relative}")

    bank_problems = validate_dialogue_bank(bank)
    problems.extend(f"bank:{problem}" for problem in bank_problems)
    checks = {
        "protocol_id": protocol.get("protocol_id") == "CDR-v1"
        and config.get("protocol_id") == "CDR-v1",
        "protocol_status": protocol.get("status") == "DESIGN_ONLY_NO_MODEL_CALLS",
        "bank_file_hash": _sha256(bank_path) == config["bank_file_sha256"],
        "bank_canonical_hash": dialogue_bank_sha256(bank)
        == gate["expected_bank_canonical_sha256"],
        "deterministic_regeneration": bank == regenerated,
        "bank_validation": not bank_problems,
        "conditions": tuple(config["conditions"]) == CONDITIONS,
        "k_l": int(config["k_repeats"]) == 1
        and int(config["l_paraphrases"]) == 0,
        "resume": bool(config["resume"]),
        "explicit_message_client": hasattr(OpenAICompatibleClient, "generate_messages"),
        "call_count": int(config["expected_total_calls"]) == 216,
        "token_ceiling": int(config["maximum_output_token_ceiling"])
        == 216 * int(config["max_output_tokens_per_call"]),
    }

    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    selected = [by_id.get(dialogue_id) for dialogue_id in config["selected_dialogue_ids"]]
    checks["selected_dialogues_exist"] = all(selected)
    if all(selected):
        checks["calibration_only"] = {dialogue["split"] for dialogue in selected} == {
            "calibration"
        }
        checks["template_coverage"] = Counter(
            dialogue["template"] for dialogue in selected
        ) == Counter({"research_budget": 1, "software_incident": 1, "study_cohort": 1})
        checks["turn_counts"] = all(len(dialogue["turns"]) == 12 for dialogue in selected)
        first = selected[0]["turns"]
        self_turn_2 = build_turn_messages(
            first[1],
            "self_history",
            [
                {"role": "user", "content": first[0]["user_prompt"]},
                {"role": "assistant", "content": first[0]["oracle_response"]},
            ],
            config["system_prompt"],
        )
        summary_turn_2 = build_turn_messages(
            first[1], "state_summary", [], config["system_prompt"]
        )
        checks["history_growth"] = len(self_turn_2) == 4
        checks["summary_isolation"] = len(summary_turn_2) == 2 and (
            "VERIFIED STATE BEFORE THIS TURN" in summary_turn_2[0]["content"]
        )

    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    probe_checks = {}
    for model in config["models"]:
        name = model["name"]
        probe_spec = config["production_probes"][name]
        probe_path = ROOT / probe_spec["path"]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        accepted = bool(probe.get("results")) and all(
            bool(result.get("accepted")) for result in probe["results"]
        )
        probe_checks[name] = bool(
            _sha256(probe_path) == probe_spec["sha256"]
            and probe.get("model") == probe_spec["provider_model"] == model["model"]
            and accepted
            and model.get("request_options") == expected_options
            and model.get("temperature") == 0.0
            and int(model.get("max_tokens")) == 512
        )
    checks["production_probes"] = all(probe_checks.values())
    checks = {name: bool(passed) for name, passed in checks.items()}
    problems.extend(name for name, passed in checks.items() if not passed)
    report = {
        "analysis": "cdr_v1_pilot_preflight",
        "no_model_calls": True,
        "checks": checks,
        "probe_checks": probe_checks,
        "bank_canonical_sha256": dialogue_bank_sha256(bank),
        "bank_file_sha256": _sha256(bank_path),
        "config_sha256": _sha256(CONFIG_PATH),
        "analysis_gate_sha256": _sha256(ANALYSIS_GATE_PATH),
        "source_sha256": source_sha256,
        "expected_calls": int(config["expected_total_calls"]),
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "pass": not problems,
        "paid_calls_authorized": False,
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
