"""Call-free preflight for the 10-cell CDR-v1.2a capacity retry."""

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

from reasoning_monitor.dialogue_experiment import (  # noqa: E402
    load_dialogue_checkpoint,
    messages_sha256,
    strict_final_integer,
)

CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_asu.yaml"
ANALYSIS_GATE_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_analysis_gate.yaml"
)
GATE_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_gate.yaml"
ORIGINAL_CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_asu.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    analysis_gate = yaml.safe_load(ANALYSIS_GATE_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    original_config = yaml.safe_load(ORIGINAL_CONFIG_PATH.read_text(encoding="utf-8"))
    selected_ids = list(original_config["selected_dialogue_ids"])
    dialogue_offsets = {dialogue_id: index for index, dialogue_id in enumerate(selected_ids)}

    assessment_spec = config["source_assessment"]
    assessment_path = ROOT / assessment_spec["path"]
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    diagnostic_spec = config["source_diagnostic"]
    diagnostic_path = ROOT / diagnostic_spec["path"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    checks = {
        "protocol": config.get("protocol_id")
        == analysis_gate.get("protocol_id")
        == gate.get("protocol_id")
        == "CDR-v1.2a",
        "stage": config.get("stage") == "exact_message_capacity_retry"
        and analysis_gate.get("stage") == "exact_message_capacity_retry_analysis"
        and gate.get("stage") == "exact_message_capacity_retry_preflight",
        "source_assessment": _sha256(assessment_path)
        == assessment_spec["sha256"]
        and assessment.get("pass") is False,
        "source_diagnostic": _sha256(diagnostic_path)
        == diagnostic_spec["sha256"]
        and diagnostic.get("decision")
        == "freeze_exact_message_capacity_retry_for_censored_cells"
        and int(diagnostic.get("capacity_censored_cells", 0)) == 10,
        "larger_ceiling": int(config["retry_max_tokens"])
        > int(config["original_max_tokens"]),
        "token_ceiling": int(config["maximum_output_token_ceiling"])
        == int(config["expected_total_calls"]) * int(config["retry_max_tokens"]),
        "resume": bool(config["resume"]),
        "paid_calls_locked": gate.get("paid_calls_authorized") is False,
    }

    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    target_checks = {}
    model_checks = {}
    probe_checks = {}
    all_target_keys: set[tuple[str, str]] = set()
    target_count = 0
    for model_name, model_spec in config["models"].items():
        source_path = ROOT / model_spec["source_path"]
        records = load_dialogue_checkpoint(source_path)
        by_qid = {str(record["qid"]): record for record in records}
        source_ok = _sha256(source_path) == model_spec["source_sha256"]
        targets_ok = []
        for target in model_spec["targets"]:
            qid = str(target["qid"])
            record = by_qid.get(qid, {})
            dialogue_id = str(record.get("dialogue_id", ""))
            expected_seed = (
                int(config["seed"])
                + 10_000 * dialogue_offsets.get(dialogue_id, -100)
                + 10 * int(record.get("turn", 0))
            )
            key = (model_name, qid)
            targets_ok.append(
                key not in all_target_keys
                and int(record.get("n_tokens", 0))
                == int(config["original_max_tokens"])
                and strict_final_integer(str(record.get("raw_text", ""))) is None
                and str(record.get("gold_answer")) == str(target["gold_answer"])
                and messages_sha256(record.get("messages") or [])
                == target["messages_sha256"]
                and int(target["call_seed"]) == expected_seed
                and len(record.get("messages") or []) == 2
            )
            all_target_keys.add(key)
            target_count += 1
        target_checks[model_name] = bool(
            source_ok
            and len(model_spec["targets"]) == int(model_spec["expected_calls"])
            and all(targets_ok)
        )

        client = model_spec["client"]
        model_checks[model_name] = bool(
            client.get("request_options") == expected_options
            and client.get("temperature") == 0.0
            and int(client.get("max_tokens")) == int(config["retry_max_tokens"])
        )
        probe_spec = model_spec["production_probe"]
        probe_path = ROOT / probe_spec["path"]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_checks[model_name] = bool(
            _sha256(probe_path) == probe_spec["sha256"]
            and probe.get("model")
            == probe_spec["provider_model"]
            == client["model"]
            and probe.get("results")
            and all(result.get("accepted") for result in probe["results"])
        )

    checks["target_count"] = target_count == int(config["expected_total_calls"]) == 10
    checks["exact_frozen_targets"] = all(target_checks.values())
    checks["model_options"] = all(model_checks.values())
    checks["production_probes"] = all(probe_checks.values())
    frozen_checks = {
        relative: _sha256(ROOT / relative) == expected
        for relative, expected in gate["expected_sha256"].items()
    }
    checks["frozen_sources"] = all(frozen_checks.values())
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "analysis": "cdr_v1_2a_exact_message_capacity_retry_preflight",
        "no_model_calls": True,
        "checks": checks,
        "target_checks": target_checks,
        "model_checks": model_checks,
        "production_probe_checks": probe_checks,
        "frozen_source_checks": frozen_checks,
        "expected_calls": target_count,
        "maximum_output_tokens": int(config["maximum_output_token_ceiling"]),
        "pass": all(checks.values()),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return int(not report["pass"])


if __name__ == "__main__":
    raise SystemExit(main())
