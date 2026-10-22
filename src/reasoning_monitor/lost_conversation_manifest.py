"""Freeze the call-free LIC-RQ-v2 sample, observability and budget contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import yaml

from reasoning_monitor.lost_conversation import load_math_tasks, sha256_file


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest paths must remain inside the repository")
    return path


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_config(config: Mapping) -> None:
    expected = {
        "protocol_id": "LIC-RQ-v2",
        "stage": "frozen_call_free_manifest",
        "execution_authorized": False,
        "conditions": ["dialogue_history", "fresh_prefix"],
        "k_repeats": 5,
        "l_paraphrases": 2,
        "es_window": 5,
        "revelation_order": "source_shard_id",
        "termination": "all_shards_no_correctness_stop",
        "primary_response_index": 0,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("LIC-RQ-v2 frozen policy changed")
    models = config.get("models")
    names = [row.get("name") for row in models] if isinstance(models, list) else []
    providers = [row.get("provider_model") for row in models] if isinstance(models, list) else []
    if names != ["gpt4o_mini", "llama4_scout", "mistral_7b"]:
        raise ValueError("The accepted three-model family is required")
    if providers != ["openai/gpt4o_mini", "aws/llama4_scout-17b", "aws/mistral-7b"]:
        raise ValueError("Unexpected ASU provider model mapping")
    analysis = config.get("analysis", {})
    if analysis.get("pre_final_candidate_metrics") != ["CS", "LS", "SS"]:
        raise ValueError("Only gold-independent pre-final candidates are allowed")
    if analysis.get("missingness_policy") != "no_imputation":
        raise ValueError("Missing metrics must remain missing")


def _budget(task_ids: list[str], tasks_by_id: Mapping, config: Mapping) -> dict:
    turns = sum(len(tasks_by_id[task_id].shards) for task_id in task_ids)
    cells = turns * len(config["models"]) * len(config["conditions"])
    calls = cells * (config["k_repeats"] + config["l_paraphrases"])
    return {
        "source_tasks": len(task_ids),
        "prefixes_per_model_condition": turns,
        "shards_per_task": dict(sorted(Counter(
            len(tasks_by_id[task_id].shards) for task_id in task_ids
        ).items())),
        "model_condition_cells": cells,
        "logical_model_calls": calls,
        "maximum_output_tokens": calls * config["max_output_tokens"],
        "currency_cost": None,
    }


def build_manifest(root: Path, config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    design_path = _inside(root, config["source_design"])
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    source = design["source"]
    source_path = _inside(root, source["path"])
    tasks, _ = load_math_tasks(source_path, source["sha256"])
    tasks_by_id = {task.task_id: task for task in tasks}

    options_path = _inside(root, config["source_audit"])
    options = _load_json(options_path)
    selected = next((row for row in options["options"]
                     if row.get("requested_science_tasks") == config["sample_option_tasks"]
                     and row.get("available") is True), None)
    if selected is None:
        raise ValueError("Frozen science sample option is unavailable")
    integration_ids = list(options["integration_task_ids"])
    science_ids = list(selected["science_task_ids"])
    if len(integration_ids) != 4 or len(science_ids) != 48:
        raise ValueError("Unexpected LIC task counts")
    if set(integration_ids) & set(science_ids):
        raise ValueError("Integration and science tasks overlap")
    if not set(integration_ids + science_ids) <= set(tasks_by_id):
        raise ValueError("Unknown source task in sample")

    semantic_path = _inside(root, config["source_semantic_audit"])
    semantics = _load_json(semantic_path)
    allowed = {row["task_id"] for row in semantics["tasks"]
               if row["status"] == "candidate"}
    if not set(integration_ids + science_ids) <= allowed:
        raise ValueError("Quarantined source task entered the frozen sample")

    integration_budget = _budget(integration_ids, tasks_by_id, config)
    science_budget = _budget(science_ids, tasks_by_id, config)
    total_budget = _budget(integration_ids + science_ids, tasks_by_id, config)
    manifest = {
        "protocol_id": config["protocol_id"],
        "stage": config["stage"],
        "no_model_calls": True,
        "execution_authorized": False,
        "execution_ready": False,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source["sha256"],
        "source_options_sha256": sha256_file(options_path),
        "source_semantic_audit_sha256": sha256_file(semantic_path),
        "selection_uses_model_outcomes": False,
        "models": config["models"],
        "conditions": config["conditions"],
        "integration_task_ids": integration_ids,
        "science_task_ids": science_ids,
        "integration_budget": integration_budget,
        "science_budget": science_budget,
        "total_budget": total_budget,
        "metric_observability": {
            "pre_final": {
                "CQ": "missing_until_gold_eligible",
                "CS": "measured_for_valid_answer_attempts",
                "RS": "missing_until_gold_eligible_and_correct_primary",
                "LS": "measured_for_valid_reasoning_steps",
                "ES": "missing_until_gold_eligible_and_window_warm",
                "SS": "measured_for_valid_repeated_reasoning_traces",
                "RQ6_mean": "missing_unless_all_six_measured",
            },
            "final": {
                "CQ": "measured_for_valid_extraction_else_failure_in_endpoint",
                "CS": "measured_for_valid_answer_attempts",
                "RS": "measured_only_for_correct_primary",
                "LS": "measured_for_valid_reasoning_steps",
                "ES": "measured_when_causal_window_is_eligible",
                "SS": "measured_for_valid_repeated_reasoning_traces",
                "RQ6_mean": "complete_case_only",
            },
        },
        "primary_claim": "paired final-turn success under dialogue history versus fresh prefix",
        "early_warning_claim": (
            "prospective association of pre-final CS/LS/SS with final-turn failure; "
            "no imputation and no claim for unavailable CQ/RS/ES"
        ),
        "integration_is_effect_gate": False,
        "science_may_start_without_new_explicit_approval": False,
        "next_authorizable_action": "four-task technical integration only",
        "pending_before_integration_approval": [
            "frozen task-specific L=2 paraphrases",
            "resumable three-model runner with progress and ETA",
            "call-free fixture replay for extraction and all metric statuses",
            "real local NLI and BERTScore preflight",
            "PNG and PDF seven-panel plot preflight",
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode("utf-8")
    manifest["manifest_content_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    assessment = {
        "protocol_id": config["protocol_id"],
        "no_model_calls": True,
        "manifest_valid": True,
        "execution_ready": False,
        "selected_dataset": "Microsoft Lost in Conversation / math",
        "integration_tasks": len(integration_ids),
        "science_tasks": len(science_ids),
        "models": [row["name"] for row in config["models"]],
        "logical_calls": total_budget["logical_model_calls"],
        "decision": (
            "Freeze one technical integration and one 48-task science campaign; "
            "do not run effect-seeking intermediate pilots."
        ),
        "important_limit": (
            "Most partial prefixes lack gold-answer eligibility. All seven panels remain, "
            "but missing CQ/RS/ES values are shown as missing rather than fabricated."
        ),
        "execution_authorized": False,
    }
    return {"execution_manifest.json": manifest, "assessment.json": assessment}


def write_artifacts(output: Path, artifacts: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
        if not path.exists():
            path.write_bytes(data)
