"""Frozen science contract and CRL analysis for LIC-RQ-v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reasoning_monitor.lost_conversation import load_math_tasks, sha256_file
from reasoning_monitor.lost_conversation_runner import expected_call_keys

EXPECTED_MODELS = (
    ("gpt4o_mini", "openai/gpt4o_mini"),
    ("llama4_scout", "aws/llama4_scout-17b"),
    ("mistral_7b", "aws/mistral-7b"),
)
EXPECTED_CONDITIONS = ["dialogue_history", "fresh_prefix"]
EXPECTED_CALLS = 11_382


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Science paths must remain inside the repository")
    return path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash_check(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path).lower() == expected.lower()


def acquisition_fingerprint(root: Path, config: Mapping) -> str:
    """Hash request-defining inputs while excluding the authorization switch."""
    request_contract = {
        key: config[key]
        for key in (
            "protocol_id", "stage", "base_url", "temperature", "max_output_tokens",
            "k_repeats", "l_paraphrases", "request_options", "models", "conditions",
            "prompt_contract",
        )
    }
    manifest = _inside(root, config["manifest_path"])
    design = _inside(root, config["design_path"])
    runner = root / "src/reasoning_monitor/lost_conversation_runner.py"
    payload = json.dumps(request_contract, sort_keys=True, separators=(",", ":"))
    payload += "".join(sha256_file(path) for path in (manifest, design, runner))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_science_plan(root: Path, config_path: Path) -> tuple[dict, dict, dict, list[str], dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    integration_path = _inside(root, config["integration_config_path"])
    integration = yaml.safe_load(integration_path.read_text(encoding="utf-8"))
    manifest_path = _inside(root, config["manifest_path"])
    manifest = _read_json(manifest_path)
    design_path = _inside(root, config["design_path"])
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    tasks, _counts = load_math_tasks(
        _inside(root, design["source"]["path"]), design["source"]["sha256"])
    tasks_by_id = {task.task_id: task for task in tasks}
    task_ids = list(manifest["science_task_ids"])

    expected_models = list(EXPECTED_MODELS)
    actual_models = [(row["name"], row["provider_model"]) for row in config["models"]]
    checks = {
        "protocol_and_stage": config.get("protocol_id") == "LIC-RQ-v2"
                              and config.get("stage") == "frozen_science",
        "authorization_state_consistent": (
            isinstance(config.get("execution_authorized"), bool)
            and config.get("execution_authorized")
            == config.get("science_execution_authorized")),
        "manifest_protocol": manifest.get("protocol_id") == "LIC-RQ-v2",
        "manifest_hash": _hash_check(
            manifest_path, config["frozen_inputs"]["manifest_sha256"]),
        "models": actual_models == expected_models
                  and actual_models == [(row["name"], row["provider_model"])
                                        for row in manifest["models"]],
        "conditions": config.get("conditions") == EXPECTED_CONDITIONS
                      and config.get("conditions") == manifest.get("conditions"),
        "k_l": (config.get("k_repeats"), config.get("l_paraphrases")) == (5, 2),
        "science_task_count": len(task_ids) == 48 and len(set(task_ids)) == 48,
        "sample_disjoint": not set(task_ids) & set(manifest["integration_task_ids"]),
        "tasks_exist": set(task_ids) <= set(tasks_by_id),
        "prompt_contract": config.get("prompt_contract") == integration.get("prompt_contract"),
        "sampling_contract": all(config.get(key) == integration.get(key) for key in (
            "base_url", "api_key_env", "temperature", "max_output_tokens", "k_repeats",
            "l_paraphrases", "es_window", "request_options", "max_retries")),
        "integration_config_hash": _hash_check(
            integration_path, config["frozen_inputs"]["integration_config_sha256"]),
    }
    frozen_paths = {
        "integration_records": config["frozen_inputs"]["integration_records_path"],
        "semantic_assessment": config["frozen_inputs"]["semantic_assessment_path"],
        "readiness_assessment": config["frozen_inputs"]["readiness_assessment_path"],
    }
    for name, value in frozen_paths.items():
        checks[f"{name}_hash"] = _hash_check(
            _inside(root, value), config["frozen_inputs"][f"{name}_sha256"])
    checks["acquisition_runner_hash"] = _hash_check(
        root / "src/reasoning_monitor/lost_conversation_runner.py",
        config["frozen_inputs"]["acquisition_runner_sha256"])
    checks["science_acquisition_hash"] = _hash_check(
        root / "src/reasoning_monitor/lost_conversation_science_acquisition.py",
        config["frozen_inputs"]["science_acquisition_sha256"])
    checks["semantic_parser_hash"] = _hash_check(
        root / "src/reasoning_monitor/lost_conversation_semantic_rescore.py",
        config["frozen_inputs"]["semantic_parser_sha256"])

    expected = len(expected_call_keys(config, task_ids, tasks_by_id))
    checks["call_budget"] = (
        expected == EXPECTED_CALLS
        and expected == manifest["science_budget"]["logical_model_calls"])
    analysis = config.get("analysis", {})
    auxiliary = analysis.get("auxiliary_metric", {})
    checks["analysis_frozen"] = (
        analysis.get("primary_endpoint") == "paired_final_turn_success"
        and analysis.get("rqeval_pre_final_candidates") == ["CS", "LS", "SS"]
        and analysis.get("missingness_policy") == "no_imputation"
        and auxiliary.get("abbreviation") == "CRL"
        and auxiliary.get("threshold") is None
        and analysis.get("auxiliary_target") == "final_positive_readiness_gap")
    checks["empty_history_transport"] = config.get("transport") == {
        "empty_primary_history_placeholder": "[EMPTY_PRIMARY_RESPONSE]",
        "preserve_empty_raw_record": True,
        "retry_empty_primary": False,
    }
    checks["production_probes"] = _validate_probes(root, config)
    report = {
        "protocol_id": "LIC-RQ-v2",
        "stage": "frozen_science_call_free_preflight",
        "no_model_calls": True,
        "science_tasks": len(task_ids),
        "expected_calls": expected,
        "maximum_output_tokens": expected * int(config["max_output_tokens"]),
        "integration_tasks_excluded": list(manifest["integration_task_ids"]),
        "acquisition_fingerprint": acquisition_fingerprint(root, config),
        "analysis": analysis,
        "checks": checks,
        "execution_ready": all(checks.values()),
        "execution_authorized": bool(config.get("science_execution_authorized")),
    }
    return config, manifest, tasks_by_id, task_ids, report


def _validate_probes(root: Path, config: Mapping) -> bool:
    for model in config["models"]:
        probe = _read_json(_inside(root, model["probe"]))
        if (probe.get("model") != model["provider_model"]
                or probe.get("base_url") != config["base_url"]
                or probe.get("api_key_env") != config["api_key_env"]
                or not probe.get("executed")
                or not probe.get("results")
                or not all(row.get("accepted") for row in probe["results"])):
            return False
    return True


def cumulative_resolution_lag(records: Sequence[Mapping]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the threshold-free CRL auxiliary metric from semantic primaries.

    Invalid response pairs stay missing. CRL is the mean positive readiness gap
    over comparable pre-final turns, with no minimum-support filter or imputation.
    """
    primaries = [row for row in records if row.get("role") == "primary"]
    indexed = {
        (row["model"], row["task_id"], int(row["turn"]), row["condition"]): row
        for row in primaries
    }
    paired = []
    keys = sorted({key[:3] for key in indexed})
    for model, task_id, turn in keys:
        history = indexed.get((model, task_id, turn, "dialogue_history"))
        fresh = indexed.get((model, task_id, turn, "fresh_prefix"))
        if history is None or fresh is None:
            continue
        readiness = {}
        for name, row in (("dialogue", history), ("fresh", fresh)):
            kind = row.get("response_kind")
            readiness[name] = 1.0 if kind == "answer_attempt" else (
                0.0 if kind == "need_more_information" else math.nan)
        comparable = all(math.isfinite(value) for value in readiness.values())
        gap = readiness["fresh"] - readiness["dialogue"] if comparable else math.nan
        total = int(history["total_turns"])
        paired.append({
            "model": model,
            "task_id": task_id,
            "turn": turn,
            "total_turns": total,
            "information_fraction": turn / total,
            "dialogue_readiness": readiness["dialogue"],
            "fresh_readiness": readiness["fresh"],
            "readiness_gap": gap,
            "positive_gap": max(gap, 0.0) if comparable else math.nan,
            "comparable": comparable,
            "is_final": turn == total,
        })
    trajectory = pd.DataFrame(paired)
    summaries = []
    for (model, task_id), task in trajectory.groupby(["model", "task_id"]):
        pre = task[~task.is_final]
        comparable = pre[pre.comparable]
        positive_flags = [bool(value > 0) for value in comparable.readiness_gap]
        longest = current = 0
        for flag in positive_flags:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        final = task[task.is_final]
        final_gap = (float(final.iloc[0].readiness_gap)
                     if len(final) and bool(final.iloc[0].comparable) else None)
        expected_prefinal = int(task.total_turns.iloc[0]) - 1
        summaries.append({
            "model": model,
            "task_id": task_id,
            "CRL": (float(comparable.positive_gap.mean()) if len(comparable) else math.nan),
            "comparable_prefinal_turns": len(comparable),
            "expected_prefinal_turns": expected_prefinal,
            "prefinal_support_rate": len(comparable) / expected_prefinal,
            "longest_positive_gap_run": longest,
            "final_readiness_gap": final_gap,
            "final_positive_readiness_gap": (final_gap > 0 if final_gap is not None else None),
        })
    return trajectory, pd.DataFrame(summaries)


def plot_crl(trajectory: pd.DataFrame, summary: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    models = list(summary.model.unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.5 * len(models)),
                             constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for row_axes, model in zip(axes, models, strict=True):
        left, right = row_axes
        data = trajectory[trajectory.model == model]
        for _task_id, task in data.groupby("task_id"):
            left.plot(task.information_fraction, task.readiness_gap,
                      color="#999999", alpha=0.25, linewidth=0.8)
        mean = data.groupby("information_fraction").readiness_gap.mean()
        left.plot(mean.index, mean, "o-", color="#0072B2", linewidth=2, label="Mean")
        left.axhline(0, color="#444444", linewidth=1)
        left.set(title=f"{model}: matched readiness gap", xlabel="Revealed information fraction",
                 ylabel="Fresh - dialogue readiness", ylim=(-1.1, 1.1), xlim=(0, 1.03))
        left.grid(alpha=0.2)
        left.legend()
        values = summary[summary.model == model].CRL.dropna()
        right.hist(values, bins=[-0.01, 0.01, 0.21, 0.41, 0.61, 0.81, 1.01],
                   color="#D55E00", edgecolor="white")
        right.set(title=f"{model}: CRL across source tasks", xlabel="CRL", ylabel="Tasks",
                  xlim=(-0.03, 1.03))
        right.grid(axis="y", alpha=0.2)
    fig.suptitle("LIC-RQ-v2 cumulative resolution lag (CRL), threshold-free")
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"cumulative_resolution_lag.{extension}"
        fig.savefig(path, dpi=150)
        paths.append(path.name)
    plt.close(fig)
    return paths
