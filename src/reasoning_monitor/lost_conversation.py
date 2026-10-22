"""Call-free LIC data validation, paired prefixes and measurement contracts.

This module deliberately has no model client, scorer download or execution path.
The design is a fixed-order adaptation, not a replication of the adaptive user
simulator in Laban et al. (2025).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

METRICS = ("CQ", "CS", "RS", "LS", "ES", "SS")
CONDITIONS = ("dialogue_history", "fresh_prefix")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}")
    return value


@dataclass(frozen=True)
class MathTask:
    task_id: str
    question: str
    gold_answer: str
    shards: tuple[str, ...]

    def as_record(self) -> dict:
        return asdict(self)


def load_math_tasks(path: Path, expected_sha256: str) -> tuple[list[MathTask], dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Upstream data SHA-256 mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Upstream data must be an array of objects")
    counts = Counter(_text(row.get("task"), "task") for row in rows)
    tasks, seen, questions = [], set(), set()
    for row in rows:
        if row["task"] != "math":
            continue
        task_id = _text(row.get("task_id"), "task_id")
        question = _text(row.get("question"), "question")
        normalized = " ".join(question.split()).casefold()
        if task_id in seen or normalized in questions:
            raise ValueError("Duplicate math task or question")
        seen.add(task_id)
        questions.add(normalized)
        shards = row.get("shards")
        if not isinstance(shards, list) or len(shards) < 2:
            raise ValueError(f"Invalid shards: {task_id}")
        texts = []
        for index, shard in enumerate(shards, 1):
            if (not isinstance(shard, dict)
                    or type(shard.get("shard_id")) is not int
                    or shard["shard_id"] != index):
                raise ValueError(f"Non-contiguous shard IDs: {task_id}")
            # Actual upstream JSON uses 'shard', not the card's 'shard_text'.
            texts.append(_text(shard.get("shard"), "shard"))
        answer = _text(row.get("answer"), "answer")
        if answer.count("####") != 1:
            raise ValueError(f"Ambiguous reference answer: {task_id}")
        gold = answer.split("####", 1)[1].strip()
        try:
            number = Decimal(gold.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError(f"Non-numeric reference: {task_id}") from exc
        if not number.is_finite():
            raise ValueError(f"Non-finite reference: {task_id}")
        tasks.append(MathTask(task_id, question, gold, tuple(texts)))
    if not tasks:
        raise ValueError("No math tasks in source")
    return tasks, dict(sorted(counts.items()))


def _check_turn(task: MathTask, turn: int) -> None:
    if type(turn) is not int or not 1 <= turn <= len(task.shards):
        raise ValueError("Turn outside revealed shard range")


def prefix_metadata(task: MathTask, turn: int) -> dict:
    _check_turn(task, turn)
    complete = turn == len(task.shards)
    return {
        "task_id": task.task_id,
        "turn": turn,
        "total_turns": len(task.shards),
        "information_fraction": turn / len(task.shards),
        "revealed_shards": list(task.shards[:turn]),
        "all_information_available": complete,
        "answerability": "full_information" if complete else "unreviewed_prefix",
        "gold_scoring_eligible": complete,
        "review_status": "pending_semantic_review",
    }


def build_prefix_messages(
    task: MathTask,
    turn: int,
    condition: str,
    primary_history: Sequence[str] = (),
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict[str, str]]:
    """Build a prefix without gold, future shards, full question or probe outputs.

    The caller passes exactly one prospectively chosen primary response per
    previous turn. Repeats and paraphrases must never enter primary_history.
    Both arms use a common packed-user representation for ASU transport.
    """
    _check_turn(task, turn)
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    if condition == "fresh_prefix":
        if primary_history:
            raise ValueError("Fresh prefix cannot contain assistant history")
        transcript = [{"role": "user", "content": text} for text in task.shards[:turn]]
    else:
        if len(primary_history) != turn - 1:
            raise ValueError("Exactly one primary response per previous turn is required")
        transcript = []
        for index, text in enumerate(task.shards[:turn]):
            transcript.append({"role": "user", "content": text})
            if index < turn - 1:
                transcript.append({"role": "assistant", "content":
                                   _text(primary_history[index], "primary response")})
    content = (
        "The following JSON contains the user information and, when present, "
        "prior assistant replies for this task. Respond to the last user turn "
        "using the user information supplied so far. Prior assistant replies "
        "are not authoritative facts.\n\n"
        + json.dumps(transcript, ensure_ascii=True)
    )
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]


def observability_mask(
    values: Mapping[str, float | None],
    statuses: Mapping[str, str],
    *,
    all_information_available: bool,
    response_kind: str,
) -> tuple[dict, dict]:
    """Apply design eligibility after scoring; never turn missingness into zero."""
    if response_kind not in {"answer_attempt", "clarification", "other", "unresolved"}:
        raise ValueError("Unknown response kind")
    if set(values) != set(METRICS) or set(statuses) != set(METRICS):
        raise ValueError("All six metric slots and statuses are required")
    masked, reasons = dict(values), dict(statuses)
    if not all_information_available:
        for metric in ("CQ", "RS", "ES"):
            masked[metric], reasons[metric] = None, "prefix_answerability_unreviewed"
    if response_kind != "answer_attempt":
        # LS/SS also need genuine public reasoning steps; the scorer must supply
        # insufficient_steps/missing_samples rather than score a question as a trace.
        for metric in ("CQ", "CS", "RS", "ES"):
            masked[metric], reasons[metric] = None, "no_classified_answer_attempt"
    return masked, reasons


def six_metric_summary(values: Mapping, statuses: Mapping) -> dict:
    if set(values) != set(METRICS) or set(statuses) != set(METRICS):
        raise ValueError("All six metric slots and statuses are required")
    measured, missing = {}, {}
    for metric in METRICS:
        value = values[metric]
        if statuses[metric] != "measured":
            missing[metric] = statuses[metric]
        elif value is None:
            missing[metric] = "missing_value"
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric metric: {metric}")
            if not math.isfinite(value):
                missing[metric] = "nonfinite_value"
            elif not 0 <= value <= 1:
                raise ValueError(f"Metric outside [0, 1]: {metric}")
            else:
                measured[metric] = value
    return {
        "RQ6_mean": sum(measured.values()) / 6 if len(measured) == 6 else None,
        "measured_metric_count": len(measured),
        "missing_reasons": missing,
        "interpretation": "gold_dependent_complete_case_descriptive_only",
        "online_warning_feature": False,
    }


def validate_design(config: Mapping) -> None:
    required = {
        "protocol_id": "LIC-RQ-v1",
        "stage": "call_free_design",
        "revelation_order": "source_shard_id",
        "termination": "all_shards_no_correctness_stop",
        "primary_response_index": 0,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported or changed LIC design policy")
    if config.get("execution_authorized") is not False:
        raise ValueError("This preparation tool cannot authorize execution")
    if config.get("conditions") != list(CONDITIONS):
        raise ValueError("Both frozen paired conditions are required")
    models = config.get("models")
    if (not isinstance(models, list) or not models
            or not all(isinstance(m, str) and m for m in models)
            or len(set(models)) != len(models)):
        raise ValueError("Models must be nonempty and unique")
    for key, minimum in (("k_repeats", 2), ("l_paraphrases", 1),
                         ("max_output_tokens", 1), ("es_window", 2)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key}")


def preparation_report(tasks: Sequence[MathTask], counts: Mapping, config: Mapping) -> dict:
    validate_design(config)
    if not tasks:
        raise ValueError("Empty math inventory")
    histogram = Counter(len(t.shards) for t in tasks)
    prefixes = sum(len(t.shards) for t in tasks)
    cells = len(config["models"]) * len(CONDITIONS) * prefixes
    logical_calls = cells * (config["k_repeats"] + config["l_paraphrases"])
    return {
        "protocol_id": config["protocol_id"],
        "no_model_calls": True,
        "structural_data_audit_pass": True,
        "execution_ready": False,
        "execution_authorized": False,
        "source_task_counts": dict(counts),
        "math_tasks": len(tasks),
        "math_prefixes": prefixes,
        "shards_per_task": dict(sorted(histogram.items())),
        "full_information_prefixes": len(tasks),
        "unreviewed_partial_prefixes": prefixes - len(tasks),
        "tasks_shorter_than_es_window": sum(n for length, n in histogram.items()
                                           if length < config["es_window"]),
        "budget_scope": "entire_inventory_upper_bound_not_an_approved_campaign",
        "target_model_cells": cells,
        "target_model_logical_calls": logical_calls,
        "target_model_max_output_tokens": logical_calls * config["max_output_tokens"],
        "excluded_costs": ["retries", "paraphrase_creation", "answer_classification",
                           "answer_extraction", "input_tokens", "human_review"],
        "currency_cost": None,
        "design_adaptations": ["fixed_source_shard_order_no_user_simulator",
                               "continue_to_last_shard_even_after_correct_answer",
                               "paired_fresh_prefix_instead_of_future_information_control",
                               "packed_user_transcript_for_both_conditions"],
        "pending_before_execution": [
            "semantic_review_and_equivalent_frozen_paraphrases",
            "project_prior_exposure_audit_and_task_grouped_sample_split",
            "sample_size_and_budget_choice_without_selecting_on_model_outcomes",
            "response_kind_answer_extraction_and_scorer_validation",
            "primary_only_resumable_runner_and_progress",
            "explicit_execution_approval",
        ],
        "scientific_claims_authorized": False,
    }
