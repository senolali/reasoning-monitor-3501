"""Fresh calibration bank and checkpointed runner for CDR-v1.4."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.data.dialogue_streams import (
    generate_dialogue_bank,
    solve_dialogue_state,
)
from reasoning_monitor.dialogue_experiment import (
    load_dialogue_checkpoint,
    messages_sha256,
)
from reasoning_monitor.primary_dialogue_validation import (
    build_primary_validation_messages,
)

PROTOCOL_ID = "CDR-v1.4"
GENERATOR_VERSION = "controlled-dialogue-replication-v1"


def generate_replication_bank(seed: int = 20261001) -> dict[str, Any]:
    """Generate nine new, namespaced calibration dialogues deterministically."""
    source = generate_dialogue_bank(seed=seed)
    selected = [
        dialogue
        for dialogue in source["dialogues"]
        if dialogue["split"] == "calibration"
        and not dialogue["dialogue_id"].endswith("-004")
    ]
    dialogues = []
    for dialogue in selected:
        item = copy.deepcopy(dialogue)
        old_id = str(item["dialogue_id"])
        new_id = old_id.replace("cdr-v1-", "cdr-v14-")
        item["dialogue_id"] = new_id
        item["split"] = "replication_calibration"
        for turn in item["turns"]:
            turn["dialogue_id"] = new_id
            turn["qid"] = str(turn["qid"]).replace(old_id, new_id)
            turn["split"] = "replication_calibration"
        dialogues.append(item)
    return {
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "source_generator_version": source["generator_version"],
        "seed": seed,
        "conditions": ["user_only_history", "state_summary"],
        "dialogues": dialogues,
    }


def validate_replication_bank(bank: dict[str, Any]) -> list[str]:
    problems = []
    dialogues = bank.get("dialogues") or []
    if bank.get("protocol_id") != PROTOCOL_ID:
        problems.append("protocol_id")
    if len(dialogues) != 9:
        problems.append("dialogue_count")
    ids = [str(dialogue.get("dialogue_id")) for dialogue in dialogues]
    if len(ids) != len(set(ids)):
        problems.append("unique_dialogue_ids")
    for dialogue in dialogues:
        dialogue_id = str(dialogue.get("dialogue_id"))
        template = str(dialogue.get("template"))
        turns = dialogue.get("turns") or []
        if dialogue.get("split") != "replication_calibration":
            problems.append(f"split:{dialogue_id}")
        if len(turns) != 12:
            problems.append(f"turn_count:{dialogue_id}")
            continue
        previous_after = None
        for number, turn in enumerate(turns, start=1):
            if int(turn.get("turn", -1)) != number:
                problems.append(f"turn_order:{turn.get('qid')}")
            if previous_after is not None and turn.get("state_before") != previous_after:
                problems.append(f"state_continuity:{turn.get('qid')}")
            if str(solve_dialogue_state(template, turn["state_after"])) != str(
                turn.get("gold_answer")
            ):
                problems.append(f"gold:{turn.get('qid')}")
            previous_after = turn.get("state_after")
    return problems


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["dialogue_id"]),
        str(record["condition"]),
        int(record["turn"]),
    )


def run_replication_model(
    *,
    bank: dict[str, Any],
    client,
    model_name: str,
    selected_dialogue_ids: Sequence[str],
    conditions: Sequence[str],
    system_prompt: str,
    checkpoint_path: str | Path,
    seed: int,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one CDR-v1.4 model with per-call checkpointing and resume."""
    if tuple(conditions) != ("user_only_history", "state_summary"):
        raise ValueError("CDR-v1.4 requires user-only and state-summary conditions")
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    missing = [item for item in selected_dialogue_ids if item not in by_id]
    if missing:
        raise ValueError(f"unknown selected dialogues: {missing}")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        _record_key(record): record for record in load_dialogue_checkpoint(checkpoint)
    }
    expected_records = []

    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for dialogue_offset, dialogue_id in enumerate(selected_dialogue_ids):
            dialogue = by_id[dialogue_id]
            for condition_offset, condition in enumerate(conditions):
                prior_user_turns: list[str] = []
                for turn in dialogue["turns"]:
                    messages = build_primary_validation_messages(
                        turn=turn,
                        condition=str(condition),
                        prior_user_turns=prior_user_turns,
                        system_prompt=system_prompt,
                    )
                    key = (dialogue_id, str(condition), int(turn["turn"]))
                    record = saved.get(key)
                    if record is None:
                        call_seed = (
                            int(seed)
                            + 10_000 * dialogue_offset
                            + 1_000 * condition_offset
                            + 10 * int(turn["turn"])
                        )
                        sample = client.generate_messages(
                            messages,
                            seed=call_seed,
                            gold=str(turn["gold_answer"]),
                        )
                        record = {
                            "protocol_id": PROTOCOL_ID,
                            "history_transport": (
                                "packed_user_only_context"
                                if condition == "user_only_history"
                                else "verified_state_summary"
                            ),
                            "model": model_name,
                            "dialogue_id": dialogue_id,
                            "template": dialogue["template"],
                            "split": dialogue["split"],
                            "qid": turn["qid"],
                            "turn": int(turn["turn"]),
                            "condition": str(condition),
                            "role": "primary",
                            "call_index": 0,
                            "call_seed": call_seed,
                            "user_prompt": str(turn["user_prompt"]),
                            "messages": messages,
                            "messages_sha256": messages_sha256(messages),
                            "context_chars": len(messages[0]["content"]),
                            "gold_answer": str(turn["gold_answer"]),
                            "state_summary_before": turn["state_summary_before"],
                            "intrinsic_updates": int(turn["intrinsic_updates"]),
                            "raw_text": sample.raw_text,
                            "answer": sample.answer,
                            "reasoning_steps": list(sample.reasoning_steps),
                            "n_tokens": int(sample.n_tokens),
                            "latency_s": float(sample.latency_s),
                            "ok": bool(sample.ok),
                            "finish_reason": sample.finish_reason,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        saved[key] = record
                        if checkpoint_callback is not None:
                            checkpoint_callback(record)
                    expected_records.append(record)
                    prior_user_turns.append(str(turn["user_prompt"]))
    return expected_records
