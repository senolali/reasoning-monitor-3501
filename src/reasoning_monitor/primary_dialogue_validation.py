"""Checkpointed CDR-v1.3 state-summary versus user-only validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.dialogue_experiment import (
    build_turn_messages,
    load_dialogue_checkpoint,
    messages_sha256,
)
from reasoning_monitor.user_only_dialogue_experiment import build_user_only_messages


def build_primary_validation_messages(
    *,
    turn: dict[str, Any],
    condition: str,
    prior_user_turns: Sequence[str],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the two prospectively matched CDR-v1.3 transport conditions."""
    if condition == "user_only_history":
        return build_user_only_messages(
            current_user_prompt=str(turn["user_prompt"]),
            prior_user_turns=prior_user_turns,
            system_prompt=system_prompt,
        )
    if condition == "state_summary":
        return build_turn_messages(
            turn,
            condition,
            [],
            system_prompt,
            user_prompt=str(turn["user_prompt"]),
        )
    raise ValueError(f"unsupported CDR-v1.3 condition: {condition}")


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["dialogue_id"]),
        str(record["condition"]),
        int(record["turn"]),
    )


def run_primary_dialogue_validation_model(
    *,
    bank: dict[str, Any],
    client,
    model_name: str,
    selected_dialogue_ids: Sequence[str],
    conditions: Sequence[str],
    system_prompt: str,
    checkpoint_path: str | Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one CDR-v1.3 model with per-call checkpointing and resume."""
    if tuple(conditions) != ("user_only_history", "state_summary"):
        raise ValueError("CDR-v1.3 requires frozen user-only and summary conditions")
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
                            "protocol_id": "CDR-v1.3",
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
                    expected_records.append(record)
                    prior_user_turns.append(str(turn["user_prompt"]))
    return expected_records

