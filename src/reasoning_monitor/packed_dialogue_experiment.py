"""Checkpointed CDR-v1.1 execution with gateway-safe packed histories."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.dialogue_experiment import (
    build_turn_messages,
    load_dialogue_checkpoint,
    messages_sha256,
)
from reasoning_monitor.dialogue_transport import build_packed_turn_messages
from reasoning_monitor.types import Sample


def _record_key(record: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(record["dialogue_id"]),
        str(record["condition"]),
        int(record["turn"]),
        str(record["role"]),
        int(record["call_index"]),
    )


def _sample_from_record(record: dict[str, Any]) -> Sample:
    return Sample(
        answer=str(record.get("answer", "")),
        reasoning_steps=tuple(record.get("reasoning_steps") or ()),
        latency_s=float(record.get("latency_s", 0.0)),
        n_tokens=int(record.get("n_tokens", 0)),
        raw_text=str(record.get("raw_text", "")),
        ok=bool(record.get("ok", False)),
        finish_reason=record.get("finish_reason"),
    )


def packed_pilot_messages(
    *,
    turn: dict[str, Any],
    condition: str,
    history: Sequence[dict[str, str]],
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Build exactly two transport messages for every CDR-v1.1 request."""
    if condition == "state_summary" or not history:
        return build_turn_messages(
            turn,
            condition,
            history,
            system_prompt,
            user_prompt=user_prompt,
        )
    return build_packed_turn_messages(
        current_user_prompt=user_prompt,
        history=history,
        system_prompt=system_prompt,
    )


def run_packed_dialogue_model(
    *,
    bank: dict[str, Any],
    client,
    model_name: str,
    selected_dialogue_ids: Sequence[str],
    conditions: Sequence[str],
    l_paraphrases: int,
    system_prompt: str,
    checkpoint_path: str | Path,
    seed: int,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run one model with packed history and per-call resume checkpoints."""
    if l_paraphrases < 0 or l_paraphrases > 2:
        raise ValueError("CDR-v1.1 supports zero to two frozen paraphrases")
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    missing = [item for item in selected_dialogue_ids if item not in by_id]
    if missing:
        raise ValueError(f"unknown selected dialogues: {missing}")

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing = load_dialogue_checkpoint(checkpoint)
    saved = {_record_key(record): record for record in existing}
    expected_records = []

    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for dialogue_offset, dialogue_id in enumerate(selected_dialogue_ids):
            dialogue = by_id[dialogue_id]
            for condition_offset, condition in enumerate(conditions):
                history: list[dict[str, str]] = []
                for turn in dialogue["turns"]:
                    prompts = [("primary", 0, str(turn["user_prompt"]))]
                    prompts.extend(
                        ("paraphrase", index, str(prompt))
                        for index, prompt in enumerate(
                            (turn.get("paraphrases") or [])[:l_paraphrases]
                        )
                    )
                    primary_sample = None
                    for role, call_index, prompt in prompts:
                        messages = packed_pilot_messages(
                            turn=turn,
                            condition=str(condition),
                            history=history,
                            system_prompt=system_prompt,
                            user_prompt=prompt,
                        )
                        if len(messages) != 2:
                            raise RuntimeError("packed transport must use two messages")
                        key = (
                            dialogue_id,
                            str(condition),
                            int(turn["turn"]),
                            role,
                            int(call_index),
                        )
                        record = saved.get(key)
                        if record is None:
                            call_seed = (
                                int(seed)
                                + 10_000 * dialogue_offset
                                + 1_000 * condition_offset
                                + 10 * int(turn["turn"])
                                + call_index
                            )
                            sample = client.generate_messages(
                                messages,
                                seed=call_seed,
                                gold=str(turn["gold_answer"]),
                            )
                            record = {
                                "protocol_id": "CDR-v1.1",
                                "history_transport": "packed_system_context",
                                "model": model_name,
                                "dialogue_id": dialogue_id,
                                "template": dialogue["template"],
                                "split": dialogue["split"],
                                "qid": turn["qid"],
                                "turn": int(turn["turn"]),
                                "condition": str(condition),
                                "role": role,
                                "call_index": int(call_index),
                                "user_prompt": prompt,
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
                            if checkpoint_callback is not None:
                                checkpoint_callback(record)
                            saved[key] = record
                        expected_records.append(record)
                        if role == "primary":
                            primary_sample = _sample_from_record(record)

                    if primary_sample is None:
                        raise RuntimeError(f"missing primary response for {turn['qid']}")
                    if condition == "self_history":
                        history.extend(
                            [
                                {"role": "user", "content": str(turn["user_prompt"])},
                                {"role": "assistant", "content": primary_sample.raw_text},
                            ]
                        )
                    elif condition == "oracle_history":
                        history.extend(
                            [
                                {"role": "user", "content": str(turn["user_prompt"])},
                                {
                                    "role": "assistant",
                                    "content": str(turn["oracle_response"]),
                                },
                            ]
                        )
    return expected_records

