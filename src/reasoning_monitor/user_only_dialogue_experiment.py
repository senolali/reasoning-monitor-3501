"""Checkpointed CDR-v1.2 user-only history mechanism arm."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.dialogue_experiment import (
    load_dialogue_checkpoint,
    messages_sha256,
)


def build_user_only_messages(
    *,
    current_user_prompt: str,
    prior_user_turns: Sequence[str],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Pack only prior user turns into the system context."""
    if prior_user_turns:
        blocks = "\n\n".join(
            f"[PRIOR USER TURN {index}]\n{content}"
            for index, content in enumerate(prior_user_turns, start=1)
        )
        system = (
            f"{system_prompt}\n\n"
            "CALLER-MANAGED PRIOR USER HISTORY:\n"
            "The blocks below contain every prior user update for this case. "
            "Assistant responses are intentionally omitted. Reconstruct the "
            "running state from the user facts before answering.\n\n"
            f"{blocks}"
        )
    else:
        system = system_prompt
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(current_user_prompt)},
    ]


def _record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["dialogue_id"]), int(record["turn"])


def run_user_only_dialogue_model(
    *,
    bank: dict[str, Any],
    client,
    model_name: str,
    selected_dialogue_ids: Sequence[str],
    system_prompt: str,
    checkpoint_path: str | Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one model with prior user facts and no prior assistant responses."""
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
            prior_user_turns: list[str] = []
            for turn in dialogue["turns"]:
                key = (dialogue_id, int(turn["turn"]))
                messages = build_user_only_messages(
                    current_user_prompt=str(turn["user_prompt"]),
                    prior_user_turns=prior_user_turns,
                    system_prompt=system_prompt,
                )
                record = saved.get(key)
                if record is None:
                    call_seed = (
                        int(seed)
                        + 10_000 * dialogue_offset
                        + 10 * int(turn["turn"])
                    )
                    sample = client.generate_messages(
                        messages,
                        seed=call_seed,
                        gold=str(turn["gold_answer"]),
                    )
                    record = {
                        "protocol_id": "CDR-v1.2",
                        "history_transport": "packed_user_only_context",
                        "model": model_name,
                        "dialogue_id": dialogue_id,
                        "template": dialogue["template"],
                        "split": dialogue["split"],
                        "qid": turn["qid"],
                        "turn": int(turn["turn"]),
                        "condition": "user_only_history",
                        "role": "primary",
                        "call_index": 0,
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

