"""Science-only LIC acquisition with explicit empty-history transport."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from reasoning_monitor.dialogue_experiment import messages_sha256
from reasoning_monitor.lost_conversation import MathTask
from reasoning_monitor.lost_conversation_runner import (
    _load_checkpoint,
    _messages,
    expected_call_keys,
    parse_response,
)

EMPTY_HISTORY_PLACEHOLDER = "[EMPTY_PRIMARY_RESPONSE]"


def acquire_science(
    *,
    config: Mapping,
    tasks: Mapping[str, MathTask],
    task_ids: Sequence[str],
    clients: Mapping[str, object],
    checkpoint: Path,
    config_hash: str,
    callback: Callable[[int], None],
) -> list[dict]:
    """Resume acquisition while preserving empty primary responses as invalid.

    The raw empty record is never rewritten or retried. A deterministic visible
    placeholder is used only when transporting that response into later dialogue
    history, because the shared message builder rejects empty assistant content.
    """
    expected = expected_call_keys(config, task_ids, tasks)
    saved = _load_checkpoint(checkpoint, config_hash)
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen science sample")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    contract = config["prompt_contract"]
    primary_instruction = contract["primary_instruction"]
    paraphrases = contract["paraphrase_instructions"]
    if len(paraphrases) != config["l_paraphrases"] or len(set(paraphrases)) != len(paraphrases):
        raise ValueError("Exactly two distinct frozen instruction paraphrases are required")

    with checkpoint.open("a", encoding="utf-8") as stream:
        for model in config["models"]:
            client = clients[model["name"]]
            for task_id in task_ids:
                task = tasks[task_id]
                for condition in config["conditions"]:
                    history: list[str] = []
                    for turn in range(1, len(task.shards) + 1):
                        requests = [
                            ("primary" if index == 0 else "repeat", index,
                             primary_instruction)
                            for index in range(config["k_repeats"])
                        ]
                        requests += [
                            ("paraphrase", index, text)
                            for index, text in enumerate(paraphrases)
                        ]
                        primary_text = None
                        for role, index, instruction in requests:
                            key = (model["name"], task_id, condition, turn, role, index)
                            messages = _messages(
                                task, turn, condition, history,
                                contract["system_prompt"], instruction)
                            request_hash = messages_sha256(messages)
                            if key in saved:
                                row = saved[key]
                                if row.get("messages_sha256") != request_hash:
                                    raise ValueError("Changed request on resume")
                            else:
                                seed = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
                                sample = client.generate_messages(messages, seed=seed, gold="")
                                parsed = parse_response(sample.raw_text)
                                row = {
                                    "protocol_id": config["protocol_id"],
                                    "model": model["name"],
                                    "provider_model": model["provider_model"],
                                    "task_id": task_id,
                                    "condition": condition,
                                    "turn": turn,
                                    "total_turns": len(task.shards),
                                    "role": role,
                                    "call_index": index,
                                    "all_information_available": turn == len(task.shards),
                                    "messages": messages,
                                    "messages_sha256": request_hash,
                                    "config_hash": config_hash,
                                    "call_seed": seed,
                                    "raw_text": sample.raw_text,
                                    "answer": parsed["answer"],
                                    "response_kind": parsed["kind"],
                                    "contract_valid": parsed["contract_valid"],
                                    "reasoning_steps": list(sample.reasoning_steps),
                                    "ok": bool(sample.ok),
                                    "n_tokens": int(sample.n_tokens),
                                    "max_tokens": int(config["max_output_tokens"]),
                                    "latency_s": sample.latency_s,
                                    "finish_reason": sample.finish_reason,
                                }
                                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                                stream.flush()
                                saved[key] = row
                                callback(len(saved))
                            if role == "primary":
                                primary_text = str(row.get("raw_text", ""))
                        if primary_text is None:
                            raise ValueError("Missing primary response record")
                        if condition == "dialogue_history":
                            history.append(primary_text if primary_text.strip()
                                           else EMPTY_HISTORY_PLACEHOLDER)
    if set(saved) != expected:
        raise ValueError("Incomplete acquisition")
    return [saved[key] for key in sorted(saved)]

