"""Checkpointed execution for controlled longitudinal dialogue streams."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.metrics.dimensions import answers_equal
from reasoning_monitor.types import Sample

_FINAL_INTEGER = re.compile(
    r"^FINAL_ANSWER:\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*$"
)


def strict_final_integer(raw_text: str) -> str | None:
    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]
    if not lines:
        return None
    match = _FINAL_INTEGER.fullmatch(lines[-1])
    return match.group(1) if match else None


def messages_sha256(messages: Sequence[dict[str, str]]) -> str:
    canonical = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_turn_messages(
    turn: dict[str, Any],
    condition: str,
    history: Sequence[dict[str, str]],
    system_prompt: str,
    *,
    user_prompt: str | None = None,
) -> list[dict[str, str]]:
    current_prompt = user_prompt or str(turn["user_prompt"])
    if condition == "state_summary":
        summary = str(turn["state_summary_before"])
        return [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nVERIFIED STATE BEFORE THIS TURN:\n{summary}",
            },
            {"role": "user", "content": current_prompt},
        ]
    if condition not in {"self_history", "oracle_history"}:
        raise ValueError(f"unknown dialogue condition: {condition}")
    return [
        {"role": "system", "content": system_prompt},
        *[dict(message) for message in history],
        {"role": "user", "content": current_prompt},
    ]


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


def _record_key(record: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(record["dialogue_id"]),
        str(record["condition"]),
        int(record["turn"]),
        str(record["role"]),
        int(record["call_index"]),
    )


def load_dialogue_checkpoint(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _model_call(
    client,
    messages: list[dict[str, str]],
    *,
    seed: int,
    gold: str,
) -> Sample:
    generate_messages = getattr(client, "generate_messages", None)
    if generate_messages is None:
        raise TypeError("dialogue execution requires client.generate_messages()")
    return generate_messages(messages, seed=seed, gold=gold)


def run_dialogue_model(
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
    """Run one model with branch-safe histories and per-call checkpointing."""
    if l_paraphrases < 0 or l_paraphrases > 2:
        raise ValueError("CDR-v1 supports zero to two frozen paraphrases")
    by_id = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    missing = [dialogue_id for dialogue_id in selected_dialogue_ids if dialogue_id not in by_id]
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
                        messages = build_turn_messages(
                            turn,
                            condition,
                            history,
                            system_prompt,
                            user_prompt=prompt,
                        )
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
                            sample = _model_call(
                                client,
                                messages,
                                seed=call_seed,
                                gold=str(turn["gold_answer"]),
                            )
                            record = {
                                "protocol_id": bank["protocol_id"],
                                "model": model_name,
                                "dialogue_id": dialogue_id,
                                "template": dialogue["template"],
                                "split": dialogue["split"],
                                "qid": turn["qid"],
                                "turn": int(turn["turn"]),
                                "condition": condition,
                                "role": role,
                                "call_index": int(call_index),
                                "user_prompt": prompt,
                                "messages": messages,
                                "messages_sha256": messages_sha256(messages),
                                "context_chars": sum(
                                    len(message["content"]) for message in messages[:-1]
                                ),
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


def records_to_turn_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("role") != "primary":
            continue
        marker = strict_final_integer(str(record.get("raw_text", "")))
        rows.append(
            {
                "model": record["model"],
                "dialogue_id": record["dialogue_id"],
                "template": record["template"],
                "split": record["split"],
                "qid": record["qid"],
                "turn": int(record["turn"]),
                "condition": record["condition"],
                "CQ_raw": float(
                    answers_equal(
                        str(record.get("answer", "")),
                        str(record["gold_answer"]),
                        "dialogue_integer",
                    )
                ),
                "strict_final_marker": marker is not None,
                "stored_answer_agreement": marker is not None
                and answers_equal(
                    marker,
                    str(record.get("answer", "")),
                    "dialogue_integer",
                ),
                "context_chars": int(record["context_chars"]),
                "n_tokens": int(record["n_tokens"]),
                "latency_s": float(record["latency_s"]),
                "ok": bool(record["ok"]),
                "finish_reason": record.get("finish_reason"),
                "intrinsic_updates": int(record["intrinsic_updates"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["model", "dialogue_id", "condition", "turn"])
    frame["previous_cq"] = frame.groupby(
        ["model", "dialogue_id", "condition"]
    )["CQ_raw"].shift(1)
    return frame.reset_index(drop=True)
