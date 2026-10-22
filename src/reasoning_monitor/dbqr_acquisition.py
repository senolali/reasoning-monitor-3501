"""Checkpointed, causally label-gated DBQR acquisition runner."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.dbqr_campaign_contract import framed_prompts
from reasoning_monitor.dbqr_json_plan import execute_json_plan, parse_plan_response
from reasoning_monitor.dbqr_prompt_contract import SYSTEM_PROMPT
from reasoning_monitor.dbqr_reference_interpreter import serialize_plan_result
from reasoning_monitor.dialogue_experiment import messages_sha256


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def load_oracle_vault(path: str | Path, *, expected_source_sha256: str) -> dict[str, Any]:
    vault = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(vault) != {
        "protocol_id", "split", "source_sha256", "answers", "vault_sha256",
    } or vault.get("protocol_id") != "DBQR-RQ-v1" or vault.get("split") != "practice":
        raise ValueError("Invalid oracle vault envelope")
    if vault.get("source_sha256") != expected_source_sha256:
        raise ValueError("Oracle vault source mismatch")
    unsigned = dict(vault)
    expected = unsigned.pop("vault_sha256")
    actual = hashlib.sha256(_canonical(unsigned).encode("ascii")).hexdigest()
    if actual != expected or not isinstance(vault.get("answers"), dict):
        raise ValueError("Oracle vault hash mismatch")
    return vault["answers"]


class CausalOracleBroker:
    """Release one canonical answer only after its turn's acquisition completes."""

    def __init__(self, answers: dict[str, Any], ordered_qids: Sequence[str]):
        if set(answers) != set(ordered_qids) or len(ordered_qids) != len(set(ordered_qids)):
            raise ValueError("Oracle vault and ordered qids differ")
        self.__answers = dict(answers)
        self.__order = list(ordered_qids)
        self.__position = 0

    @property
    def released_count(self) -> int:
        return self.__position

    def release_completed_turn(self, qid: str):
        if self.__position >= len(self.__order) or self.__order[self.__position] != qid:
            raise ValueError("Oracle answers must be released once in causal order")
        self.__position += 1
        return self.__answers[qid]


def _record_key(record: dict) -> tuple[str, str, str, int, str, int]:
    return (
        str(record["model"]),
        str(record["dialogue_id"]),
        str(record["condition"]),
        int(record["turn"]),
        str(record["role"]),
        int(record["call_index"]),
    )


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt acquisition checkpoint") from None
    return records


def _oracle_response(answer) -> str:
    return (
        "1. Use the canonical prior answer for the oracle-history control.\n"
        f"FINAL_ANSWER_JSON: {_canonical(answer)}"
    )


def _user_prompt(case: dict) -> str:
    return (
        f"QUESTION:\n{case['question']}\n\n"
        f"COMPACT_TABLE_SCHEMA_JSON:\n{_canonical(case['schema'])}"
    )


def run_acquisition(
    *,
    cases: Sequence[dict],
    table_loader: Callable[[dict], dict],
    oracle_broker_factory: Callable[[Sequence[str]], CausalOracleBroker],
    clients: dict[str, Any],
    config: dict,
    checkpoint_path: str | Path,
    config_sha256: str,
    selected_dialogue_ids: Sequence[str],
    max_turn: int | None = None,
    fixture_mode: bool = False,
    checkpoint_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Acquire K repeats and L framing variants without exposing current gold."""
    if config.get("conditions") != ["self_history", "oracle_answer_history"]:
        raise ValueError("Unexpected history conditions")
    model_names = [model["name"] for model in config.get("models", [])]
    if set(model_names) != set(clients):
        raise ValueError("Configured models and clients differ")
    selected = [case for case in cases if case["dialogue_id"] in selected_dialogue_ids]
    selected.sort(key=lambda row: (selected_dialogue_ids.index(row["dialogue_id"]), row["turn"]))
    if not selected:
        raise ValueError("No selected acquisition cases")
    ordered_qids = [case["qid"] for case in selected]
    broker = oracle_broker_factory(ordered_qids)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    for record in _load_checkpoint(checkpoint):
        key = _record_key(record)
        if record.get("config_sha256") != config_sha256 or key in saved:
            raise ValueError("Mismatched or duplicate acquisition checkpoint")
        saved[key] = record

    expected = []
    histories = {
        (model, dialogue, condition): []
        for model in model_names for dialogue in selected_dialogue_ids
        for condition in config["conditions"]
    }
    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for case_offset, case in enumerate(selected):
            if max_turn is not None and int(case["turn"]) > max_turn:
                return expected
            tables = table_loader(case)
            base_prompt = _user_prompt(case)
            variants = framed_prompts(base_prompt, config)
            for model_offset, model in enumerate(model_names):
                for condition_offset, condition in enumerate(config["conditions"]):
                    history = histories[(model, case["dialogue_id"], condition)]
                    prompts = [
                        ("repeat", index, base_prompt)
                        for index in range(int(config["k_repeats"]))
                    ] + [
                        ("framing", index, prompt) for index, prompt in enumerate(variants)
                    ]
                    primary_response = None
                    for role, call_index, prompt in prompts:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            *history,
                            {"role": "user", "content": prompt},
                        ]
                        input_characters = sum(len(message["content"]) for message in messages)
                        if input_characters > int(config["max_input_characters_per_call"]):
                            raise ValueError("Input character ceiling exceeded")
                        key = (model, case["dialogue_id"], condition, int(case["turn"]),
                               role, int(call_index))
                        record = saved.get(key)
                        if record is None:
                            seed = (20261001 + case_offset * 10_000 + model_offset * 1_000
                                    + condition_offset * 100 + call_index)
                            sample = clients[model].generate_messages(messages, seed=seed)
                            answer = None
                            plan_error = None
                            reasoning = []
                            if sample.ok:
                                try:
                                    reasoning, plan = parse_plan_response(sample.raw_text)
                                    result, _env = execute_json_plan(plan, tables)
                                    answer = serialize_plan_result(result)
                                except Exception as exc:
                                    plan_error = f"{type(exc).__name__}: {exc}"
                            record = {
                                "protocol_id": "DBQR-RQ-v1",
                                "config_sha256": config_sha256,
                                "fixture_mode": bool(fixture_mode),
                                "model": model,
                                "dialogue_id": case["dialogue_id"],
                                "turn": int(case["turn"]),
                                "qid": case["qid"],
                                "condition": condition,
                                "role": role,
                                "call_index": int(call_index),
                                "messages": messages,
                                "messages_sha256": messages_sha256(messages),
                                "input_characters": input_characters,
                                "raw_text": sample.raw_text,
                                "reasoning_steps": reasoning,
                                "executed_result": answer,
                                "plan_error": plan_error,
                                "ok": bool(sample.ok and plan_error is None),
                                "n_tokens": int(sample.n_tokens),
                                "latency_s": float(sample.latency_s),
                                "finish_reason": sample.finish_reason,
                            }
                            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                            handle.flush()
                            saved[key] = record
                            if checkpoint_callback is not None:
                                checkpoint_callback(record)
                        elif record.get("messages_sha256") != messages_sha256(messages):
                            raise ValueError("Changed messages on acquisition resume")
                        expected.append(record)
                        if role == "repeat" and call_index == 0:
                            primary_response = record["raw_text"]
                    if primary_response is None:
                        raise RuntimeError("Missing primary response")
                    if condition == "self_history":
                        history.extend([
                            {"role": "user", "content": base_prompt},
                            {"role": "assistant", "content": primary_response},
                        ])
            oracle_answer = broker.release_completed_turn(case["qid"])
            for model in model_names:
                history = histories[(model, case["dialogue_id"], "oracle_answer_history")]
                history.extend([
                    {"role": "user", "content": base_prompt},
                    {"role": "assistant", "content": _oracle_response(oracle_answer)},
                ])
    return expected
