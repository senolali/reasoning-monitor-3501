"""Call-free, checkpointed replay of DBQR Practice history branches."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from reasoning_monitor.dbqr_answers import typed_answer_equal
from reasoning_monitor.dbqr_audit import MAX_BYTES, SOURCE_SHA256, restricted_load
from reasoning_monitor.dbqr_json_plan import (
    PLAN_MARKER,
    execute_json_plan,
    parse_plan_response,
    translate_reference_plan,
)
from reasoning_monitor.dbqr_reference_interpreter import result_to_answer
from reasoning_monitor.dbqr_schema import (
    SCHEMA_VERSION,
    compact_schema,
    private_table_aliases,
)
from reasoning_monitor.dialogue_experiment import messages_sha256

FIXTURE_PROTOCOL = "DBQR-RQ-v1-fixture"
CONDITIONS = ("self_history", "oracle_answer_history")
SYSTEM_PROMPT = (
    "Solve the current DBQR table-planning question from the public compact schema. "
    "Write numbered reasoning lines and end with exactly one FINAL_PLAN_JSON line."
)


def _natural(text: str):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def load_practice_fixture_cases(blob: bytes, expected_sha256: str = SOURCE_SHA256):
    """Load Practice engineering cases; gold plans remain outside user prompts."""
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")
    cases = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        question_names = sorted(
            (name for name in members if "/questions/" in name), key=_natural)
        for question_name in question_names:
            question = json.load(archive.extractfile(members[question_name]))
            table_name = question_name.replace("/questions/", "/tables/").replace(
                ".json", ".pkl")
            private_tables, _globals = restricted_load(
                archive.extractfile(members[table_name]).read())
            schema, public_tables = compact_schema(private_tables)
            plan = translate_reference_plan(
                question["program"], private_table_aliases(private_tables))
            path = PurePosixPath(question_name)
            cases.append({
                "dialogue_id": path.parent.name,
                "turn": int(question["questionID"]),
                "qid": f"{path.parent.name}/{path.stem}",
                "question": str(question["question"]),
                "schema": schema,
                "tables": public_tables,
                "fixture_plan": plan,
                "gold_answer": question["answer"],
            })
    return sorted(cases, key=lambda case: (_natural(case["dialogue_id"]), case["turn"])), digest


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _fixture_response(plan: dict) -> str:
    return (
        "1. Apply the declared operations to the aliased query-result tables.\n"
        "2. Return the locally executed typed result.\n"
        f"{PLAN_MARKER} {_canonical(plan)}"
    )


def _oracle_response(answer) -> str:
    return (
        "1. Use the canonical prior answer for the history-control branch.\n"
        f"FINAL_ANSWER_JSON: {_canonical(answer)}"
    )


def _user_prompt(case: dict[str, Any]) -> str:
    return (
        f"QUESTION:\n{case['question']}\n\n"
        f"COMPACT_TABLE_SCHEMA_JSON:\n{_canonical(case['schema'])}"
    )


def _config_hash(source_sha256: str, conditions: Sequence[str], system_prompt: str) -> str:
    payload = {
        "protocol": FIXTURE_PROTOCOL,
        "source_sha256": source_sha256,
        "schema_version": SCHEMA_VERSION,
        "conditions": list(conditions),
        "system_prompt": system_prompt,
    }
    return hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest()


def _key(record: dict) -> tuple[str, str, int]:
    return str(record["dialogue_id"]), str(record["condition"]), int(record["turn"])


def _load_checkpoint(path: Path):
    if not path.exists():
        return [], False
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    truncated = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt checkpoint before final line") from None
            truncated = True
    return records, truncated


def _repair_truncated_tail(path: Path, records: Sequence[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".repair")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_fixture_replay(
    cases: Sequence[dict[str, Any]],
    *,
    source_sha256: str,
    checkpoint_path: str | Path,
    conditions: Sequence[str] = CONDITIONS,
    system_prompt: str = SYSTEM_PROMPT,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_after_new_records: int | None = None,
) -> list[dict[str, Any]]:
    """Replay both branches with deterministic fixtures and per-turn resume."""
    if tuple(conditions) != CONDITIONS:
        raise ValueError(f"Fixture conditions must be exactly {CONDITIONS}")
    if not cases:
        raise ValueError("Fixture replay requires cases")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    existing, truncated = _load_checkpoint(checkpoint)
    if truncated:
        _repair_truncated_tail(checkpoint, existing)
    config_hash = _config_hash(source_sha256, conditions, system_prompt)
    saved = {}
    for record in existing:
        if record.get("config_sha256") != config_hash or _key(record) in saved:
            raise ValueError("Mismatched config or duplicate checkpoint record")
        saved[_key(record)] = record

    expected = []
    new_records = 0
    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        dialogue_ids = list(dict.fromkeys(str(case["dialogue_id"]) for case in cases))
        for dialogue_id in dialogue_ids:
            dialogue = [case for case in cases if case["dialogue_id"] == dialogue_id]
            for condition in conditions:
                history: list[dict[str, str]] = []
                for case in dialogue:
                    user_prompt = _user_prompt(case)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        *history,
                        {"role": "user", "content": user_prompt},
                    ]
                    key = (dialogue_id, condition, int(case["turn"]))
                    record = saved.get(key)
                    response = _fixture_response(case["fixture_plan"])
                    if record is None:
                        reasoning, plan = parse_plan_response(response)
                        result, _env = execute_json_plan(plan, case["tables"])
                        answer = result_to_answer(result, case["gold_answer"])
                        record = {
                            "protocol_id": FIXTURE_PROTOCOL,
                            "config_sha256": config_hash,
                            "source_sha256": source_sha256,
                            "no_model_calls": True,
                            "dialogue_id": dialogue_id,
                            "turn": int(case["turn"]),
                            "qid": case["qid"],
                            "condition": condition,
                            "messages": messages,
                            "messages_sha256": messages_sha256(messages),
                            "response": response,
                            "reasoning_steps": reasoning,
                            "plan_sha256": hashlib.sha256(
                                _canonical(plan).encode("ascii")).hexdigest(),
                            "executed_answer": answer,
                            "gold_answer": case["gold_answer"],
                            "correct": typed_answer_equal(answer, case["gold_answer"]),
                        }
                        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                        handle.flush()
                        saved[key] = record
                        new_records += 1
                        if checkpoint_callback is not None:
                            checkpoint_callback(record)
                    else:
                        if (record.get("messages_sha256") != messages_sha256(messages)
                                or record.get("response") != response):
                            raise ValueError("Changed request or fixture on resume")
                    expected.append(record)
                    history.extend([
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": (
                            response if condition == "self_history"
                            else _oracle_response(case["gold_answer"]))},
                    ])
                    if (stop_after_new_records is not None
                            and new_records >= stop_after_new_records):
                        return expected
    return expected
