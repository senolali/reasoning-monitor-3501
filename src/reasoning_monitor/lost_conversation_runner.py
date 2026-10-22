"""Resumable LIC-RQ-v2 acquisition, strict parsing and missingness-aware scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_monitor.dialogue_experiment import messages_sha256
from reasoning_monitor.lost_conversation import MathTask, build_prefix_messages
from reasoning_monitor.metrics.dimensions import answers_equal
from reasoning_monitor.models.clients import reasoning_steps_from_text

METRICS = ("CQ", "CS", "RS", "LS", "ES", "SS")
ROLES = ("primary", "repeat", "paraphrase")
FINAL_RE = re.compile(r"^FINAL_ANSWER:\s*(.+?)\s*$")
STATUS_LINE = "STATUS: NEED_MORE_INFORMATION"


def parse_response(text: str) -> dict:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    final = lines[-1]
    match = FINAL_RE.fullmatch(final)
    if match:
        raw = match.group(1).strip().replace(",", "")
        raw = raw.removeprefix("$").strip()
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return {"kind": "invalid", "answer": "", "contract_valid": False}
        if not value.is_finite():
            return {"kind": "invalid", "answer": "", "contract_valid": False}
        answer = format(value.normalize(), "f")
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        return {"kind": "answer_attempt", "answer": answer, "contract_valid": True}
    if final == STATUS_LINE:
        return {"kind": "need_more_information", "answer": "", "contract_valid": True}
    return {"kind": "invalid", "answer": "", "contract_valid": False}


def call_key(row: Mapping) -> tuple:
    return tuple(row[key] for key in (
        "model", "task_id", "condition", "turn", "role", "call_index"
    ))


def expected_call_keys(
    config: Mapping, task_ids: Sequence[str], tasks: Mapping[str, MathTask]
) -> set[tuple]:
    keys = set()
    for model in config["models"]:
        for task_id in task_ids:
            for condition in config["conditions"]:
                for turn in range(1, len(tasks[task_id].shards) + 1):
                    for index in range(config["k_repeats"]):
                        role = "primary" if index == 0 else "repeat"
                        keys.add((model["name"], task_id, condition, turn, role, index))
                    for index in range(config["l_paraphrases"]):
                        keys.add((model["name"], task_id, condition, turn, "paraphrase", index))
    return keys


def _messages(task: MathTask, turn: int, condition: str, history: Sequence[str],
              system_prompt: str, instruction: str) -> list[dict[str, str]]:
    messages = build_prefix_messages(
        task, turn, condition, primary_history=history, system_prompt=system_prompt)
    messages[-1]["content"] += "\n\nRESPONSE CONTRACT:\n" + instruction
    return messages


def _load_checkpoint(path: Path, config_hash: str) -> dict[tuple, dict]:
    saved = {}
    if not path.exists():
        return saved
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = call_key(row)
        if row.get("config_hash") != config_hash or key in saved:
            raise ValueError(f"Invalid checkpoint at line {line_number}")
        saved[key] = row
    return saved


def acquire(
    *,
    config: Mapping,
    tasks: Mapping[str, MathTask],
    task_ids: Sequence[str],
    clients: Mapping[str, object],
    checkpoint: Path,
    config_hash: str,
    callback: Callable[[int], None],
) -> list[dict]:
    expected = expected_call_keys(config, task_ids, tasks)
    saved = _load_checkpoint(checkpoint, config_hash)
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen integration sample")
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
                        requests = [("primary" if i == 0 else "repeat", i, primary_instruction)
                                    for i in range(config["k_repeats"])]
                        requests += [("paraphrase", i, text)
                                     for i, text in enumerate(paraphrases)]
                        primary_text = None
                        for role, index, instruction in requests:
                            key = (model["name"], task_id, condition, turn, role, index)
                            messages = _messages(
                                task, turn, condition, history, contract["system_prompt"], instruction)
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
                                    "model": model["name"], "provider_model": model["provider_model"],
                                    "task_id": task_id, "condition": condition, "turn": turn,
                                    "total_turns": len(task.shards), "role": role, "call_index": index,
                                    "all_information_available": turn == len(task.shards),
                                    "messages": messages, "messages_sha256": request_hash,
                                    "config_hash": config_hash, "call_seed": seed,
                                    "raw_text": sample.raw_text, "answer": parsed["answer"],
                                    "response_kind": parsed["kind"],
                                    "contract_valid": parsed["contract_valid"],
                                    "reasoning_steps": list(sample.reasoning_steps),
                                    "ok": bool(sample.ok), "n_tokens": int(sample.n_tokens),
                                    "max_tokens": int(config["max_output_tokens"]),
                                    "latency_s": sample.latency_s,
                                    "finish_reason": sample.finish_reason,
                                }
                                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                                stream.flush()
                                saved[key] = row
                                callback(len(saved))
                            if role == "primary":
                                primary_text = row["raw_text"]
                        if primary_text is None:
                            raise ValueError("Missing primary response")
                        if condition == "dialogue_history":
                            history.append(primary_text)
    if set(saved) != expected:
        raise ValueError("Incomplete acquisition")
    return [saved[key] for key in sorted(saved)]


def _valid(row: Mapping) -> bool:
    return bool(row.get("ok") and row.get("contract_valid")
                and row.get("finish_reason") != "length"
                and int(row.get("n_tokens", 0)) < int(row.get("max_tokens", 2048)))


def score_records(records: Sequence[Mapping], tasks: Mapping[str, MathTask], scorers,
                  *, k: int = 5, n_paraphrases: int = 2) -> pd.DataFrame:
    grouped: dict[tuple, list[Mapping]] = {}
    for row in records:
        key = (row["model"], row["task_id"], row["condition"], row["turn"])
        grouped.setdefault(key, []).append(row)
    result_rows = []
    for key, cell in sorted(grouped.items()):
        model, task_id, condition, turn = key
        task = tasks[task_id]
        primary = next(row for row in cell if row["role"] == "primary")
        runs = sorted((row for row in cell if row["role"] in {"primary", "repeat"}),
                      key=lambda row: row["call_index"])
        paras = sorted((row for row in cell if row["role"] == "paraphrase"),
                       key=lambda row: row["call_index"])
        values = {metric: math.nan for metric in METRICS}
        status = {metric: "missing_samples" for metric in METRICS}
        full = turn == len(task.shards)
        primary_valid = _valid(primary)
        if primary_valid:
            steps = reasoning_steps_from_text(primary["raw_text"])
            values["LS"] = scorers.coherence(steps)
            status["LS"] = "measured" if np.isfinite(values["LS"]) else "insufficient_steps"
        if len(runs) == k and all(_valid(row) for row in runs):
            if all(row["response_kind"] == "answer_attempt" for row in runs):
                pairs = list(combinations(runs, 2))
                values["CS"] = sum(answers_equal(a["answer"], b["answer"], "dialogue_integer")
                                   for a, b in pairs) / len(pairs)
                status["CS"] = "measured"
            else:
                status["CS"] = "non_answer_response"
            traces = [" ".join(reasoning_steps_from_text(row["raw_text"])) for row in runs]
            if all(traces):
                pairs = list(combinations(traces, 2))
                values["SS"] = float(np.mean(scorers.similarity(
                    [a for a, _ in pairs], [b for _, b in pairs])))
                status["SS"] = "measured"
            else:
                status["SS"] = "insufficient_steps"
        if not full:
            status["CQ"] = "prefix_gold_ineligible"
            status["RS"] = "prefix_gold_ineligible"
            status["ES"] = "prefix_gold_ineligible"
        elif primary_valid and primary["response_kind"] == "answer_attempt":
            cq = float(answers_equal(primary["answer"], task.gold_answer, "dialogue_integer"))
            values["CQ"], status["CQ"] = cq, "measured"
            status["ES"] = "insufficient_gold_history"
            if cq == 0:
                status["RS"] = "C_subset_excluded"
            elif len(paras) == n_paraphrases and all(
                    _valid(row) and row["response_kind"] == "answer_attempt" for row in paras):
                values["RS"] = sum(answers_equal(row["answer"], task.gold_answer,
                                                  "dialogue_integer") for row in paras
                                   ) / n_paraphrases
                status["RS"] = "measured"
        else:
            status["CQ"] = "invalid_final_answer"
            status["RS"] = "invalid_final_answer"
            status["ES"] = "insufficient_gold_history"
        measured = [values[m] for m in METRICS
                    if status[m] == "measured" and np.isfinite(values[m])]
        rq6 = float(sum(measured) / 6) if len(measured) == 6 else math.nan
        result_rows.append({
            "model": model, "task_id": task_id, "condition": condition,
            "turn": turn, "total_turns": len(task.shards),
            "information_fraction": turn / len(task.shards),
            **values, **{f"{metric}_status": status[metric] for metric in METRICS},
            "RQ6_mean": rq6,
            "RQ6_mean_status": "measured" if np.isfinite(rq6) else "incomplete_case",
        })
    return pd.DataFrame(result_rows)


def operational_report(records: Sequence[Mapping], expected_calls: int) -> dict:
    counts = Counter(row["model"] for row in records)
    return {
        "records": len(records),
        "expected_records": expected_calls,
        "complete": len(records) == expected_calls,
        "records_by_model": dict(sorted(counts.items())),
        "success_rate": sum(bool(row.get("ok")) for row in records) / len(records),
        "contract_valid_rate": sum(bool(row.get("contract_valid")) for row in records) / len(records),
        "truncations": sum(row.get("finish_reason") == "length" for row in records),
        "distinct_raw_responses": len({row.get("raw_text") for row in records}),
    }
