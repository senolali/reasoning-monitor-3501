"""Strict JSON plan envelope and execution for DBQR model outputs."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

import pandas as pd

from reasoning_monitor.dbqr_reference_interpreter import (
    MAX_OUTPUT_CELLS,
    MAX_STEPS,
    ReferencePlanInterpreter,
    _copy,
    _result_size,
    parse_statement,
)

PLAN_MARKER = "FINAL_PLAN_JSON:"
PLAN_VERSION = "DBQR-PLAN-v1"
MAX_PLAN_CHARS = 40_000
MAX_VALUE_DEPTH = 8
MAX_COLLECTION_ITEMS = 256
MAX_STRING_CHARS = 4_096
_ID = re.compile(r"s[1-9]\d{0,2}")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def parse_plan_response(text: str) -> tuple[list[str], dict]:
    """Parse numbered reasoning and one final-line JSON plan."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(PLAN_MARKER):
        raise ValueError("Missing final-line plan marker")
    payload = lines[-1][len(PLAN_MARKER):].strip()
    if not payload or len(payload) > MAX_PLAN_CHARS:
        raise ValueError("Invalid plan payload length")
    reasoning = lines[:-1]
    if not reasoning or any(re.fullmatch(r"[1-9]\d*\.\s+.+", line) is None
                            for line in reasoning):
        raise ValueError("Reasoning must use numbered non-empty lines")
    plan = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    return reasoning, plan


def _resolve(value, env: dict[str, Any], depth: int = 0):
    if depth > MAX_VALUE_DEPTH:
        raise ValueError("Plan value nesting exceeds the limit")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            raise ValueError("Plan string exceeds the limit")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite plan number")
        return value
    if isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ValueError("Plan collection exceeds the limit")
        return [_resolve(item, env, depth + 1) for item in value]
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            reference = value["ref"]
            if not isinstance(reference, str) or reference not in env:
                raise ValueError(f"Unknown or forward reference: {reference}")
            return _copy(env[reference])
        if len(value) > MAX_COLLECTION_ITEMS or any(not isinstance(key, str) for key in value):
            raise ValueError("Invalid plan object")
        return {key: _resolve(item, env, depth + 1) for key, item in value.items()}
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def execute_json_plan(
    plan: dict,
    tables: dict[str, pd.DataFrame],
    *,
    max_steps: int = MAX_STEPS,
    max_output_cells: int = MAX_OUTPUT_CELLS,
):
    if set(plan) != {"version", "steps"} or plan.get("version") != PLAN_VERSION:
        raise ValueError("Invalid plan envelope")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > max_steps:
        raise ValueError("Invalid plan steps")
    engine = ReferencePlanInterpreter(
        max_steps=max_steps, max_output_cells=max_output_cells)
    env: dict[str, Any] = {}
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Every plan step must be an object")
        step_id = step.get("id")
        operation = step.get("op")
        if not isinstance(step_id, str) or _ID.fullmatch(step_id) is None or step_id in env:
            raise ValueError("Invalid or duplicate step id")
        if operation == "query":
            if set(step) != {"id", "op", "table"}:
                raise ValueError("Invalid query step")
            alias = step.get("table")
            if not isinstance(alias, str) or alias not in tables:
                raise ValueError(f"Unknown table alias: {alias}")
            result = _copy(tables[alias])
        else:
            if set(step) != {"id", "op", "args", "kwargs"}:
                raise ValueError("Invalid operation step")
            if not isinstance(operation, str) or operation == "query":
                raise ValueError("Invalid operation")
            args = _resolve(step.get("args"), env)
            kwargs = _resolve(step.get("kwargs"), env)
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ValueError("Operation args/kwargs have invalid types")
            result = engine._dispatch(operation, args, kwargs)
        if _result_size(result) > max_output_cells:
            raise ValueError("Intermediate output exceeds cell limit")
        env[step_id] = result
    return env[step_id], env


def _ast_to_json(node: ast.AST, references: dict[str, str]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "null":
            return None
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id not in references:
            raise ValueError(f"Unknown reference while translating: {node.id}")
        return {"ref": references[node.id]}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_to_json(item, references) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            str(_ast_to_json(key, references)): _ast_to_json(value, references)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_to_json(node.operand, references)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    raise ValueError(f"Unsupported reference-plan value: {type(node).__name__}")


def translate_reference_plan(program: list[str], private_aliases: dict[str, str]) -> dict:
    """Translate pinned Practice plans to fixture JSON without query source text."""
    references = {}
    steps = []
    for number, statement in enumerate(program, start=1):
        target, call = parse_statement(statement)
        step_id = f"s{number}"
        operation = call.func.id
        if operation == "query":
            if target not in private_aliases:
                raise ValueError(f"Missing query table for fixture step: {target}")
            step = {"id": step_id, "op": "query", "table": private_aliases[target]}
        else:
            step = {
                "id": step_id,
                "op": operation,
                "args": [_ast_to_json(node, references) for node in call.args],
                "kwargs": {
                    keyword.arg: _ast_to_json(keyword.value, references)
                    for keyword in call.keywords
                },
            }
        steps.append(step)
        references[target] = step_id
    return {"version": PLAN_VERSION, "steps": steps}
