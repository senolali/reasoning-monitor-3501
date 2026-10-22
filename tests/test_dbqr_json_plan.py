import json

import pandas as pd
import pytest

from reasoning_monitor.dbqr_json_plan import (
    PLAN_MARKER,
    PLAN_VERSION,
    execute_json_plan,
    parse_plan_response,
)


def simple_plan():
    return {
        "version": PLAN_VERSION,
        "steps": [
            {"id": "s1", "op": "query", "table": "table_1"},
            {"id": "s2", "op": "sum", "args": [{"ref": "s1"}], "kwargs": {}},
        ],
    }


def test_plan_envelope_and_execution():
    raw = "1. Sum the supplied table.\n" + PLAN_MARKER + " " + json.dumps(simple_plan())
    reasoning, plan = parse_plan_response(raw)
    result, env = execute_json_plan(
        plan, {"table_1": pd.DataFrame({2022: [2, 3]})})
    assert reasoning == ["1. Sum the supplied table."]
    assert result.iloc[0] == 5
    assert set(env) == {"s1", "s2"}


@pytest.mark.parametrize("raw,match", [
    ("Reasoning\n" + PLAN_MARKER + ' {"version":"DBQR-PLAN-v1","steps":[]}',
     "numbered"),
    ("1. x\n" + PLAN_MARKER + ' {"version":"DBQR-PLAN-v1","version":"x","steps":[]}',
     "Duplicate"),
    ("1. x\n" + PLAN_MARKER + " {\"version\":NaN,\"steps\":[]}", "Non-finite"),
])
def test_plan_parser_fails_closed(raw, match):
    with pytest.raises(ValueError, match=match):
        parse_plan_response(raw)


def test_forward_reference_and_unknown_operation_are_rejected():
    plan = simple_plan()
    plan["steps"][1]["args"] = [{"ref": "s3"}]
    with pytest.raises(ValueError, match="forward reference"):
        execute_json_plan(plan, {"table_1": pd.DataFrame({"x": [1]})})
    plan = simple_plan()
    plan["steps"][1]["op"] = "python"
    with pytest.raises(ValueError, match="Unknown operation"):
        execute_json_plan(plan, {"table_1": pd.DataFrame({"x": [1]})})


def test_extra_query_fields_are_rejected():
    plan = simple_plan()
    plan["steps"][0]["code"] = "import os"
    with pytest.raises(ValueError, match="Invalid query step"):
        execute_json_plan(plan, {"table_1": pd.DataFrame({"x": [1]})})
