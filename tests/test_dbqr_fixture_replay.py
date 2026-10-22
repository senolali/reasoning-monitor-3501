import json

import pandas as pd

from reasoning_monitor.dbqr_fixture_replay import run_fixture_replay
from reasoning_monitor.dbqr_json_plan import PLAN_VERSION


def _case(turn, value):
    return {
        "dialogue_id": "chat-1",
        "turn": turn,
        "qid": f"chat-1/q{turn}",
        "question": f"What is value {turn}?",
        "schema": {"schema_version": "test", "tables": []},
        "tables": {"table_1": pd.DataFrame({"x": [value]})},
        "fixture_plan": {
            "version": PLAN_VERSION,
            "steps": [
                {"id": "s1", "op": "query", "table": "table_1"},
                {"id": "s2", "op": "sum", "args": [{"ref": "s1"}], "kwargs": {}},
            ],
        },
        "gold_answer": value,
    }


def test_fixture_replay_branches_and_resumes(tmp_path):
    cases = [_case(1, 10), _case(2, 20)]
    checkpoint = tmp_path / "records.jsonl"
    first = run_fixture_replay(
        cases,
        source_sha256="fixture",
        checkpoint_path=checkpoint,
        stop_after_new_records=1,
    )
    assert len(first) == 1
    callbacks = []
    complete = run_fixture_replay(
        cases,
        source_sha256="fixture",
        checkpoint_path=checkpoint,
        checkpoint_callback=callbacks.append,
    )
    assert len(complete) == 4
    assert len(callbacks) == 3
    resumed_callbacks = []
    resumed = run_fixture_replay(
        cases,
        source_sha256="fixture",
        checkpoint_path=checkpoint,
        checkpoint_callback=resumed_callbacks.append,
    )
    assert resumed == complete
    assert not resumed_callbacks
    self_second = next(
        row for row in resumed if row["condition"] == "self_history" and row["turn"] == 2)
    oracle_second = next(
        row for row in resumed
        if row["condition"] == "oracle_answer_history" and row["turn"] == 2)
    assert "FINAL_PLAN_JSON:" in self_second["messages"][2]["content"]
    assert "FINAL_ANSWER_JSON:" in oracle_second["messages"][2]["content"]
    assert all(json.loads(line)["no_model_calls"] for line in checkpoint.read_text().splitlines())
