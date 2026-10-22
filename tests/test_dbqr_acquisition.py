import json

import pandas as pd
import pytest

from reasoning_monitor.dbqr_acquisition import CausalOracleBroker, run_acquisition
from reasoning_monitor.dbqr_json_plan import PLAN_MARKER, PLAN_VERSION
from reasoning_monitor.types import Sample


class _PlanClient:
    def __init__(self):
        self.calls = 0

    def generate_messages(self, messages, seed=0, gold=""):
        assert gold == ""
        self.calls += 1
        plan = {
            "version": PLAN_VERSION,
            "steps": [
                {"id": "s1", "op": "query", "table": "table_1"},
                {"id": "s2", "op": "sum", "args": [{"ref": "s1"}], "kwargs": {}},
            ],
        }
        raw = "1. Sum the table.\n" + PLAN_MARKER + " " + json.dumps(plan)
        return Sample(answer="", reasoning_steps=(), latency_s=0.0, n_tokens=20,
                      raw_text=raw, finish_reason="stop")


def _cases():
    return [{
        "dialogue_id": "chat-1",
        "turn": turn,
        "qid": f"chat-1/q{turn}",
        "question": f"Question {turn}?",
        "schema": {"schema_version": "test", "tables": []},
        "table_member": f"q{turn}.pkl",
    } for turn in (1, 2)]


def _config():
    return {
        "models": [{"name": "model"}],
        "conditions": ["self_history", "oracle_answer_history"],
        "k_repeats": 2,
        "l_framing_variants": 1,
        "framing_variants": ["Preserve the exact request."],
        "max_input_characters_per_call": 10_000,
    }


def test_causal_broker_rejects_early_or_out_of_order_release():
    broker = CausalOracleBroker({"q1": 1, "q2": 2}, ["q1", "q2"])
    with pytest.raises(ValueError, match="causal order"):
        broker.release_completed_turn("q2")
    assert broker.release_completed_turn("q1") == 1
    with pytest.raises(ValueError, match="causal order"):
        broker.release_completed_turn("q1")


def test_acquisition_stages_resume_and_never_pass_gold_to_client(tmp_path):
    cases = _cases()
    tables = {
        "chat-1/q1": {"table_1": pd.DataFrame({"x": [10]})},
        "chat-1/q2": {"table_1": pd.DataFrame({"x": [20]})},
    }
    answers = {"chat-1/q1": 10, "chat-1/q2": 20}
    factory = lambda order: CausalOracleBroker(answers, order)
    checkpoint = tmp_path / "records.jsonl"
    first_client = _PlanClient()
    first = run_acquisition(
        cases=cases,
        table_loader=lambda case: tables[case["qid"]],
        oracle_broker_factory=factory,
        clients={"model": first_client},
        config=_config(),
        checkpoint_path=checkpoint,
        config_sha256="config",
        selected_dialogue_ids=["chat-1"],
        max_turn=1,
        fixture_mode=True,
    )
    assert len(first) == 6
    assert first_client.calls == 6
    second_client = _PlanClient()
    complete = run_acquisition(
        cases=cases,
        table_loader=lambda case: tables[case["qid"]],
        oracle_broker_factory=factory,
        clients={"model": second_client},
        config=_config(),
        checkpoint_path=checkpoint,
        config_sha256="config",
        selected_dialogue_ids=["chat-1"],
        fixture_mode=True,
    )
    assert len(complete) == 12
    assert second_client.calls == 6
    final_client = _PlanClient()
    assert run_acquisition(
        cases=cases,
        table_loader=lambda case: tables[case["qid"]],
        oracle_broker_factory=factory,
        clients={"model": final_client},
        config=_config(),
        checkpoint_path=checkpoint,
        config_sha256="config",
        selected_dialogue_ids=["chat-1"],
        fixture_mode=True,
    ) == complete
    assert final_client.calls == 0
    assert all("gold_answer" not in row and row["executed_result"] for row in complete)
    oracle_turn_two = next(row for row in complete if row["turn"] == 2
                           and row["condition"] == "oracle_answer_history")
    assert "FINAL_ANSWER_JSON: 10" in oracle_turn_two["messages"][2]["content"]
