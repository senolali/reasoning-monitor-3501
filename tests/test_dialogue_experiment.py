from __future__ import annotations

from reasoning_monitor.data.dialogue_streams import generate_dialogue_bank
from reasoning_monitor.dialogue_experiment import (
    records_to_turn_frame,
    run_dialogue_model,
)
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.types import Sample


class _GoldClient:
    def __init__(self):
        self.calls = []

    def generate_messages(self, messages, seed=0, gold=""):
        self.calls.append(messages)
        raw = f"The verified arithmetic gives {gold}.\nFINAL_ANSWER: {gold}"
        return Sample(
            answer=gold,
            reasoning_steps=reasoning_steps_from_text(raw),
            latency_s=0.1,
            n_tokens=8,
            raw_text=raw,
            finish_reason="stop",
        )


def test_dialogue_runner_isolates_history_conditions_and_resumes(tmp_path):
    bank = generate_dialogue_bank()
    dialogue_id = "cdr-v1-budget-001"
    checkpoint = tmp_path / "records.jsonl"
    client = _GoldClient()
    kwargs = {
        "bank": bank,
        "client": client,
        "model_name": "test-model",
        "selected_dialogue_ids": [dialogue_id],
        "conditions": ["self_history", "oracle_history", "state_summary"],
        "l_paraphrases": 0,
        "system_prompt": "Continue the case and end with FINAL_ANSWER.",
        "checkpoint_path": checkpoint,
        "seed": 19,
    }

    records = run_dialogue_model(**kwargs)

    assert len(records) == 36
    assert len(client.calls) == 36
    self_turn_two = next(
        record
        for record in records
        if record["condition"] == "self_history" and record["turn"] == 2
    )
    oracle_turn_two = next(
        record
        for record in records
        if record["condition"] == "oracle_history" and record["turn"] == 2
    )
    summary_turn_two = next(
        record
        for record in records
        if record["condition"] == "state_summary" and record["turn"] == 2
    )
    assert [message["role"] for message in self_turn_two["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "verified arithmetic" in self_turn_two["messages"][2]["content"].lower()
    assert "Using the verified running state" in oracle_turn_two["messages"][2]["content"]
    assert len(summary_turn_two["messages"]) == 2
    assert "VERIFIED STATE BEFORE THIS TURN" in summary_turn_two["messages"][0]["content"]

    frame = records_to_turn_frame(records)
    assert len(frame) == 36
    assert frame["CQ_raw"].eq(1.0).all()
    assert frame["strict_final_marker"].all()

    resumed_client = _GoldClient()
    resumed = run_dialogue_model(**{**kwargs, "client": resumed_client})
    assert len(resumed) == 36
    assert resumed_client.calls == []
