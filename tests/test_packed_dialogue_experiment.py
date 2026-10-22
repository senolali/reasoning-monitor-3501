from __future__ import annotations

from reasoning_monitor.data.dialogue_streams import generate_dialogue_bank
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.packed_dialogue_experiment import run_packed_dialogue_model
from reasoning_monitor.types import Sample


class _GoldClient:
    def __init__(self):
        self.calls = []

    def generate_messages(self, messages, seed=0, gold=""):
        self.calls.append(messages)
        raw = f"Calculation complete.\nFINAL_ANSWER: {gold}"
        return Sample(
            answer=gold,
            reasoning_steps=reasoning_steps_from_text(raw),
            latency_s=0.1,
            n_tokens=8,
            raw_text=raw,
            finish_reason="stop",
        )


def test_packed_runner_uses_two_messages_and_resumes(tmp_path):
    client = _GoldClient()
    checkpoint = tmp_path / "records.jsonl"
    kwargs = {
        "bank": generate_dialogue_bank(),
        "client": client,
        "model_name": "test-model",
        "selected_dialogue_ids": ["cdr-v1-budget-001"],
        "conditions": ["self_history", "oracle_history", "state_summary"],
        "l_paraphrases": 0,
        "system_prompt": "Continue and give FINAL_ANSWER.",
        "checkpoint_path": checkpoint,
        "seed": 5,
    }

    records = run_packed_dialogue_model(**kwargs)

    assert len(records) == 36
    assert len(client.calls) == 36
    assert all(len(record["messages"]) == 2 for record in records)
    packed = next(
        record
        for record in records
        if record["condition"] == "self_history" and record["turn"] == 2
    )
    assert "CALLER-MANAGED PRIOR DIALOGUE TRANSCRIPT" in packed["messages"][0]["content"]

    resumed_client = _GoldClient()
    resumed = run_packed_dialogue_model(**{**kwargs, "client": resumed_client})
    assert len(resumed) == 36
    assert resumed_client.calls == []

