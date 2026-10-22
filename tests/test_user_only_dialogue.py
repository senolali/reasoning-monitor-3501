from __future__ import annotations

from reasoning_monitor.data.dialogue_streams import generate_dialogue_bank
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.types import Sample
from reasoning_monitor.user_only_dialogue_experiment import (
    build_user_only_messages,
    run_user_only_dialogue_model,
)


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


def test_user_only_messages_omit_assistant_content():
    messages = build_user_only_messages(
        current_user_prompt="Apply update two.",
        prior_user_turns=["Initial value is ten."],
        system_prompt="Track the case.",
    )

    assert len(messages) == 2
    assert "[PRIOR USER TURN 1]" in messages[0]["content"]
    assert "Initial value is ten." in messages[0]["content"]
    assert "PRIOR ASSISTANT" not in messages[0]["content"]


def test_user_only_runner_resumes(tmp_path):
    checkpoint = tmp_path / "records.jsonl"
    client = _GoldClient()
    kwargs = {
        "bank": generate_dialogue_bank(),
        "client": client,
        "model_name": "test-model",
        "selected_dialogue_ids": ["cdr-v1-budget-001"],
        "system_prompt": "Track the case.",
        "checkpoint_path": checkpoint,
        "seed": 3,
    }

    records = run_user_only_dialogue_model(**kwargs)

    assert len(records) == 12
    assert len(client.calls) == 12
    assert all(len(record["messages"]) == 2 for record in records)
    assert all(record["condition"] == "user_only_history" for record in records)
    resumed_client = _GoldClient()
    resumed = run_user_only_dialogue_model(**{**kwargs, "client": resumed_client})
    assert len(resumed) == 12
    assert resumed_client.calls == []

