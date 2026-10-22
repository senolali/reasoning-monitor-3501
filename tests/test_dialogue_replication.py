from __future__ import annotations

from reasoning_monitor.dialogue_replication import (
    generate_replication_bank,
    run_replication_model,
    validate_replication_bank,
)
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.types import Sample


class _GoldClient:
    def __init__(self):
        self.calls = 0

    def generate_messages(self, messages, seed=0, gold=""):
        self.calls += 1
        raw = f"FINAL_ANSWER: {gold}"
        return Sample(
            answer=gold,
            reasoning_steps=reasoning_steps_from_text(raw),
            latency_s=0.1,
            n_tokens=5,
            raw_text=raw,
            finish_reason="stop",
        )


def test_replication_bank_is_fresh_deterministic_and_valid():
    first = generate_replication_bank()
    second = generate_replication_bank()

    assert first == second
    assert validate_replication_bank(first) == []
    assert len(first["dialogues"]) == 9
    assert all(
        dialogue["dialogue_id"].startswith("cdr-v14-")
        and dialogue["split"] == "replication_calibration"
        for dialogue in first["dialogues"]
    )


def test_replication_runner_reports_new_checkpoints_and_resumes(tmp_path):
    bank = generate_replication_bank()
    dialogue_id = bank["dialogues"][0]["dialogue_id"]
    checkpoint = tmp_path / "records.jsonl"
    client = _GoldClient()
    callbacks = []
    kwargs = {
        "bank": bank,
        "client": client,
        "model_name": "test-model",
        "selected_dialogue_ids": [dialogue_id],
        "conditions": ["user_only_history", "state_summary"],
        "system_prompt": "Track the case.",
        "checkpoint_path": checkpoint,
        "seed": 7,
        "checkpoint_callback": callbacks.append,
    }

    records = run_replication_model(**kwargs)

    assert len(records) == 24
    assert client.calls == 24
    assert len(callbacks) == 24
    resumed_client = _GoldClient()
    resumed_callbacks = []
    resumed = run_replication_model(
        **{
            **kwargs,
            "client": resumed_client,
            "checkpoint_callback": resumed_callbacks.append,
        }
    )
    assert len(resumed) == 24
    assert resumed_client.calls == 0
    assert resumed_callbacks == []
