from reasoning_monitor.lost_conversation import MathTask
from reasoning_monitor.lost_conversation_science_acquisition import (
    EMPTY_HISTORY_PLACEHOLDER,
    acquire_science,
)
from reasoning_monitor.models.clients import Sample


class EmptyPrimaryClient:
    def __init__(self):
        self.messages = []

    def generate_messages(self, messages, seed=0, gold=""):
        self.messages.append(messages)
        primary_call = len(self.messages) in {1, 8}
        text = "" if primary_call else "STATUS: NEED_MORE_INFORMATION"
        return Sample(answer="", reasoning_steps=(), latency_s=0.1,
                      n_tokens=2048 if not text else 10, raw_text=text,
                      finish_reason="stop")


def test_empty_primary_is_preserved_and_transport_uses_placeholder(tmp_path):
    task = MathTask("task", "question", "5", ("first", "second"))
    config = {
        "protocol_id": "LIC-RQ-v2",
        "models": [{"name": "model", "provider_model": "provider"}],
        "conditions": ["dialogue_history"],
        "k_repeats": 5,
        "l_paraphrases": 2,
        "max_output_tokens": 2048,
        "prompt_contract": {
            "system_prompt": "system",
            "primary_instruction": "primary",
            "paraphrase_instructions": ["p1", "p2"],
        },
    }
    client = EmptyPrimaryClient()
    checkpoint = tmp_path / "records.jsonl"
    records = acquire_science(
        config=config, tasks={"task": task}, task_ids=["task"],
        clients={"model": client}, checkpoint=checkpoint,
        config_hash="stable", callback=lambda _n: None)
    primaries = [row for row in records if row["role"] == "primary"]
    assert primaries[0]["raw_text"] == ""
    second_turn_primary_messages = client.messages[7]
    assert any(EMPTY_HISTORY_PLACEHOLDER in message.get("content", "")
               for message in second_turn_primary_messages)
    assert len(records) == 14
