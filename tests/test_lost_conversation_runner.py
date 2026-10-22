from pathlib import Path

from reasoning_monitor.lost_conversation import MathTask
from reasoning_monitor.lost_conversation_runner import (
    acquire,
    expected_call_keys,
    parse_response,
    score_records,
)
from reasoning_monitor.models.clients import Sample

ROOT = Path(__file__).resolve().parents[1]


class FixtureClient:
    def generate_messages(self, messages, seed=0, gold=""):
        text = "1. I track the revealed facts.\n2. More information is required.\n\nSTATUS: NEED_MORE_INFORMATION"
        if "second shard" in messages[-1]["content"]:
            text = "1. Combine the two values.\n2. The result is 5.\n\nFINAL_ANSWER: 5"
        return Sample(answer="", reasoning_steps=("Step one", "Step two"), latency_s=0.1,
                      n_tokens=30, raw_text=text)


class FixtureScorers:
    def coherence(self, steps):
        return 1.0 if len(steps) >= 2 else float("nan")

    def similarity(self, left, right):
        return [1.0] * len(left)


def config():
    return {
        "protocol_id": "LIC-RQ-v2", "conditions": ["dialogue_history", "fresh_prefix"],
        "k_repeats": 5, "l_paraphrases": 2, "max_output_tokens": 2048,
        "models": [{"name": "fixture", "provider_model": "fixture/model"}],
        "prompt_contract": {
            "system_prompt": "System",
            "primary_instruction": "Primary contract",
            "paraphrase_instructions": ["Paraphrase one", "Paraphrase two"],
        },
    }


def test_strict_response_parser():
    assert parse_response("1. Work\nFINAL_ANSWER: $1,200.50") == {
        "kind": "answer_attempt", "answer": "1200.5", "contract_valid": True}
    assert parse_response("1. Missing\nSTATUS: NEED_MORE_INFORMATION")["kind"] == "need_more_information"
    assert parse_response("Answer: 4")["contract_valid"] is False


def test_resumable_acquisition_and_missingness(tmp_path):
    task = MathTask("task-1", "question", "5", ("first shard", "second shard"))
    tasks = {task.task_id: task}
    cfg = config()
    checkpoint = tmp_path / "records.jsonl"
    records = acquire(config=cfg, tasks=tasks, task_ids=[task.task_id],
                      clients={"fixture": FixtureClient()}, checkpoint=checkpoint,
                      config_hash="hash", callback=lambda _n: None)
    assert len(records) == 28
    assert len(expected_call_keys(cfg, [task.task_id], tasks)) == 28
    assert len(acquire(config=cfg, tasks=tasks, task_ids=[task.task_id],
                       clients={"fixture": FixtureClient()}, checkpoint=checkpoint,
                       config_hash="hash", callback=lambda _n: None)) == 28
    frame = score_records(records, tasks, FixtureScorers())
    early = frame[frame.turn == 1].iloc[0]
    final = frame[frame.turn == 2].iloc[0]
    assert early.CQ_status == "prefix_gold_ineligible"
    assert early.LS == 1.0 and early.SS == 1.0
    assert final.CQ == 1.0 and final.RS == 1.0
    assert final.ES_status == "insufficient_gold_history"
    assert frame.RQ6_mean.isna().all()
