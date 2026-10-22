from __future__ import annotations

from reasoning_monitor.data.dialogue_streams import generate_dialogue_bank
from reasoning_monitor.evaluation.primary_dialogue_validation import (
    assess_primary_validation,
)
from reasoning_monitor.models.clients import reasoning_steps_from_text
from reasoning_monitor.primary_dialogue_validation import (
    build_primary_validation_messages,
    run_primary_dialogue_validation_model,
)
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


def _record(model: str, dialogue: str, condition: str, turn: int) -> dict:
    user_correct = turn not in {7, 8}
    correct = condition == "state_summary" or user_correct
    answer = "10" if correct else "11"
    system = (
        "VERIFIED STATE BEFORE THIS TURN"
        if condition == "state_summary"
        else (
            "CALLER-MANAGED PRIOR USER HISTORY" if turn > 1 else "base context"
        )
    )
    return {
        "protocol_id": "CDR-v1.3",
        "history_transport": (
            "verified_state_summary"
            if condition == "state_summary"
            else "packed_user_only_context"
        ),
        "model": model,
        "dialogue_id": dialogue,
        "template": "test",
        "split": "calibration",
        "qid": f"{dialogue}-T{turn:02d}",
        "turn": turn,
        "condition": condition,
        "role": "primary",
        "call_index": 0,
        "gold_answer": "10",
        "answer": answer,
        "raw_text": f"FINAL_ANSWER: {answer}",
        "reasoning_steps": [],
        "context_chars": 100,
        "n_tokens": 20,
        "latency_s": 0.1,
        "ok": True,
        "finish_reason": "stop",
        "intrinsic_updates": 1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "question"},
        ],
    }


def test_primary_validation_messages_keep_conditions_separate():
    dialogue = generate_dialogue_bank()["dialogues"][0]
    turn = dialogue["turns"][1]
    prior = [dialogue["turns"][0]["user_prompt"]]
    user_messages = build_primary_validation_messages(
        turn=turn,
        condition="user_only_history",
        prior_user_turns=prior,
        system_prompt="Track the case.",
    )
    summary_messages = build_primary_validation_messages(
        turn=turn,
        condition="state_summary",
        prior_user_turns=prior,
        system_prompt="Track the case.",
    )

    assert prior[0] in user_messages[0]["content"]
    assert "PRIOR ASSISTANT" not in user_messages[0]["content"]
    assert "VERIFIED STATE BEFORE THIS TURN" in summary_messages[0]["content"]
    assert prior[0] not in summary_messages[0]["content"]


def test_primary_validation_runner_resumes(tmp_path):
    checkpoint = tmp_path / "records.jsonl"
    client = _GoldClient()
    kwargs = {
        "bank": generate_dialogue_bank(),
        "client": client,
        "model_name": "test-model",
        "selected_dialogue_ids": ["cdr-v1-budget-002"],
        "conditions": ["user_only_history", "state_summary"],
        "system_prompt": "Track the case.",
        "checkpoint_path": checkpoint,
        "seed": 7,
    }

    records = run_primary_dialogue_validation_model(**kwargs)

    assert len(records) == 24
    assert len(client.calls) == 24
    resumed_client = _GoldClient()
    resumed = run_primary_dialogue_validation_model(
        **{**kwargs, "client": resumed_client}
    )
    assert len(resumed) == 24
    assert resumed_client.calls == []


def test_cluster_aware_gate_detects_late_history_load():
    models = ("model-a", "model-b")
    dialogues = [f"dialogue-{index}" for index in range(9)]
    records_by_model = {
        model: [
            _record(model, dialogue, condition, turn)
            for dialogue in dialogues
            for condition in ("user_only_history", "state_summary")
            for turn in range(1, 13)
        ]
        for model in models
    }
    gate = {
        "expected_models": list(models),
        "expected_dialogues": dialogues,
        "expected_conditions": ["user_only_history", "state_summary"],
        "expected_turns_per_dialogue": 12,
        "expected_records_per_model": 216,
        "expected_post_baseline_pairs_per_model": 99,
        "max_output_tokens": 1024,
        "minimum_success_rate": 1.0,
        "minimum_strict_final_marker_rate": 0.99,
        "minimum_stored_answer_agreement_rate": 0.99,
        "maximum_truncations": 0,
        "maximum_token_ceiling_hits": 0,
        "minimum_state_summary_accuracy": 0.95,
        "minimum_mean_dialogue_history_load_gap": 0.10,
        "maximum_one_sided_sign_flip_p": 0.05,
        "minimum_positive_dialogue_gaps": 7,
        "minimum_models_meeting_primary_gate": 2,
        "early_turn_start": 2,
        "early_turn_end": 6,
        "late_turn_start": 7,
        "late_turn_end": 12,
        "minimum_mean_longitudinal_interaction": 0.05,
        "maximum_longitudinal_sign_flip_p": 0.05,
        "minimum_positive_dialogue_interactions": 6,
        "minimum_models_meeting_longitudinal_gate": 1,
        "require_nonnegative_interaction_all_models": True,
    }

    report = assess_primary_validation(records_by_model, gate)

    assert report["pass"] is True
    assert report["primary_qualifying_models"] == list(models)
    assert report["longitudinal_qualifying_models"] == list(models)
