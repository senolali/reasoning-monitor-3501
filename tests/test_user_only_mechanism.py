from __future__ import annotations

from reasoning_monitor.evaluation.user_only_mechanism import assess_user_only_mechanism


def _record(model: str, condition: str, turn: int, correct: bool) -> dict:
    answer = "10" if correct else "11"
    return {
        "protocol_id": "CDR-v1.2",
        "history_transport": (
            "packed_user_only_context"
            if condition == "user_only_history"
            else "packed_system_context"
        ),
        "model": model,
        "dialogue_id": "dialogue-1",
        "template": "test",
        "split": "calibration",
        "qid": f"{condition}-{turn}",
        "turn": turn,
        "condition": condition,
        "role": "primary",
        "call_index": 0,
        "gold_answer": "10",
        "answer": answer,
        "raw_text": f"FINAL_ANSWER: {answer}",
        "reasoning_steps": [],
        "context_chars": 100,
        "n_tokens": 10,
        "latency_s": 0.1,
        "ok": True,
        "finish_reason": "stop",
        "intrinsic_updates": 1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "CALLER-MANAGED PRIOR USER HISTORY"
                    if turn > 1 and condition == "user_only_history"
                    else "context"
                ),
            },
            {"role": "user", "content": "question"},
        ],
    }


def test_user_only_gate_detects_history_load():
    models = ["model-a", "model-b"]
    user = {}
    existing = {}
    for model in models:
        user[model] = [
            _record(model, "user_only_history", turn, turn <= 3)
            for turn in range(1, 10)
        ]
        existing[model] = [
            _record(model, condition, turn, True)
            for condition in ("self_history", "state_summary")
            for turn in range(1, 10)
        ]
    gate = {
        "expected_models": models,
        "expected_records_per_model": 9,
        "max_output_tokens": 512,
        "minimum_success_rate": 1.0,
        "minimum_strict_final_marker_rate": 0.99,
        "minimum_stored_answer_agreement_rate": 0.99,
        "maximum_truncations": 0,
        "maximum_token_ceiling_hits": 0,
        "minimum_history_load_gap": 0.10,
        "maximum_one_sided_exact_p": 0.05,
        "minimum_user_only_failure_dialogues": 1,
        "minimum_models_meeting_history_load_gate": 2,
    }

    report = assess_user_only_mechanism(user, existing, gate)

    assert report["pass"] is True
    assert report["qualifying_models"] == models
