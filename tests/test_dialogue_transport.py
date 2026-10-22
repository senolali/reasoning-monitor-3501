from __future__ import annotations

from reasoning_monitor.dialogue_transport import build_packed_turn_messages
from reasoning_monitor.evaluation.dialogue_transport import (
    assess_packed_transport_probe,
    audit_native_history_transport,
)


def test_packed_history_is_carried_in_one_system_message():
    messages = build_packed_turn_messages(
        current_user_prompt="Apply the next update.",
        history=[
            {"role": "user", "content": "Initial value is 10."},
            {"role": "assistant", "content": "FINAL_ANSWER: 10"},
        ],
        system_prompt="Continue the case.",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "[PRIOR USER TURN 1]" in messages[0]["content"]
    assert "Initial value is 10." in messages[0]["content"]
    assert "FINAL_ANSWER: 10" in messages[0]["content"]
    assert messages[1]["content"] == "Apply the next update."


def test_packed_transport_probe_gate():
    gate = {
        "expected_models": ["model-a"],
        "expected_conditions": ["self_history", "oracle_history"],
        "expected_records_per_model": 2,
        "minimum_success_rate": 1.0,
        "minimum_accuracy": 1.0,
        "minimum_condition_accuracy": 1.0,
        "minimum_strict_final_marker_rate": 1.0,
        "maximum_truncations": 0,
        "pass_authorizes": "call-free preparation only",
    }
    records = [
        {
            "condition": condition,
            "ok": True,
            "correct": True,
            "strict_final_marker": True,
            "finish_reason": "stop",
        }
        for condition in gate["expected_conditions"]
    ]

    report = assess_packed_transport_probe({"model-a": records}, gate)

    assert report["pass"] is True
    assert report["models"]["model-a"]["accuracy"] == 1.0


def test_native_transport_audit_detects_shared_post_turn_one_collapse():
    records = []
    for condition in ("self_history", "oracle_history", "state_summary"):
        for turn in (1, 2):
            correct = condition == "state_summary" or turn == 1
            answer = "10" if correct else "missing"
            records.append(
                {
                    "model": "model-a",
                    "dialogue_id": "dialogue-1",
                    "template": "test",
                    "split": "calibration",
                    "qid": f"{condition}-{turn}",
                    "turn": turn,
                    "condition": condition,
                    "role": "primary",
                    "gold_answer": "10",
                    "answer": answer,
                    "raw_text": f"FINAL_ANSWER: {answer}",
                    "context_chars": 10,
                    "n_tokens": 2,
                    "latency_s": 0.1,
                    "ok": True,
                    "finish_reason": "stop",
                    "intrinsic_updates": 1,
                    "messages": [{"role": "system", "content": "x"}]
                    if turn == 1 or condition == "state_summary"
                    else [
                        {"role": "system", "content": "x"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                    ],
                }
            )

    report = audit_native_history_transport({"model-a": records})

    assert report["transport_failure_signature_detected"] is True
    assert report["scientific_interpretation_authorized"] is False
