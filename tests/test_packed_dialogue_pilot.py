from __future__ import annotations

from reasoning_monitor.evaluation.packed_dialogue_pilot import (
    assess_packed_dialogue_pilot,
)


def _records(model: str, self_correct_through: int) -> list[dict]:
    records = []
    for dialogue in ("budget", "incident", "cohort"):
        for condition in ("self_history", "oracle_history", "state_summary"):
            for turn in range(1, 13):
                correct = condition != "self_history" or turn <= self_correct_through
                answer = "10" if correct else "11"
                records.append(
                    {
                        "protocol_id": "CDR-v1.1",
                        "history_transport": "packed_system_context",
                        "model": model,
                        "dialogue_id": f"cdr-v1-{dialogue}-001",
                        "template": dialogue,
                        "split": "calibration",
                        "qid": f"{dialogue}-{turn}",
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
                                    "CALLER-MANAGED PRIOR DIALOGUE TRANSCRIPT"
                                    if turn > 1 and condition != "state_summary"
                                    else "context"
                                ),
                            },
                            {"role": "user", "content": "question"},
                        ],
                    }
                )
    return records


def test_packed_pilot_gate_uses_post_baseline_contrast():
    gate = {
        "expected_models": ["model-a", "model-b"],
        "expected_dialogues": [
            "cdr-v1-budget-001",
            "cdr-v1-incident-001",
            "cdr-v1-cohort-001",
        ],
        "expected_conditions": ["self_history", "oracle_history", "state_summary"],
        "expected_turns_per_dialogue": 12,
        "expected_records_per_model": 108,
        "max_output_tokens": 512,
        "minimum_success_rate": 1.0,
        "minimum_strict_final_marker_rate": 0.99,
        "minimum_stored_answer_agreement_rate": 0.99,
        "maximum_truncations": 0,
        "maximum_token_ceiling_hits": 0,
        "minimum_state_summary_accuracy": 0.95,
        "minimum_oracle_history_accuracy": 0.85,
        "minimum_context_penalty_for_expansion": 0.10,
        "minimum_self_history_first_failures_for_expansion": 2,
        "minimum_models_meeting_phenomenon_gate": 1,
    }
    report = assess_packed_dialogue_pilot(
        {
            "model-a": _records("model-a", self_correct_through=8),
            "model-b": _records("model-b", self_correct_through=12),
        },
        gate,
    )

    assert report["pass"] is True
    assert report["qualifying_models"] == ["model-a"]
    assert report["models"]["model-a"]["context_penalty"] > 0.10
