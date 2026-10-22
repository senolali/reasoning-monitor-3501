from __future__ import annotations

from reasoning_monitor.evaluation.user_only_capacity_retry import (
    assess_capacity_retry,
)


def _record(
    model: str,
    condition: str,
    turn: int,
    *,
    correct: bool,
    retry: bool = False,
) -> dict:
    answer = "10" if correct else "11"
    message_hash = f"hash-{turn}"
    return {
        "protocol_id": "CDR-v1.2a" if retry else "CDR-v1.2",
        "history_transport": (
            "packed_user_only_context"
            if condition == "user_only_history"
            else "packed_system_context"
        ),
        "model": model,
        "dialogue_id": "dialogue-1",
        "template": "test",
        "split": "calibration",
        "qid": f"user-{turn}" if condition == "user_only_history" else f"{condition}-{turn}",
        "turn": turn,
        "condition": condition,
        "role": "primary",
        "retry_role": "capacity_retry" if retry else None,
        "call_index": 0,
        "gold_answer": "10",
        "answer": answer,
        "raw_text": f"FINAL_ANSWER: {answer}",
        "reasoning_steps": [],
        "context_chars": 100,
        "n_tokens": 30,
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
        "messages_sha256": message_hash,
        "source_messages_sha256": message_hash if retry else None,
    }


def test_capacity_retry_replaces_only_target_and_resolves_mechanism():
    model = "model-a"
    original = [
        _record(
            model,
            "user_only_history",
            turn,
            correct=turn <= 2,
        )
        for turn in range(1, 5)
    ]
    retry = [
        _record(
            model,
            "user_only_history",
            4,
            correct=False,
            retry=True,
        )
    ]
    existing = [
        _record(model, condition, turn, correct=True)
        for condition in ("self_history", "state_summary")
        for turn in range(1, 5)
    ]
    gate = {
        "expected_models": [model],
        "expected_retry_counts": {model: 1},
        "expected_records_per_model": 4,
        "max_output_tokens": 1024,
        "minimum_success_rate": 1.0,
        "minimum_strict_final_marker_rate": 1.0,
        "minimum_stored_answer_agreement_rate": 1.0,
        "maximum_truncations": 0,
        "maximum_token_ceiling_hits": 0,
        "minimum_history_load_gap": 0.1,
        "maximum_one_sided_exact_p": 0.25,
        "minimum_user_only_failure_dialogues": 1,
    }

    report = assess_capacity_retry(
        {model: retry},
        {model: original},
        {model: existing},
        gate,
    )

    assert report["pass"] is True
    assert report["mechanism_resolved"] is True
    assert report["capacity_adjusted_qualifying_models"] == [model]
    assert report["models"][model]["capacity_adjusted"]["mechanism_pass"] is True
