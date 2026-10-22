from __future__ import annotations

from reasoning_monitor.evaluation.primary_validation_sensitivity import (
    diagnose_primary_validation_sensitivity,
)


def _record(model: str, dialogue: str, condition: str, turn: int) -> dict:
    user_correct = turn not in {7, 8}
    answer = "10" if condition == "state_summary" or user_correct else "11"
    raw = f"FINAL_ANSWER: {answer}"
    n_tokens = 20
    if model == "model-b" and condition == "user_only_history" and turn == 8:
        answer = "unfinished"
        raw = "unfinished"
        n_tokens = 100
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
        "raw_text": raw,
        "reasoning_steps": [],
        "context_chars": 100,
        "n_tokens": n_tokens,
        "latency_s": 0.1,
        "ok": True,
        "finish_reason": "stop",
        "intrinsic_updates": 1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "VERIFIED STATE BEFORE THIS TURN"
                    if condition == "state_summary"
                    else (
                        "CALLER-MANAGED PRIOR USER HISTORY"
                        if turn > 1
                        else "base context"
                    )
                ),
            },
            {"role": "user", "content": "question"},
        ],
    }


def test_sensitivity_preserves_gate_and_bounds_ceiling_cell():
    models = ["model-a", "model-b"]
    dialogues = [f"dialogue-{index}" for index in range(9)]
    records = {
        model: [
            _record(model, dialogue, condition, turn)
            for dialogue in dialogues
            for condition in ("user_only_history", "state_summary")
            for turn in range(1, 13)
        ]
        for model in models
    }
    gate = {
        "expected_models": models,
        "expected_dialogues": dialogues,
        "expected_conditions": ["user_only_history", "state_summary"],
        "expected_turns_per_dialogue": 12,
        "expected_records_per_model": 216,
        "expected_post_baseline_pairs_per_model": 99,
        "max_output_tokens": 100,
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

    report = diagnose_primary_validation_sensitivity(records, gate)

    assert report["frozen_gate_preserved_as_failed"] is True
    assert report["additional_model_calls_authorized"] is False
    model_b = report["models"]["model-b"]
    assert len(model_b["capacity_censored_user_only_cells"]) == 9
    observed = model_b["capacity_scenarios"]["observed_censored_as_recorded"]
    optimistic = model_b["capacity_scenarios"][
        "optimistic_all_censored_answers_correct"
    ]
    assert optimistic["mean_dialogue_history_load_gap"] < observed[
        "mean_dialogue_history_load_gap"
    ]


def test_threshold_resolution_reports_one_answer_shortfall():
    model = "model-a"
    dialogues = [f"dialogue-{index}" for index in range(9)]
    records = [
        _record(model, dialogue, condition, turn)
        for dialogue in dialogues
        for condition in ("user_only_history", "state_summary")
        for turn in range(1, 13)
    ]
    # Five errors among 99 post-baseline state-summary cells gives 94/99.
    for record in records:
        if record["condition"] == "state_summary" and record["turn"] > 1:
            if sum(
                item["condition"] == "state_summary"
                and item["turn"] > 1
                and item["answer"] == "11"
                for item in records
            ) < 5:
                record["answer"] = "11"
                record["raw_text"] = "FINAL_ANSWER: 11"
    gate = {
        "expected_models": [model],
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
        "minimum_models_meeting_primary_gate": 1,
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

    report = diagnose_primary_validation_sensitivity({model: records}, gate)
    resolution = report["models"][model]["state_summary_threshold_resolution"]

    assert resolution["correct"] == 94
    assert resolution["minimum_correct_required"] == 95
    assert resolution["correct_answer_shortfall"] == 1
