from __future__ import annotations

from reasoning_monitor.evaluation.user_only_capacity import (
    diagnose_user_only_capacity,
)


def _record(
    model: str,
    dialogue: str,
    condition: str,
    turn: int,
    *,
    correct: bool,
    censored: bool = False,
) -> dict:
    answer = "10" if correct else "11"
    return {
        "model": model,
        "dialogue_id": dialogue,
        "template": "test",
        "split": "calibration",
        "qid": f"{dialogue}-{condition}-{turn}",
        "turn": turn,
        "condition": condition,
        "role": "primary",
        "gold_answer": "10",
        "answer": answer if not censored else "unfinished",
        "raw_text": (
            f"FINAL_ANSWER: {answer}" if not censored else "unfinished reasoning"
        ),
        "context_chars": 100,
        "n_tokens": 512 if censored else 20,
        "latency_s": 0.1,
        "ok": True,
        "finish_reason": "stop",
        "intrinsic_updates": 1,
    }


def test_capacity_diagnostic_computes_optimistic_bounds():
    model = "model-a"
    user = []
    state = []
    for dialogue in ("d1", "d2"):
        for turn in range(1, 6):
            censored = turn == 5
            correct = turn not in {4, 5}
            user.append(
                _record(
                    model,
                    dialogue,
                    "user_only_history",
                    turn,
                    correct=correct,
                    censored=censored,
                )
            )
            state.append(
                _record(
                    model,
                    dialogue,
                    "state_summary",
                    turn,
                    correct=True,
                )
            )
    config = {
        "expected_models": [model],
        "max_output_tokens": 512,
        "minimum_history_load_gap": 0.20,
        "maximum_one_sided_exact_p": 0.25,
        "minimum_failure_dialogues": 2,
    }

    report = diagnose_user_only_capacity({model: user}, {model: state}, config)
    result = report["models"][model]

    assert result["capacity_censored_cells"] == 2
    assert result["all_marker_failures_capacity_censored"] is True
    assert result["user_only_accuracy_bounds"] == [0.5, 0.75]
    assert result["state_summary_advantage_bounds"] == [0.25, 0.5]
    assert result["primary_signal_robust_to_all_censored_answers_being_correct"]
