from __future__ import annotations

from reasoning_monitor.evaluation.packed_dialogue_diagnostic import (
    diagnose_packed_dialogue_pilot,
)


def _records(model: str) -> list[dict]:
    records = []
    for condition in ("self_history", "oracle_history", "state_summary"):
        for turn in range(1, 10):
            if condition == "state_summary":
                correct = True
            elif condition == "self_history":
                correct = turn < 4
            else:
                correct = turn < 3
            answer = "10" if correct else "11"
            context = 100 * turn if condition == "self_history" else 50 * turn
            records.append(
                {
                    "model": model,
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
                    "context_chars": context,
                    "n_tokens": 5,
                    "latency_s": 0.1,
                    "ok": True,
                    "finish_reason": "stop",
                    "intrinsic_updates": 1,
                }
            )
    return records


def test_diagnostic_separates_exploratory_signal_from_claim():
    report = diagnose_packed_dialogue_pilot(
        {"model-a": _records("model-a"), "model-b": _records("model-b")}
    )

    assert report["exploratory_cross_model_context_signal"] is True
    assert report["oracle_control_information_asymmetry"] is True
    assert report["decision"] == "freeze_format_matched_successor_design"
    assert report["scientific_claim_authorized"] is False
