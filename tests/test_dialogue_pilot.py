from __future__ import annotations

import pytest

from reasoning_monitor.evaluation.dialogue_pilot import assess_dialogue_pilot


def _records(model: str, *, self_correct_turns: int) -> list[dict]:
    records = []
    dialogues = ("budget", "incident", "cohort")
    for dialogue_index, dialogue in enumerate(dialogues):
        for condition in ("self_history", "oracle_history", "state_summary"):
            for turn in range(1, 13):
                correct = condition != "self_history" or turn <= self_correct_turns
                gold = str(100 + dialogue_index + turn)
                answer = gold if correct else str(int(gold) + 1)
                raw = f"Concise calculation.\nFINAL_ANSWER: {answer}"
                records.append(
                    {
                        "model": model,
                        "dialogue_id": f"cdr-v1-{dialogue}-001",
                        "template": dialogue,
                        "split": "calibration",
                        "qid": f"{dialogue}-{turn}",
                        "turn": turn,
                        "condition": condition,
                        "role": "primary",
                        "call_index": 0,
                        "gold_answer": gold,
                        "answer": answer,
                        "raw_text": raw,
                        "reasoning_steps": ["Concise calculation."],
                        "context_chars": turn * 100,
                        "n_tokens": 8,
                        "latency_s": 0.1,
                        "ok": True,
                        "finish_reason": "stop",
                        "intrinsic_updates": 1,
                    }
                )
    return records


def test_dialogue_pilot_gate_detects_context_degradation():
    gate = {
        "expected_models": ["gpt4o_mini", "llama4_scout"],
        "expected_dialogues": [
            "cdr-v1-budget-001",
            "cdr-v1-incident-001",
            "cdr-v1-cohort-001",
        ],
        "expected_conditions": ["self_history", "oracle_history", "state_summary"],
        "expected_turns_per_dialogue": 12,
        "expected_primary_records_per_model": 108,
        "expected_total_records_per_model": 108,
        "minimum_success_rate": 1.0,
        "minimum_strict_final_marker_rate": 0.99,
        "minimum_stored_answer_agreement_rate": 0.99,
        "maximum_truncations": 0,
        "minimum_context_penalty_for_expansion": 0.10,
        "minimum_self_history_first_failures_for_expansion": 2,
        "minimum_models_meeting_phenomenon_gate": 1,
    }
    result = assess_dialogue_pilot(
        {
            "gpt4o_mini": _records("gpt4o_mini", self_correct_turns=8),
            "llama4_scout": _records("llama4_scout", self_correct_turns=12),
        },
        gate,
    )

    assert result["pass"] is True
    assert result["qualifying_models"] == ["gpt4o_mini"]
    assert result["models"]["gpt4o_mini"]["context_penalty"] == pytest.approx(1 / 3)
    assert result["models"]["llama4_scout"]["context_penalty"] == 0.0
