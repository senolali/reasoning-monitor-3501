from __future__ import annotations

from reasoning_monitor.data.loaders import _render_answer_first_prompt
from reasoning_monitor.evaluation.proposal_s1_answer_first_pilot import (
    assess_model_records,
    stream_sequence_sha256,
)
from reasoning_monitor.models.clients import final_answer_from_text, reasoning_steps_from_text
from reasoning_monitor.types import Query


def _query(dataset: str) -> Query:
    return Query(
        qid=f"{dataset}-1",
        t=1,
        dataset=dataset,
        domain="math" if dataset != "arc_challenge" else "science",
        prompt="Solve this problem.",
        gold_answer="B" if dataset == "arc_challenge" else "42",
    )


def test_answer_first_prompt_uses_dataset_typed_contracts():
    arc = _render_answer_first_prompt(_query("arc_challenge"))
    math = _render_answer_first_prompt(_query("math"))

    assert "first output line must be exactly: FINAL_CHOICE: X" in arc
    assert "one option letter from A through D" in arc
    assert "FINAL_ANSWER: <answer>" in math
    assert "at most eight short numbered reasoning steps" in math


def test_typed_answer_parser_preserves_math_and_excludes_marker_from_trace():
    raw = "FINAL_ANSWER: \\frac{3}{7}\n1. Reduce the expression.\n2. Stop."

    assert final_answer_from_text(raw) == r"\frac{3}{7}"
    steps = reasoning_steps_from_text(raw)
    assert steps
    assert all("FINAL_ANSWER" not in step for step in steps)
    assert "Reduce the expression" in steps[0]


def test_typed_choice_parser_ignores_letters_in_reasoning():
    raw = "FINAL_CHOICE: C\n1. Option A is tempting.\n2. C is supported."

    assert final_answer_from_text(raw) == "C"
    assert all("FINAL_CHOICE" not in step for step in reasoning_steps_from_text(raw))


def _records() -> list[dict]:
    records = []
    correct_by_dataset = {"gsm8k": 2, "arc_challenge": 1, "math": 0}
    t = 0
    for dataset in ("gsm8k", "arc_challenge", "math"):
        for index in range(2):
            t += 1
            gold = "A" if dataset == "arc_challenge" else "10"
            is_correct = index < correct_by_dataset[dataset]
            answer = gold if is_correct else ("B" if dataset == "arc_challenge" else "11")
            marker = "FINAL_CHOICE" if dataset == "arc_challenge" else "FINAL_ANSWER"
            records.append(
                {
                    "t": t,
                    "qid": f"{dataset}-{index}",
                    "dataset": dataset,
                    "role": "run",
                    "call_index": 0,
                    "raw_text": f"{marker}: {answer}\n1. Brief reason.",
                    "answer": answer,
                    "gold_answer": gold,
                    "ok": True,
                    "n_tokens": 20,
                    "finish_reason": "stop",
                }
            )
    return records


def _gate(records: list[dict]) -> dict:
    return {
        "expected_queries_per_dataset": 2,
        "expected_stream_sequence_sha256": stream_sequence_sha256(records),
        "max_output_tokens": 1024,
        "minimum_success_rate": 1.0,
        "minimum_first_line_marker_rate": 1.0,
        "minimum_stored_answer_agreement_rate": 1.0,
        "maximum_truncations": 0,
        "minimum_gsm8k_math_gap": 0.15,
        "maximum_one_sided_trend_p": 1.0,
    }


def test_s1_answer_first_gate_accepts_observable_separated_profile():
    records = _records()
    result = assess_model_records(records, _gate(records))

    assert result["accuracy"] == {"gsm8k": 1.0, "arc_challenge": 0.5, "math": 0.0}
    assert result["first_line_marker_rate"] == 1.0
    assert result["stored_answer_agreement_rate"] == 1.0
    assert result["inference_valid"] is True
    assert result["overall_gate_pass"] is True


def test_s1_answer_first_gate_rejects_missing_first_line_answer():
    records = _records()
    records[0]["raw_text"] = "Reasoning first.\nFINAL_ANSWER: 10"
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["first_line_marker"] is False
    assert result["checks"]["stored_answer_agreement"] is False
    assert result["inference_valid"] is False
    assert result["overall_gate_pass"] is False

