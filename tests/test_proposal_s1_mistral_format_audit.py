from reasoning_monitor.evaluation.proposal_s1_mistral_format_audit import (
    audit_records,
    classify_strict_format_failure,
    extract_relaxed_typed_answer,
)


def _record(dataset: str, raw_text: str, answer: str, gold: str, t: int):
    return {
        "t": t,
        "qid": f"{dataset}-{t}",
        "dataset": dataset,
        "role": "run",
        "call_index": 0,
        "raw_text": raw_text,
        "answer": answer,
        "gold_answer": gold,
    }


def test_relaxed_choice_accepts_description_but_strict_contract_does_not():
    raw = "1. Reason.\nFINAL_CHOICE: C. stopwatch"

    assert extract_relaxed_typed_answer(raw, "arc_challenge") == "C"
    assert classify_strict_format_failure(raw, "arc_challenge") == (
        "choice_marker_has_extra_text"
    )


def test_relaxed_answer_recovers_marker_inside_numbered_step():
    raw = "1. Reason.\n6. FINAL_ANSWER: 16"

    assert extract_relaxed_typed_answer(raw, "gsm8k") == "16"
    assert classify_strict_format_failure(raw, "gsm8k") == (
        "marker_inside_numbered_step"
    )


def test_audit_keeps_strict_and_sensitivity_results_separate():
    records = [
        _record("gsm8k", "1. Reason.\nFINAL_ANSWER: 10", "10", "10", 1),
        _record(
            "arc_challenge",
            "1. Reason.\nFINAL_CHOICE: B. option text",
            "B",
            "B",
            2,
        ),
        _record("math", "6. FINAL_ANSWER: 3", "3", "3", 3),
    ]

    result = audit_records(records)

    assert result["strict_observable"] == 1
    assert result["relaxed_observable"] == 3
    assert result["strict_format_failures"] == 2
    assert result["datasets"]["arc_challenge"]["relaxed_correct"] == 1
    assert result["datasets"]["math"]["relaxed_correct"] == 1

