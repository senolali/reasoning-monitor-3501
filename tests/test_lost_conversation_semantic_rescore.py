from reasoning_monitor.lost_conversation_semantic_rescore import (
    readiness_gap,
    semantic_parse,
    semantic_records,
)


def test_semantic_parser_accepts_observed_marker_variants():
    assert semantic_parse("4. FINAL_ANSWER: STATUS: NEED_MORE_INFORMATION.")["kind"] == (
        "need_more_information")
    assert semantic_parse("FINAL_ANSWER: 79 miles.")["answer"] == "79"
    assert semantic_parse(
        "6. Therefore, FINAL_ANSWER: $64 when expressed in dollars. (128 kg * $0.50/kg)"
    )["answer"] == "64"
    assert semantic_parse("8. FINAL_ANSWER: 50 + 32 = 82 degrees.")["answer"] == "82"


def test_semantic_parser_remains_conservative():
    assert semantic_parse("The answer cannot be determined.")["contract_valid"] is False
    assert semantic_parse("FINAL_ANSWER: <To be determined>")["contract_valid"] is False
    assert semantic_parse("The calculation uses 10 and 20.")["contract_valid"] is False


def test_semantic_records_maps_parser_kind_to_scoring_field():
    source = {"raw_text": "FINAL_ANSWER: 79 miles.", "contract_valid": False,
              "response_kind": "invalid", "answer": ""}
    row = semantic_records([source])[0]
    assert row["contract_valid"] is True
    assert row["response_kind"] == "answer_attempt"
    assert row["answer"] == "79"


def test_readiness_gap_detects_history_lag_before_final():
    rows = []
    for turn, history_kind, fresh_kind in (
            (1, "need_more_information", "need_more_information"),
            (2, "need_more_information", "answer_attempt"),
            (3, "need_more_information", "answer_attempt")):
        for condition, kind in (("dialogue_history", history_kind),
                                ("fresh_prefix", fresh_kind)):
            rows.append({"model": "m", "task_id": "t", "turn": turn, "total_turns": 3,
                         "condition": condition, "role": "primary", "response_kind": kind})
    frame, summary = readiness_gap(rows)
    assert list(frame.readiness_gap) == [0.0, 1.0, 1.0]
    assert summary["m"]["tasks_with_pre_final_positive_gap"] == 1
    assert summary["m"]["median_lead_turns"] == 1.0
    assert summary["m"]["tasks_with_positive_final_gap"] == 1
