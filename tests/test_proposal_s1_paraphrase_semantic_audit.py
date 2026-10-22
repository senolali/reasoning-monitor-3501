import pandas as pd

from reasoning_monitor.evaluation.proposal_s1_paraphrase_semantic_audit import (
    audit_model_records,
    time_answers_equivalent,
)


def _records(paraphrase_answers):
    rows = [
        {
            "qid": "q1",
            "dataset": "gsm8k",
            "role": "run",
            "call_index": 0,
            "answer": "95 minutes",
            "gold_answer": "95",
        }
    ]
    rows.extend(
        {
            "qid": "q1",
            "dataset": "gsm8k",
            "role": "paraphrase",
            "call_index": index,
            "answer": answer,
            "gold_answer": "95",
        }
        for index, answer in enumerate(paraphrase_answers)
    )
    return rows


def test_time_answers_equivalent_converts_units():
    assert time_answers_equivalent("5700 seconds", "95 minutes") is True
    assert time_answers_equivalent("95 seconds", "95 minutes") is False
    assert time_answers_equivalent("95", "95 minutes") is False


def test_semantic_audit_removes_false_rs_variation_from_unit_conversion():
    frame = pd.DataFrame(
        [{"qid": "q1", "dataset": "gsm8k", "RS_raw": 2 / 3}]
    )

    result = audit_model_records(
        _records(["95 minutes", "95 minutes", "5700 seconds"]), frame
    )

    assert result["raw_rs_unique_values"] == 1
    assert result["semantic_rs_by_qid"] == {"q1": 1.0}
    assert len(result["unit_equivalence_corrections"]) == 1
