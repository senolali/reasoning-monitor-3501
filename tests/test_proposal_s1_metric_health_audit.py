import pandas as pd

from reasoning_monitor.evaluation.proposal_s1_metric_health_audit import audit_model


def test_audit_identifies_wrapper_paraphrases_and_constant_rs():
    records = []
    primary = "Question?\nOutput format: FINAL_ANSWER: <answer>"
    paraphrases = (
        f"Consider the following problem. {primary}",
        f"{primary} Please work through this carefully.",
        f"Here is a question that needs solving: {primary}",
    )
    for index in range(5):
        records.append(
            {
                "qid": "q1",
                "dataset": "gsm8k",
                "role": "run",
                "call_index": index,
                "prompt": primary,
                "raw_text": f"reason {index}",
                "answer": "10",
            }
        )
    for index, prompt in enumerate(paraphrases):
        records.append(
            {
                "qid": "q1",
                "dataset": "gsm8k",
                "role": "paraphrase",
                "call_index": index,
                "prompt": prompt,
                "raw_text": "reason",
                "answer": "10",
            }
        )
    frame = pd.DataFrame([{"CQ_raw": 1.0, "RS_raw": 1.0}])

    result = audit_model(records, frame)

    assert result["all_paraphrases_retain_original_prompt_verbatim"] is True
    assert result["rs_evaluable_equals_cq_correct_primary"] is True
    assert result["rs_all_one"] is True
    assert result["k_raw_diverse_queries"] == 1
    assert result["k_answer_diverse_queries"] == 0

