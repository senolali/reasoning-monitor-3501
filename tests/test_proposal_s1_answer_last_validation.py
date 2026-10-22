from __future__ import annotations

from pathlib import Path

import yaml

from reasoning_monitor.data.loaders import (
    _render_answer_last_bounded_prompt,
    _render_answer_last_stress_prompt,
)
from reasoning_monitor.evaluation.proposal_s1_answer_last_validation import (
    assess_model_records,
    stream_sequence_sha256,
)
from reasoning_monitor.types import Query
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _arc_query() -> Query:
    return Query(
        qid="arc_challenge-1",
        t=1,
        dataset="arc_challenge",
        domain="science",
        prompt="Choose the best answer.\nA. One\nB. Two\nC. Three\nD. Four",
        gold_answer="A",
    )


def test_arc_answer_last_prompt_requires_a_final_choice_line():
    prompt = _render_answer_last_bounded_prompt(_arc_query())

    assert "at most six short numbered reasoning steps" in prompt
    assert "FINAL_CHOICE: X" in prompt
    assert "one option letter from A through D" in prompt
    assert prompt.endswith("Do not write anything after the FINAL_CHOICE line.")


def test_stress_prompt_uses_dataset_typed_plain_final_lines():
    arc = _render_answer_last_stress_prompt(_arc_query())
    gsm8k = _render_answer_last_stress_prompt(
        Query(
            qid="gsm8k-1",
            t=1,
            dataset="gsm8k",
            domain="math_word_problem",
            prompt="What is 2 + 2?",
            gold_answer="4",
        )
    )

    for prompt in (arc, gsm8k):
        assert "no more than six numbered lines" in prompt
        assert "best answer even if uncertain" in prompt
        assert "inside LaTeX" in prompt
    assert "plain text exactly in this form: FINAL_CHOICE: X" in arc
    assert arc.endswith("Do not write anything after the FINAL_CHOICE line.")
    assert "plain text exactly in this form: FINAL_ANSWER: <answer>" in gsm8k
    assert gsm8k.endswith("Do not write anything after the FINAL_ANSWER line.")


def _records() -> list[dict]:
    records = []
    correct_by_dataset = {"gsm8k": 2, "arc_challenge": 1, "math": 0}
    t = 0
    for dataset in ("gsm8k", "arc_challenge", "math"):
        for index in range(2):
            t += 1
            gold = "A" if dataset == "arc_challenge" else "10"
            is_correct = index < correct_by_dataset[dataset]
            answer = gold if is_correct else (
                "B" if dataset == "arc_challenge" else "11"
            )
            marker = (
                "FINAL_CHOICE" if dataset == "arc_challenge" else "FINAL_ANSWER"
            )
            records.append(
                {
                    "t": t,
                    "qid": f"{dataset}-{index}",
                    "dataset": dataset,
                    "role": "run",
                    "call_index": 0,
                    "raw_text": f"1. Brief reason.\n{marker}: {answer}",
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
        "max_output_tokens": 2048,
        "minimum_success_rate": 1.0,
        "minimum_final_line_marker_rate": 1.0,
        "minimum_stored_answer_agreement_rate": 1.0,
        "maximum_truncations": 0,
        "maximum_step_contract_violations": 0,
        "forbid_code_fences": True,
        "minimum_gsm8k_math_gap": 0.15,
        "maximum_one_sided_trend_p": 1.0,
    }


def test_full_answer_last_gate_accepts_separated_complete_outputs():
    records = _records()
    result = assess_model_records(records, _gate(records))

    assert result["accuracy"] == {
        "gsm8k": 1.0,
        "arc_challenge": 0.5,
        "math": 0.0,
    }
    assert result["inference_valid"] is True
    assert result["overall_gate_pass"] is True


def test_full_answer_last_gate_rejects_wrong_arc_marker():
    records = _records()
    arc = next(record for record in records if record["dataset"] == "arc_challenge")
    arc["raw_text"] = "1. Brief reason.\nFINAL_ANSWER: A"
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["final_line_marker"] is False
    assert result["checks"]["stored_answer_agreement"] is False
    assert result["inference_valid"] is False


def test_full_answer_last_gate_rejects_nonfinal_duplicate_and_truncated_outputs():
    records = _records()
    records[0]["raw_text"] = "FINAL_ANSWER: 10\n1. More text."
    records[1]["raw_text"] = "FINAL_ANSWER: 9\nFINAL_ANSWER: 10"
    records[2]["finish_reason"] = "length"
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["final_line_marker"] is False
    assert result["checks"]["stored_answer_agreement"] is False
    assert result["checks"]["no_truncation"] is False
    assert result["overall_gate_pass"] is False


def test_full_answer_last_gate_rejects_line_seven_and_code_fences():
    records = _records()
    records[0]["raw_text"] = "7. Too late.\nFINAL_ANSWER: 10"
    records[1]["raw_text"] = "```text\n1. Reason.\n```\nFINAL_ANSWER: 10"
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["step_contract"] is False
    assert result["checks"]["final_line_marker"] is False
    assert result["inference_valid"] is False
    assert result["overall_gate_pass"] is False


def test_full_answer_last_validation_has_the_frozen_180_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_answer_last_validation_asu.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 180


def test_llama4_full_validation_has_the_frozen_90_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_llama4_scout_validation_asu.yaml")
        .read_text(encoding="utf-8")
    )
    gate = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_llama4_scout_validation_gate.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 90
    assert raw["stream_design"] == "proposal_multidataset"
    assert raw["dataset_prompt_mode"] == "answer_last_stress"
    assert raw["proposal_queries_per_dataset"] == 30
    assert len(raw["models"]) == 1
    assert raw["models"][0]["name"] == "llama4_scout"
    assert raw["models"][0]["model"] == "aws/llama4_scout-17b"
    assert gate["expected_models"] == ["llama4_scout"]
    assert gate["expected_stream_sequence_sha256"] == (
        "791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca"
    )
    assert gate["maximum_step_contract_violations"] == 0


def test_mistral_full_validation_has_the_frozen_asu_90_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_mistral_validation_asu.yaml")
        .read_text(encoding="utf-8")
    )
    gate = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_mistral_validation_gate.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 90
    assert raw["stream_design"] == "proposal_multidataset"
    assert raw["dataset_prompt_mode"] == "answer_last_stress"
    assert raw["proposal_queries_per_dataset"] == 30
    assert len(raw["models"]) == 1
    assert raw["models"][0]["name"] == "mistral_7b"
    assert raw["models"][0]["model"] == "aws/mistral-7b"
    assert gate["expected_models"] == ["mistral_7b"]
    assert gate["expected_provider_model"] == "aws/mistral-7b"
    assert gate["expected_stream_sequence_sha256"] == (
        "791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca"
    )
    assert gate["maximum_step_contract_violations"] == 0
    assert "ASU K=5 repeat-diversity" in gate["pass_authorizes"]
