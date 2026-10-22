from __future__ import annotations

from pathlib import Path

import yaml

from reasoning_monitor.data.loaders import (
    LoadedDataset,
    _render_answer_last_bounded_prompt,
)
from reasoning_monitor.data.proposal_scenarios import (
    build_proposal_answer_order_calibration_stream,
)
from reasoning_monitor.evaluation.proposal_s1_answer_last_calibration import (
    assess_model_records,
    stream_sequence_sha256,
)
from reasoning_monitor.models.clients import final_answer_from_text, reasoning_steps_from_text
from reasoning_monitor.types import Query
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _query(dataset: str, index: int = 1) -> Query:
    return Query(
        qid=f"{dataset}-{index}",
        t=index,
        dataset=dataset,
        domain="math",
        prompt="Solve this problem.",
        gold_answer="10",
    )


def test_answer_last_prompt_requires_bounded_reasoning_and_a_final_typed_line():
    prompt = _render_answer_last_bounded_prompt(_query("gsm8k"))

    assert "at most six short numbered reasoning steps" in prompt
    assert "FINAL_ANSWER: <answer>" in prompt
    assert prompt.endswith("Do not write anything after the FINAL_ANSWER line.")


def test_typed_answer_parser_uses_last_marker_and_removes_all_markers_from_trace():
    raw = (
        "FINAL_ANSWER: 9\n"
        "1. Compute the requested value.\n"
        "2. The result is 10.\n"
        "FINAL_ANSWER: 10"
    )

    assert final_answer_from_text(raw) == "10"
    steps = reasoning_steps_from_text(raw)
    assert steps == ["1. Compute the requested value.", "2. The result is 10."]


def test_answer_order_stream_preserves_the_frozen_sequence():
    datasets = {
        name: LoadedDataset(
            name=name,
            queries=[_query(name, 1), _query(name, 2)],
            source="test",
            content_hash=f"{name}-hash",
        )
        for name in ("gsm8k", "math")
    }
    sequence = [
        {"dataset": "gsm8k", "qid": "gsm8k-2"},
        {"dataset": "gsm8k", "qid": "gsm8k-1"},
        {"dataset": "math", "qid": "math-2"},
        {"dataset": "math", "qid": "math-1"},
    ]

    spec, stream, plan = build_proposal_answer_order_calibration_stream(
        datasets,
        sequence,
        queries_per_dataset=2,
    )

    assert spec.change_point == 2
    assert [query.qid for query in stream] == [item["qid"] for item in sequence]
    assert [query.t for query in stream] == [1, 2, 3, 4]
    assert [query.difficulty for query in stream] == ["easy", "easy", "hard", "hard"]
    assert [row["phase"] for row in plan] == [1, 1, 2, 2]


def _records() -> list[dict]:
    records = []
    t = 0
    for dataset in ("gsm8k", "math"):
        for index in range(2):
            t += 1
            answer = "10" if dataset == "gsm8k" or index == 0 else "11"
            records.append(
                {
                    "t": t,
                    "qid": f"{dataset}-{index}",
                    "dataset": dataset,
                    "role": "run",
                    "call_index": 0,
                    "raw_text": f"1. Brief reason.\nFINAL_ANSWER: {answer}",
                    "answer": answer,
                    "gold_answer": "10",
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
        "minimum_gsm8k_accuracy": 0.6,
    }


def test_answer_last_gate_accepts_complete_observable_outputs():
    records = _records()
    result = assess_model_records(records, _gate(records))

    assert result["accuracy"] == {"gsm8k": 1.0, "math": 0.5}
    assert result["operationally_valid"] is True
    assert result["overall_gate_pass"] is True


def test_answer_last_gate_rejects_nonfinal_marker_and_truncation():
    records = _records()
    records[0]["raw_text"] = "FINAL_ANSWER: 10\n1. More text."
    records[1]["finish_reason"] = "length"
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["final_line_marker"] is False
    assert result["checks"]["stored_answer_agreement"] is False
    assert result["checks"]["no_truncation"] is False
    assert result["operationally_valid"] is False
    assert result["overall_gate_pass"] is False


def test_answer_last_calibration_config_has_the_frozen_40_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_answer_last_calibration_asu.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 40
