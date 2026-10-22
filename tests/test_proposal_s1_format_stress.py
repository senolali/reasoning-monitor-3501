from __future__ import annotations

from pathlib import Path

import yaml

from reasoning_monitor.data.loaders import (
    LoadedDataset,
    _render_answer_last_stress_prompt,
)
from reasoning_monitor.data.proposal_scenarios import (
    build_proposal_frozen_selection_stream,
)
from reasoning_monitor.evaluation.proposal_s1_format_stress import (
    assess_model_records,
    stream_sequence_sha256,
)
from reasoning_monitor.types import Query
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _query(index: int = 1) -> Query:
    return Query(
        qid=f"math-{index}",
        t=index,
        dataset="math",
        domain="math",
        prompt="Solve this problem.",
        gold_answer="10",
    )


def test_stress_prompt_has_a_hard_line_budget_and_plain_final_marker():
    prompt = _render_answer_last_stress_prompt(_query())

    assert prompt == "\n".join(
        [
            "Solve this problem.",
            "Output format:",
            "Solve using no more than six numbered lines.",
            "Each numbered line must contain exactly one short sentence.",
            "Do not continue reasoning after line 6.",
            "Immediately after the last reasoning line, output your best answer even if uncertain.",
            "The final line must be plain text exactly in this form: FINAL_ANSWER: <answer>",
            "Do not put the FINAL_ANSWER line inside LaTeX, a code block, brackets, or a list item.",
            "Do not write anything after the FINAL_ANSWER line.",
        ]
    )
    assert "no more than six numbered lines" in prompt
    assert "Do not continue reasoning after line 6." in prompt
    assert "best answer even if uncertain" in prompt
    assert "plain text exactly in this form: FINAL_ANSWER" in prompt
    assert "inside LaTeX" in prompt
    assert prompt.endswith("Do not write anything after the FINAL_ANSWER line.")


def test_frozen_selection_stream_preserves_qids_and_roles():
    datasets = {
        "math": LoadedDataset(
            name="math",
            queries=[_query(1), _query(2)],
            source="test",
            content_hash="hash",
        )
    }
    sequence = [
        {"dataset": "math", "qid": "math-2", "selection_role": "control"},
        {
            "dataset": "math",
            "qid": "math-1",
            "selection_role": "section58_failure",
        },
    ]

    spec, stream, plan = build_proposal_frozen_selection_stream(
        datasets,
        sequence,
        n_queries=2,
    )

    assert spec.change_points == []
    assert [query.qid for query in stream] == ["math-2", "math-1"]
    assert [query.t for query in stream] == [1, 2]
    assert [row["selection_role"] for row in plan] == [
        "control",
        "section58_failure",
    ]


def test_frozen_selection_stream_attaches_paraphrases_to_same_output_contract():
    source = _render_answer_last_stress_prompt(_query())
    datasets = {
        "math": LoadedDataset(
            name="math",
            queries=[
                Query(
                    qid="math-1",
                    t=1,
                    dataset="math",
                    domain="math",
                    prompt=source,
                    gold_answer="10",
                )
            ],
            source="test",
            content_hash="hash",
        )
    }
    sequence = [
        {
            "dataset": "math",
            "qid": "math-1",
            "source_prompt_body": "Solve this problem.",
            "paraphrases": ["Determine the solution.", "Find the answer."],
        }
    ]

    _, stream, _ = build_proposal_frozen_selection_stream(
        datasets,
        sequence,
        n_queries=1,
    )

    source_contract = source.split("\nOutput format:\n", maxsplit=1)[1]
    assert len(stream[0].paraphrases) == 2
    assert all(
        prompt.endswith(source_contract) for prompt in stream[0].paraphrases
    )
    assert stream[0].paraphrases[0].startswith("Determine the solution.")


def _records() -> list[dict]:
    return [
        {
            "t": 1,
            "qid": "math-failure",
            "dataset": "math",
            "role": "run",
            "call_index": 0,
            "raw_text": "1. Compute briefly.\nFINAL_ANSWER: 10",
            "answer": "10",
            "gold_answer": "10",
            "ok": True,
            "n_tokens": 20,
            "finish_reason": "stop",
        },
        {
            "t": 2,
            "qid": "math-control",
            "dataset": "math",
            "role": "run",
            "call_index": 0,
            "raw_text": "1. Compute briefly.\nFINAL_ANSWER: 11",
            "answer": "11",
            "gold_answer": "10",
            "ok": True,
            "n_tokens": 20,
            "finish_reason": "stop",
        },
    ]


def _gate(records: list[dict]) -> dict:
    return {
        "expected_queries_per_model": 2,
        "expected_sequence_sha256": stream_sequence_sha256(records),
        "expected_failure_qids": ["math-failure"],
        "expected_control_qids": ["math-control"],
        "max_output_tokens": 2048,
        "minimum_success_rate": 1.0,
        "minimum_final_line_marker_rate": 1.0,
        "minimum_stored_answer_agreement_rate": 1.0,
        "maximum_truncations": 0,
        "maximum_step_contract_violations": 0,
    }


def test_format_stress_gate_accepts_complete_contract_compliant_outputs():
    records = _records()
    result = assess_model_records(records, _gate(records))

    assert result["subgroups"]["section58_failures"]["accuracy"] == 1.0
    assert result["subgroups"]["controls"]["accuracy"] == 0.0
    assert result["overall_gate_pass"] is True


def test_format_stress_gate_rejects_latex_marker_extra_step_and_truncation():
    records = _records()
    records[0]["raw_text"] = (
        "1. A.\n2. B.\n3. C.\n4. D.\n5. E.\n6. F.\n7. G.\n"
        r"\text{FINAL_ANSWER: } 10"
    )
    records[0]["n_tokens"] = 2048
    result = assess_model_records(records, _gate(records))

    assert result["checks"]["final_line_marker"] is False
    assert result["checks"]["stored_answer_agreement"] is False
    assert result["checks"]["no_truncation"] is False
    assert result["checks"]["step_contract"] is False
    assert result["overall_gate_pass"] is False


def test_format_stress_config_has_the_frozen_20_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_format_stress_asu.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert estimate(raw)["model_calls"] == 20


def test_llama4_scout_stress_config_changes_only_the_model_arm():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_llama4_scout_stress_asu.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 10
    assert len(raw["models"]) == 1
    assert raw["models"][0]["name"] == "llama4_scout"
    assert raw["models"][0]["model"] == "aws/llama4_scout-17b"
    assert raw["dataset_selection_manifest"].endswith(
        "proposal_s1_format_stress.json"
    )
    assert raw["dataset_prompt_mode"] == "answer_last_stress"
