from __future__ import annotations

import json

import pandas as pd

from reasoning_monitor.evaluation.risk_adjusted_trajectory_calibration import (
    assess_calibration_model,
    reasoning_line_count,
    strict_final_integer,
)


def _gate() -> dict:
    return {
        "expected_queries_per_model": 144,
        "expected_families": 24,
        "expected_l_paraphrases": 3,
        "expected_records_per_model": 576,
        "max_reasoning_lines": 8,
        "minimum_success_rate": 1.0,
        "minimum_final_marker_rate": 0.99,
        "minimum_stored_answer_agreement_rate": 0.99,
        "maximum_truncations": 0,
        "maximum_step_contract_violations": 0,
        "minimum_level_one_correct": 16,
        "minimum_first_failures": 6,
        "minimum_censored_families": 4,
        "minimum_event_templates": 2,
        "minimum_distinct_predictor_values": 3,
    }


def _synthetic_outputs() -> tuple[list[dict], pd.DataFrame]:
    records = []
    rows = []
    templates = ("arithmetic_chain", "shortest_path", "precedence_schedule")
    for family_index in range(24):
        family = f"family-{family_index:02d}"
        template = templates[family_index % 3]
        failure_level = 3 + family_index % 3 if family_index < 12 else None
        for level in range(1, 7):
            correct = int(failure_level is None or level < failure_level)
            qid = f"{family}-L{level}"
            rows.append(
                {
                    "family_id": family,
                    "trajectory_level": level,
                    "trajectory_split": "calibration",
                    "domain": template,
                    "qid": qid,
                    "CQ_raw": correct,
                    "RS_raw": 0.5 + family_index / 100 + level / 1000,
                    "LS_raw": 0.6 + family_index / 100 + level / 1000,
                    "ES_raw": 0.7 + family_index / 100 + level / 1000,
                    "call_failed": False,
                }
            )
            for role, count in (("run", 1), ("paraphrase", 3)):
                for call_index in range(count):
                    records.append(
                        {
                            "qid": qid,
                            "role": role,
                            "call_index": call_index,
                            "raw_text": "1. Compute carefully.\nFINAL_ANSWER: 42",
                            "answer": "42",
                            "ok": True,
                            "finish_reason": "stop",
                        }
                    )
    return records, pd.DataFrame(rows)


def test_strict_final_integer_contract():
    assert strict_final_integer("1. Work\nFINAL_ANSWER: 1,024") == "1,024"
    assert strict_final_integer("FINAL_ANSWER: 7\nextra") is None
    assert strict_final_integer("Answer: 7") is None
    assert reasoning_line_count("1. A\n2. B\nFINAL_ANSWER: 7") == 2


def test_calibration_gate_accepts_healthy_event_density():
    records, per_query = _synthetic_outputs()

    result = assess_calibration_model("model-a", records, per_query, _gate())

    assert result["pass"] is True
    assert result["risk_set"]["events"] == 12
    assert result["censored_families"] == 12
    assert result["event_templates"] == 3
    assert json.loads(json.dumps(result))["pass"] is True
