from reasoning_monitor.evaluation.supergpqa_pilot_assessment import (
    _write_report,
    assess_records,
)
from reasoning_monitor.evaluation.supergpqa_pool_audit import (
    _label_balanced_sample,
    row_issues,
)
from reasoning_monitor.metrics.dimensions import normalize_answer
from reasoning_monitor.models.clients import final_answer_from_text


def _row(**updates):
    value = {
        "uuid": "u1",
        "question": "Question?",
        "options": ["first", "second", "third", "fourth"],
        "answer": "second",
        "answer_letter": "B",
        "discipline": "Science",
        "field": "Chemistry",
        "subfield": "Analytical Chemistry",
        "difficulty": "middle",
        "is_calculation": False,
    }
    value.update(updates)
    return value


def test_supergpqa_row_validation_accepts_matching_choice():
    assert row_issues(_row()) == set()


def test_supergpqa_row_validation_detects_bad_key_and_duplicate_options():
    issues = row_issues(
        _row(answer="not second", options=["same", "same", "third", "fourth"])
    )
    assert issues == {"answer_option_mismatch", "duplicate_options"}


def test_label_balanced_sample_is_deterministic_and_balanced():
    rows = []
    for label in "ABC":
        for index in range(4):
            rows.append(_row(uuid=f"{label}{index}", answer_letter=label))

    first = _label_balanced_sample(rows, 6, "test")
    second = _label_balanced_sample(rows, 6, "test")

    assert [row["uuid"] for row in first] == [row["uuid"] for row in second]
    assert {label: sum(row["answer_letter"] == label for row in first) for label in "ABC"} == {
        "A": 2,
        "B": 2,
        "C": 2,
    }


def test_supergpqa_choice_parser_uses_anchored_final_marker():
    raw = "A may appear in the reasoning.\nFINAL_CHOICE: H\n"
    assert final_answer_from_text(raw) == "H"
    assert normalize_answer(raw, "supergpqa") == "h"
    assert normalize_answer("H.", "supergpqa") == "h"


def test_supergpqa_answer_first_prompt_is_unambiguous():
    from reasoning_monitor.data.loaders import _render_supergpqa_prompt

    prompt = _render_supergpqa_prompt(_row(), "choice_first")

    assert "first output line must be exactly: FINAL_CHOICE: X" in prompt
    assert "at most four short sentences" in prompt


def test_supergpqa_pilot_gate_accepts_separated_parser_compliant_profile():
    gate = {
        "section": 53,
        "expected_queries_per_tier": 20,
        "max_output_tokens": 512,
        "minimum_success_rate": 1.0,
        "minimum_valid_choice_rate": 1.0,
        "minimum_final_choice_marker_rate": 0.95,
        "minimum_easy_hard_gap": 0.20,
        "minimum_tier_accuracy": 0.10,
        "maximum_tier_accuracy": 0.90,
        "require_monotonic_nonincreasing": True,
    }
    correct_by_tier = {"easy": 16, "medium": 10, "hard": 6}
    records = []
    for tier, n_correct in correct_by_tier.items():
        for index in range(20):
            gold = "A"
            answer = gold if index < n_correct else "B"
            records.append(
                {
                    "role": "run",
                    "qid": f"{tier}-{index}",
                    "difficulty": tier,
                    "answer": answer,
                    "gold_answer": gold,
                    "raw_text": f"Reasoning.\nFINAL_CHOICE: {answer}",
                    "ok": True,
                    "finish_reason": "stop",
                    "n_tokens": 100,
                }
            )

    result = assess_records(records, gate)

    assert result["accuracy"] == {"easy": 0.8, "medium": 0.5, "hard": 0.3}
    assert result["ordered_trend"]["one_sided_p"] < 0.05
    assert result["inference_valid"] is True
    assert result["overall_gate_pass"] is True


def test_supergpqa_pilot_gate_detects_gateway_token_ceiling_despite_stop_reason():
    gate = {
        "section": 52,
        "expected_queries_per_tier": 1,
        "max_output_tokens": 512,
        "minimum_success_rate": 1.0,
        "minimum_valid_choice_rate": 1.0,
        "minimum_final_choice_marker_rate": 0.95,
        "minimum_easy_hard_gap": 0.20,
        "minimum_tier_accuracy": 0.0,
        "maximum_tier_accuracy": 1.0,
        "require_monotonic_nonincreasing": False,
    }
    records = [
        {
            "role": "run",
            "qid": tier,
            "difficulty": tier,
            "answer": "unfinished reasoning",
            "gold_answer": "A",
            "raw_text": "unfinished reasoning",
            "ok": True,
            "finish_reason": "stop",
            "n_tokens": 512,
        }
        for tier in ("easy", "medium", "hard")
    ]

    result = assess_records(records, gate)

    assert result["truncations"] == 3
    assert result["checks"]["execution"] is False
    assert result["inference_valid"] is False


def test_supergpqa_validation_report_uses_frozen_significance_threshold(tmp_path):
    gate = {
        "section": 54,
        "minimum_easy_hard_gap": 0.20,
        "maximum_one_sided_trend_p": 0.05,
    }
    result = {
        "success_rate": 1.0,
        "truncations": 0,
        "valid_choice_rate": 1.0,
        "final_choice_marker_rate": 1.0,
        "first_line_choice_marker_rate": 1.0,
        "correct": {"easy": 5, "medium": 6, "hard": 1},
        "valid_choice_counts": {"easy": 20, "medium": 20, "hard": 20},
        "tier_counts": {"easy": 20, "medium": 20, "hard": 20},
        "accuracy": {"easy": 0.25, "medium": 0.30, "hard": 0.05},
        "wilson_95_ci": {
            "easy": [0.11, 0.47],
            "medium": [0.15, 0.52],
            "hard": [0.01, 0.24],
        },
        "accuracy_bounds": {
            "easy": [0.25, 0.25],
            "medium": [0.30, 0.30],
            "hard": [0.05, 0.05],
        },
        "easy_hard_gap": 0.20,
        "ordered_trend": {"z": -1.58, "one_sided_p": 0.0569},
        "easy_hard_fisher_one_sided_p": 0.0909,
        "inference_valid": True,
        "checks": {"ordered_trend_significance": False},
        "overall_gate_pass": False,
    }
    report_path = tmp_path / "report.md"

    _write_report(report_path, result, gate)

    report = report_path.read_text(encoding="utf-8")
    assert "Overall validation gate: **FAIL**" in report
    assert "threshold is one-sided ordered-trend p <= 0.050" in report
    assert "No significance threshold" not in report
