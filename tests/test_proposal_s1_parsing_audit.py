from reasoning_monitor.evaluation.proposal_s1_parsing_audit import (
    extract_strict_final_answer,
)


def test_arc_boxed_choice_is_extracted_from_full_reasoning():
    raw = "Reasoning mentions option A.\nThe final answer is: $\\boxed{C}$"

    answer, method = extract_strict_final_answer(raw, "arc_challenge", False)

    assert answer == "C"
    assert method == "final_boxed"


def test_arc_prose_without_final_choice_is_unresolved():
    raw = "A thermometer measures temperature. None of the options fit."

    answer, method = extract_strict_final_answer(raw, "arc_challenge", False)

    assert answer is None
    assert method == "unresolved_choice"


def test_capped_math_without_final_marker_is_censored():
    raw = "Step 12: substitute x=3 and continue the calculation"

    answer, method = extract_strict_final_answer(raw, "math", True)

    assert answer is None
    assert method == "censored_at_ceiling"


def test_capped_math_with_explicit_final_box_is_observed():
    raw = r"Calculation complete. Final answer: $\boxed{\frac{3}{4}}$"

    answer, method = extract_strict_final_answer(raw, "math", True)

    assert answer == r"\frac{3}{4}"
    assert method == "boxed"
