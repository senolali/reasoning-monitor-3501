import pytest

from reasoning_monitor.dbqr_answers import parse_final_answer, typed_answer_equal


def test_final_line_json_contract():
    assert parse_final_answer("Reasoning.\nFINAL_ANSWER_JSON: 12.5") == 12.5
    assert parse_final_answer('FINAL_ANSWER_JSON: {"2021": 4}') == {"2021": 4}
    with pytest.raises(ValueError, match="final-line"):
        parse_final_answer("FINAL_ANSWER_JSON: 2\nextra")
    with pytest.raises(ValueError, match="Non-finite"):
        parse_final_answer("FINAL_ANSWER_JSON: NaN")


def test_typed_numeric_comparison_does_not_truncate_integers():
    assert typed_answer_equal("12", 12)
    assert not typed_answer_equal(12.9, 12)
    assert typed_answer_equal("1,234.125", 1234.125)
    assert not typed_answer_equal("1234.13", 1234.125)


def test_list_primary_and_ordered_sensitivity():
    assert typed_answer_equal(["B", "a"], ["A", "B"])
    assert not typed_answer_equal(["B", "a"], ["A", "B"], ordered_lists=True)
    assert not typed_answer_equal(["A", "A"], ["A", "B"])


def test_dictionary_requires_exact_keys_and_typed_values():
    assert typed_answer_equal({"2021": "12.50"}, {"2021": 12.5})
    assert not typed_answer_equal({"2021": 12.5, "extra": 1}, {"2021": 12.5})
