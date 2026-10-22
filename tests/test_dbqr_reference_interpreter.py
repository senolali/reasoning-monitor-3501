import pandas as pd
import pytest

from reasoning_monitor.dbqr_answers import typed_answer_equal
from reasoning_monitor.dbqr_reference_interpreter import (
    ReferencePlanInterpreter,
    parse_statement,
    result_to_answer,
)


def test_series_filter_and_singleton_header_answer():
    table = pd.DataFrame({2021: [100, 400], 2022: [200, 500]}, index=["a", "b"])
    program = [
        "q = query(null)",
        "totals = sum(q, axis='columns')",
        "kept = loc_filter(totals, axis='rows', condition='ge', val='700')",
        "head = headers(kept, axis='rows')",
    ]
    result, _env = ReferencePlanInterpreter().execute(program, {"q": table})
    assert result_to_answer(result, ["b"]) == ["b"]


def test_choices_ignores_transport_value():
    program = ["q = query(null)", "answer = choices(q, choice='no')"]
    result, _env = ReferencePlanInterpreter().execute(
        program, {"q": pd.DataFrame({"x": [1]})})
    assert result_to_answer(result, "no") == "no"


def test_singleton_tuple_is_unwrapped_for_scalar_label():
    assert result_to_answer([(2022,)], 2022) == 2022


def test_single_row_table_becomes_year_keyed_answer_object():
    result = pd.DataFrame([[10.0, 20.0]], columns=[(2021,), (2022,)])
    assert result_to_answer(result, {"2021": 10.0, "2022": 20.0}) == {
        "2021": 10.0,
        "2022": 20.0,
    }


@pytest.mark.parametrize("answer,label", [
    ("greater", "higher"), ("greater", "more"), ("lower", "less")])
def test_predeclared_ordinal_synonyms(answer, label):
    assert typed_answer_equal(answer, label)


@pytest.mark.parametrize("statement", [
    "x = __import__('os').system('dir')",
    "x = query(__import__('os'))",
    "x = table[0]",
    "x = sum(table); y = sum(table)",
    "x, y = sum(table)",
])
def test_parser_rejects_non_declarative_source(statement):
    with pytest.raises(ValueError):
        parse_statement(statement)


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError, match="Unknown operation"):
        ReferencePlanInterpreter().execute(
            ["q = query(null)", "x = shell(q)"], {"q": pd.DataFrame({"x": [1]})})


def test_duplicate_target_is_rejected():
    with pytest.raises(ValueError, match="Duplicate target"):
        ReferencePlanInterpreter().execute(
            ["q = query(null)", "q = sum(q)"], {"q": pd.DataFrame({"x": [1]})})


def test_statement_length_is_bounded():
    with pytest.raises(ValueError, match="statement length"):
        parse_statement("x = choices(choice='" + "a" * 10_001 + "')")


def test_collection_output_size_is_bounded():
    interpreter = ReferencePlanInterpreter(max_output_cells=1)
    with pytest.raises(ValueError, match="cell limit"):
        interpreter.execute(
            ["q = query(null)", "x = headers(q)"],
            {"q": pd.DataFrame({"x": [1]}, index=["a"])},
        )
