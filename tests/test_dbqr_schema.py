import pandas as pd
import pytest

from reasoning_monitor.dbqr_schema import compact_schema


def test_schema_is_deterministic_and_omits_values_and_private_keys():
    tables = {
        "var_q1_p2": pd.DataFrame(
            [[123.45, None]],
            index=pd.MultiIndex.from_tuples(
                [("PRIVATE COMPANY", "us-gaap:PrivateConcept")],
                names=["company", "concept"],
            ),
            columns=pd.Index([2021, 2022], name="year"),
        )
    }
    first, aliases = compact_schema(tables)
    second, _aliases = compact_schema(tables)
    rendered = str(first)
    assert first == second
    assert list(aliases) == ["table_1"]
    assert first["tables"][0]["year_candidates"] == [2021, 2022]
    assert first["tables"][0]["entity_count"] == 1
    assert "123.45" not in rendered
    assert "PRIVATE COMPANY" not in rendered
    assert "us-gaap:PrivateConcept" not in rendered
    assert "var_q1_p2" not in rendered


def test_alias_order_is_natural_not_lexicographic():
    second = pd.DataFrame({"x": [2]})
    tenth = pd.DataFrame({"x": [10]})
    _schema, aliases = compact_schema({"var_q10": tenth, "var_q2": second})
    assert aliases["table_1"] is second
    assert aliases["table_2"] is tenth


def test_table_and_cell_limits_fail_closed():
    frame = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="collection"):
        compact_schema({f"q{i}": frame for i in range(17)})
    with pytest.raises(ValueError, match="cell limit"):
        compact_schema({"q": pd.DataFrame({"x": range(500_001)})})
