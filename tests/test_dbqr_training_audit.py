import pytest

from reasoning_monitor.dbqr_training_audit import operation_signature


def test_operation_signature_excludes_values_but_preserves_structure():
    first = ["a = query(company_1, year=2020)", "b = mean(a, axis='rows')"]
    second = ["x = query(company_9, year=2022)", "y = mean(x, axis='columns')"]
    assert operation_signature(first) == operation_signature(second) == ("query", "mean")


def test_operation_signature_rejects_unparsed_reference_code():
    with pytest.raises(ValueError, match="Unrecognised"):
        operation_signature(["not an operation"])
