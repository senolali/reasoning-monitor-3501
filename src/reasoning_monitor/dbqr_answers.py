"""Strict, call-free parsing and typed comparison for DBQR answers."""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

MARKER = "FINAL_ANSWER_JSON:"
_ORDINAL_WORDS = {
    "greater": "greater",
    "higher": "greater",
    "more": "greater",
    "lower": "lower",
    "less": "lower",
}


def parse_final_answer(text: str):
    """Parse one JSON value from the final non-empty line only."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[-1].startswith(MARKER):
        raise ValueError("Missing final-line JSON marker")
    payload = lines[-1][len(MARKER):].strip()
    if not payload:
        raise ValueError("Empty final answer")

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(payload, parse_constant=reject_constant)
    if isinstance(value, bool) or value is None:
        raise ValueError("Boolean and null answers are not valid DBQR values")
    return value


def _decimal(value) -> Decimal:
    if isinstance(value, bool):
        raise InvalidOperation
    if isinstance(value, (int, float, str)):
        result = Decimal(str(value).replace(",", "").strip())
        if result.is_finite():
            return result
    raise InvalidOperation


def _scalar_equal(answer, label) -> bool:
    if isinstance(label, str):
        if not isinstance(answer, str):
            return False
        prediction = re.sub(r"\s+", " ", answer.strip()).casefold()
        reference = re.sub(r"\s+", " ", label.strip()).casefold()
        return _ORDINAL_WORDS.get(prediction, prediction) == _ORDINAL_WORDS.get(
            reference, reference)
    if isinstance(label, int) and not isinstance(label, bool):
        try:
            prediction = _decimal(answer)
            return prediction == prediction.to_integral_value() == Decimal(label)
        except (InvalidOperation, ValueError):
            return False
    if isinstance(label, float):
        try:
            prediction = _decimal(answer).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
            reference = _decimal(label).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
            return prediction == reference
        except (InvalidOperation, ValueError):
            return False
    return False


def typed_answer_equal(answer, label, *, ordered_lists: bool = False) -> bool:
    """Compare JSON values with unordered-list primary and ordered sensitivity."""
    if isinstance(label, list):
        if not isinstance(answer, list) or len(answer) != len(label):
            return False
        if ordered_lists:
            return all(typed_answer_equal(a, b, ordered_lists=True)
                       for a, b in zip(answer, label, strict=True))
        unmatched = list(answer)
        for reference in label:
            for index, prediction in enumerate(unmatched):
                if typed_answer_equal(prediction, reference):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    if isinstance(label, dict):
        return (isinstance(answer, dict) and set(answer) == set(label)
                and all(typed_answer_equal(answer[key], value, ordered_lists=ordered_lists)
                        for key, value in label.items()))
    return _scalar_equal(answer, label)
