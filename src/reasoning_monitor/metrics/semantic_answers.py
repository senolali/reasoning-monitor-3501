"""Prospectively frozen semantic answer equivalence for CS and RS."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from reasoning_monitor.metrics.dimensions import answers_equal, symbolic_answers_equal

_LATEX_TEXT = re.compile(r"\\(?:text|mathrm)\{([^{}]+)\}")
_QUANTITY = re.compile(
    r"^\s*([-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:e[-+]?\d+)?|[-+]?\d+\s*/\s*\d+)\s*"
    r"([a-zA-Z\u00b0]+(?:\s+[a-zA-Z]+)?)\.?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    dimension: str
    scale: Decimal
    offset: Decimal


@dataclass(frozen=True, slots=True)
class SemanticAnswerPolicy:
    name: str
    source_path: str
    source_sha256: str
    units: dict[str, UnitDefinition]
    require_explicit_units_on_both: bool

    def equivalent(self, left: str, right: str, dataset: str) -> bool:
        if dataset == "gsm8k":
            left_surface = _quantity_surface(left)
            right_surface = _quantity_surface(right)
            if left_surface is not None or right_surface is not None:
                if left_surface is None or right_surface is None:
                    return False
                parsed_left = self._parse_quantity(left)
                parsed_right = self._parse_quantity(right)
                if parsed_left is None or parsed_right is None:
                    return left_surface == right_surface
                return self.quantities_equal(left, right)
        if answers_equal(left, right, dataset):
            return True
        if dataset == "math":
            return symbolic_answers_equal(left, right)
        return False

    def quantities_equal(self, left: str, right: str) -> bool:
        parsed_left = self._parse_quantity(left)
        parsed_right = self._parse_quantity(right)
        if parsed_left is None or parsed_right is None:
            return False
        left_value, left_unit = parsed_left
        right_value, right_unit = parsed_right
        if left_unit.dimension != right_unit.dimension:
            return False
        left_base = left_value * left_unit.scale + left_unit.offset
        right_base = right_value * right_unit.scale + right_unit.offset
        return math.isclose(
            float(left_base), float(right_base), rel_tol=1e-9, abs_tol=1e-9
        )

    def _parse_quantity(
        self, answer: str
    ) -> tuple[Decimal, UnitDefinition] | None:
        value = answer.strip()
        if value.startswith(r"\(") and value.endswith(r"\)"):
            value = value[2:-2].strip()
        if value.startswith(r"\[") and value.endswith(r"\]"):
            value = value[2:-2].strip()
        value = _LATEX_TEXT.sub(r"\1", value)
        match = _QUANTITY.fullmatch(value)
        if match is None:
            return None
        unit = self.units.get(_normalise_unit(match.group(2)))
        if unit is None:
            return None
        number = _decimal_value(match.group(1))
        return (number, unit) if number is not None else None


def _normalise_unit(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower().rstrip("."))


def _quantity_surface(answer: str) -> tuple[str, str] | None:
    value = answer.strip()
    if value.startswith(r"\(") and value.endswith(r"\)"):
        value = value[2:-2].strip()
    if value.startswith(r"\[") and value.endswith(r"\]"):
        value = value[2:-2].strip()
    value = _LATEX_TEXT.sub(r"\1", value)
    match = _QUANTITY.fullmatch(value)
    if match is None:
        return None
    number = _decimal_value(match.group(1))
    if number is None:
        return None
    return (str(number.normalize()), _normalise_unit(match.group(2)))


def _decimal_value(value: str) -> Decimal | None:
    surface = value.replace(",", "").replace(" ", "")
    try:
        if "/" in surface:
            numerator, denominator = surface.split("/", maxsplit=1)
            divisor = Decimal(denominator)
            return None if divisor == 0 else Decimal(numerator) / divisor
        return Decimal(surface)
    except (InvalidOperation, ValueError):
        return None


def _validate_policy(raw: dict[str, Any], path: Path) -> SemanticAnswerPolicy:
    if raw.get("name") != "semantic-v1":
        raise ValueError("semantic answer policy name must be 'semantic-v1'")
    if raw.get("require_explicit_units_on_both") is not True:
        raise ValueError("semantic-v1 requires explicit units on both answers")
    units: dict[str, UnitDefinition] = {}
    for dimension, definitions in (raw.get("dimensions") or {}).items():
        for definition in definitions:
            canonical = str(definition["unit"])
            unit = UnitDefinition(
                dimension=str(dimension),
                scale=Decimal(str(definition["scale"])),
                offset=Decimal(str(definition.get("offset", 0))),
            )
            aliases = [canonical, *(definition.get("aliases") or [])]
            for alias in aliases:
                key = _normalise_unit(str(alias))
                if key in units:
                    raise ValueError(f"duplicate semantic unit alias: {key}")
                units[key] = unit
    if not units:
        raise ValueError("semantic-v1 must define at least one unit")
    return SemanticAnswerPolicy(
        name="semantic-v1",
        source_path=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units=units,
        require_explicit_units_on_both=True,
    )


@lru_cache(maxsize=8)
def load_semantic_policy(path: str) -> SemanticAnswerPolicy:
    policy_path = Path(path)
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return _validate_policy(raw, policy_path)


def semantic_answers_equal(
    left: str,
    right: str,
    dataset: str,
    policy_path: str,
) -> bool:
    return load_semantic_policy(policy_path).equivalent(left, right, dataset)


def semantic_rs_answer_correct(
    answer: str,
    gold: str,
    primary_answer: str,
    dataset: str,
    policy_path: str,
) -> bool:
    policy = load_semantic_policy(policy_path)
    if (
        dataset == "gsm8k"
        and _quantity_surface(answer) is not None
        and _quantity_surface(primary_answer) is not None
    ):
        return policy.equivalent(answer, primary_answer, dataset)
    return answers_equal(answer, gold, dataset) or policy.equivalent(
        answer, primary_answer, dataset
    )
