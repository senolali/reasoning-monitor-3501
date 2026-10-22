"""Deterministic, value-free compact schemas for DBQR query-result tables."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

SCHEMA_VERSION = "DBQR-SCHEMA-v1"
MAX_TABLES = 16
MAX_TOTAL_CELLS = 500_000
MAX_SCHEMA_CHARS = 12_000


def _natural_key(text: str):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _kind(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, (float, np.floating)):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _year(value):
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    if isinstance(value, (int, np.integer)) and 1900 <= int(value) <= 2100:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"(?:19|20)\d{2}", value.strip()):
        return int(value)
    return None


def _axis_schema(axis: pd.Index) -> dict:
    levels = []
    years = set()
    for position in range(axis.nlevels):
        values = axis.get_level_values(position)
        kinds = Counter(_kind(value) for value in values)
        for value in values:
            parsed = _year(value)
            if parsed is not None:
                years.add(parsed)
        levels.append({
            "position": position,
            "name": _json_scalar(axis.names[position]),
            "cardinality": int(values.nunique(dropna=False)),
            "label_kinds": dict(sorted(kinds.items())),
        })
    return {"levels": levels, "years": sorted(years)}


def private_table_aliases(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Return the local-only private-key to public-alias mapping."""
    return {
        private_key: f"table_{number}"
        for number, private_key in enumerate(sorted(tables, key=_natural_key), start=1)
    }


def compact_schema(tables: dict[str, pd.DataFrame]):
    """Return public schema plus private alias mapping.

    The public object contains no table keys, axis labels other than bounded
    calendar-year candidates, or cell values. The private mapping is retained
    locally for later plan execution and must never be included in a prompt.
    """
    if not isinstance(tables, dict) or not 0 < len(tables) <= MAX_TABLES:
        raise ValueError("Invalid table collection")
    private_aliases = private_table_aliases(tables)
    ordered = sorted(tables.items(), key=lambda item: _natural_key(item[0]))
    total_cells = 0
    public_tables = []
    aliases = {}
    for private_key, frame in ordered:
        if not isinstance(private_key, str) or not isinstance(frame, pd.DataFrame):
            raise ValueError("DBQR schemas require string-keyed DataFrames")
        cells = int(frame.shape[0] * frame.shape[1])
        total_cells += cells
        if total_cells > MAX_TOTAL_CELLS:
            raise ValueError("Table collection exceeds the cell limit")
        alias = private_aliases[private_key]
        aliases[alias] = frame
        row_axis = _axis_schema(frame.index)
        column_axis = _axis_schema(frame.columns)
        public_tables.append({
            "alias": alias,
            "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
            "row_axis": row_axis,
            "column_axis": column_axis,
            "year_candidates": sorted(set(row_axis["years"] + column_axis["years"])),
            "entity_count": int(frame.index.get_level_values(0).nunique(dropna=False)),
            "dtype_counts": dict(sorted(Counter(str(dtype) for dtype in frame.dtypes).items())),
            "missing_cells": int(frame.isna().to_numpy().sum()),
        })
    schema = {"schema_version": SCHEMA_VERSION, "tables": public_tables}
    rendered = json.dumps(schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    schema["schema_sha256"] = hashlib.sha256(rendered.encode("ascii")).hexdigest()
    final_rendered = json.dumps(
        schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(final_rendered) > MAX_SCHEMA_CHARS:
        raise ValueError("Compact schema exceeds the character limit")
    return schema, aliases
