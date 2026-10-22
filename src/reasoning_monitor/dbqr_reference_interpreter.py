"""Restricted interpreter for DBQR reference plans.

This is an independent implementation of the documented table-operation
semantics. It accepts assignment-only expressions and never evaluates source.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAX_STEPS = 64
MAX_OUTPUT_CELLS = 200_000
MAX_STATEMENT_CHARS = 10_000
MAX_AST_NODES = 512
TERMINALS = {"empty", "invalid_data", "choices"}


def _copy(value):
    return value.copy(deep=True) if isinstance(value, (pd.DataFrame, pd.Series)) else value


def _todf(value, index):
    if np.isscalar(value) and not isinstance(value, str):
        labels = list(index) if isinstance(index, (pd.Index, list, tuple)) else [index]
        label = labels[0] if labels else "Value"
        return pd.Series([value], index=[label])
    if isinstance(value, (pd.DataFrame, pd.Series)) and value.empty:
        raise ValueError("Empty intermediate table")
    return value


def _constant(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(" ", "").lower()
    if text in {"", "-", "nan"}:
        return np.nan
    if text.endswith("%"):
        return float(text[:-1])
    if text[-1:] in "kmbt":
        return float(text[:-1]) * (1e3 ** ("kmbt".index(text[-1]) + 1))
    return float(text)


def _selector(text, levels: int):
    if not isinstance(text, str):
        return text
    compact = text.strip()
    if compact == ":":
        return slice(None)
    if compact == ":, :":
        return (slice(None), slice(None))
    if compact.startswith("[") and compact.endswith("]"):
        expression = re.sub(r",\s*]$", "]", compact)
        values = ast.literal_eval(expression)
        if not isinstance(values, list):
            raise ValueError("Selector must be a list")
        return values
    raise ValueError(f"Unsupported selector: {text}")


def _select(df, idx=":", columns=":"):
    row_key = _selector(idx, df.index.nlevels)
    if isinstance(df, pd.Series):
        return df.loc[row_key]
    column_key = _selector(columns, df.columns.nlevels)
    return df.loc[row_key, column_key]


def _const_value(source="custom", val=None, step=None):
    if source == "custom":
        return _constant(val)
    if isinstance(step, pd.Series):
        return step.iloc[0]
    if isinstance(step, pd.DataFrame):
        return step.iloc[0, 0]
    raise ValueError("Step-sourced constant requires a table")


def _eval_node(node: ast.AST, env: dict[str, Any]):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "null":
            return None
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        if node.id not in env:
            raise ValueError(f"Unknown reference: {node.id}")
        return env[node.id]
    if isinstance(node, ast.List):
        return [_eval_node(item, env) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(item, env) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_node(key, env): _eval_node(value, env)
                for key, value in zip(node.keys, node.values, strict=True)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_node(node.operand, env)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value
    raise ValueError(f"Forbidden expression: {type(node).__name__}")


def _validate_value_node(node: ast.AST, *, allow_template: bool = False):
    if isinstance(node, (ast.Constant, ast.Name)):
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            _validate_value_node(item, allow_template=allow_template)
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ValueError("Expanded dictionaries are forbidden")
            _validate_value_node(key, allow_template=allow_template)
            _validate_value_node(value, allow_template=allow_template)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        _validate_value_node(node.operand, allow_template=allow_template)
        return
    if allow_template and isinstance(node, ast.JoinedStr):
        for value in node.values:
            _validate_value_node(value, allow_template=True)
        return
    if allow_template and isinstance(node, ast.FormattedValue):
        if node.conversion not in {-1, 115, 114, 97} or node.format_spec is not None:
            raise ValueError("Formatted query values may not use format specifications")
        _validate_value_node(node.value, allow_template=True)
        return
    raise ValueError(f"Forbidden expression: {type(node).__name__}")


def parse_statement(statement: str):
    if not isinstance(statement, str) or len(statement) > MAX_STATEMENT_CHARS:
        raise ValueError("Invalid statement length")
    tree = ast.parse(statement, mode="exec")
    if sum(1 for _node in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Plan statement is too complex")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("Only one assignment is allowed per statement")
    assignment = tree.body[0]
    if (len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
            or not isinstance(assignment.value, ast.Call)
            or not isinstance(assignment.value.func, ast.Name)):
        raise ValueError("Invalid plan statement")
    if any(keyword.arg is None for keyword in assignment.value.keywords):
        raise ValueError("Expanded keyword arguments are forbidden")
    allow_template = assignment.value.func.id == "query"
    for node in assignment.value.args:
        _validate_value_node(node, allow_template=allow_template)
    for keyword in assignment.value.keywords:
        _validate_value_node(keyword.value, allow_template=allow_template)
    return assignment.targets[0].id, assignment.value


def _result_size(value) -> int:
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return int(value.size)
    if isinstance(value, dict):
        return len(value) + sum(_result_size(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value) + sum(_result_size(item) for item in value)
    return 1


@dataclass
class ReferencePlanInterpreter:
    max_steps: int = MAX_STEPS
    max_output_cells: int = MAX_OUTPUT_CELLS

    def execute(self, program: list[str], tables: dict[str, pd.DataFrame]):
        if not program or len(program) > self.max_steps:
            raise ValueError("Invalid plan length")
        env: dict[str, Any] = {}
        for statement in program:
            target, call = parse_statement(statement)
            operation = call.func.id
            if target in env:
                raise ValueError(f"Duplicate target: {target}")
            if operation == "query":
                if target not in tables:
                    raise ValueError(f"Missing query result: {target}")
                result = _copy(tables[target])
            else:
                args = [_eval_node(node, env) for node in call.args]
                kwargs = {keyword.arg: _eval_node(keyword.value, env)
                          for keyword in call.keywords}
                result = self._dispatch(operation, args, kwargs)
            cells = _result_size(result)
            if cells > self.max_output_cells:
                raise ValueError("Intermediate output exceeds cell limit")
            env[target] = result
        return env[target], env

    def _dispatch(self, operation, args, kwargs):
        method = getattr(self, f"op_{operation}", None)
        if method is None or operation.startswith("_"):
            raise ValueError(f"Unknown operation: {operation}")
        return method(*args, **kwargs)

    def op_loc(self, df, idx=":", columns=":"):
        return _select(_copy(df), idx, columns)

    def op_loc_filter(self, df, axis="rows", operation="all", condition="equal",
                      source="custom", val=None, step=None, idx=":", columns=":"):
        df = _copy(df)
        value = _const_value(source, val, step)
        selected = _select(df, idx, columns)
        comparisons = {"equal": selected == value, "gt": selected > value,
                       "lt": selected < value, "ge": selected >= value,
                       "le": selected <= value}
        if condition not in comparisons or operation not in {"all", "any"}:
            raise ValueError("Unsupported filter")
        if isinstance(selected, pd.Series):
            return df.loc[comparisons[condition]]
        mask = getattr(comparisons[condition], operation)(axis=0 if axis == "rows" else 1)
        return df.loc[:, mask] if axis == "rows" else df.loc[mask, :]

    def op_merge(self, df, dfs, axis="rows", ignore_index=False, replace_columns=True):
        values = [_copy(df), *[_copy(value) for value in dfs]]
        if any(isinstance(value, pd.Series) for value in values):
            values = [value.iloc[:, 0] if isinstance(value, pd.DataFrame) else value
                      for value in values]
            return pd.concat(values, ignore_index=ignore_index)
        if axis == "rows":
            if replace_columns:
                for value in values[1:]:
                    value.columns = values[0].columns
            return pd.concat(values, axis=0, ignore_index=ignore_index)
        return pd.concat(values, axis=1, ignore_index=ignore_index)

    def op_intersect(self, df, dfs, axis="rows", level=None):
        del axis
        df = _copy(df)
        index = df.index
        if level is None:
            for value in dfs:
                index = index.intersection(value.index)
        else:
            tuples = list(index)
            for value in dfs:
                if isinstance(value, (list, tuple)):
                    refs = set(value)
                elif len(value.index) and isinstance(value.index[0], str):
                    refs = set(value.index)
                else:
                    refs = {item[level] for item in value.index}
                tuples = [item for item in tuples if item[level] in refs]
            index = pd.MultiIndex.from_tuples(tuples)
        return df.loc[index]

    def op_headers(self, df, axis="rows", level=None):
        index = df.index if isinstance(df, pd.Series) or axis == "rows" else df.columns
        if index.nlevels == 1:
            return list(index)
        grouped = [list(dict.fromkeys(item[position] for item in index))
                   for position in range(index.nlevels)]
        return grouped if level is None else grouped[level]

    def op_abs(self, df):
        value = _copy(df).abs()
        return _todf(value, value.index)

    def op_sum(self, df, axis="rows", level=None, keep_col=False):
        df = _copy(df)
        if isinstance(df, pd.Series):
            result = df.sum() if level is None else df.groupby(level=level).sum()
            index = [("Sum",)] if level is None else result.index
        elif level is None:
            result = df.sum(axis=0 if axis == "rows" else 1)
            index = df.columns if axis == "rows" else df.index
        else:
            result = (df.groupby(level=level, axis=0).sum() if axis == "rows"
                      else df.T.groupby(level=level).sum().T)
            index = df.columns if axis == "rows" else df.index
        if keep_col and isinstance(result, pd.Series):
            result = pd.DataFrame(result, columns=[("Sum",)])
        return _todf(result, index)

    def op_mean(self, df, axis="rows", level=None, keep_col=False, skipna=True):
        df = _copy(df)
        if isinstance(df, pd.Series):
            result = (df.mean(skipna=skipna) if level is None
                      else df.groupby(level=level).mean())
            index = [("Average",)] if level is None else result.index
        elif level is None:
            result = df.mean(axis=0 if axis == "rows" else 1, skipna=skipna)
            index = df.columns if axis == "rows" else df.index
        else:
            result = (df.groupby(level=level, axis=0).mean() if axis == "rows"
                      else df.T.groupby(level=level).mean().T)
            index = df.columns if axis == "rows" else df.index
        if keep_col and isinstance(result, pd.Series):
            result = pd.DataFrame(result, columns=[("Average",)])
        return _todf(result, index)

    def op_diff(self, df, axis="rows"):
        df = _copy(df)
        if isinstance(df, pd.Series):
            return df.diff().iloc[1:]
        return (df.diff(axis=0).iloc[1:, :] if axis == "rows"
                else df.diff(axis=1).iloc[:, 1:])

    def op_div(self, df, axis="rows", keep_col=False):
        df = _copy(df)
        if keep_col:
            if isinstance(df, pd.Series):
                return df.iloc[1:].div(df.iloc[:-1].to_numpy())
            if axis == "rows":
                result = df.iloc[:, 1:].div(df.iloc[:, :-1].to_numpy())
                return result
            return df.iloc[1:, :].div(df.iloc[:-1, :].to_numpy())
        if isinstance(df, pd.Series):
            return _todf(df.iloc[1] / df.iloc[0], [("Divide",)])
        result = df.iloc[:, 1] / df.iloc[:, 0] if axis == "rows" else df.iloc[1, :] / df.iloc[0, :]
        return _todf(result, result.index)

    def op_sort(self, df, by=None, axis="rows", ascending=False):
        df = _copy(df)
        if isinstance(df, pd.Series) or by is None:
            return df.sort_values(ascending=ascending)
        if isinstance(by, str):
            by = ast.literal_eval(by.replace("null", "None"))
        if not by:
            return df
        if isinstance(by[0], list):
            by = [tuple(item) for item in by]
        return df.sort_values(by=list(by), axis=0 if axis == "rows" else 1,
                              ascending=ascending)

    def op_k_end(self, df, k, direction="first", axis="rows", reverse=False):
        df = _copy(df)
        if isinstance(k, str):
            text = k.strip().lower()
            if not text.isdigit():
                raise ValueError("Only integer k is supported")
            k = int(text)
        if reverse:
            k = -k
            direction = "last" if direction == "first" else "first"
        selection = slice(None, k) if direction == "first" else slice(-k, None)
        if isinstance(df, pd.Series):
            return df.iloc[selection]
        return df.iloc[:, selection] if axis == "rows" else df.iloc[selection, :]

    def op_compare(self, df, axis="rows", greater="greater", lower="lower", equal="equal"):
        def label(value):
            return greater if value > 0 else lower if value < 0 else equal if value == 0 else np.nan

        if isinstance(df, pd.Series):
            return df.diff().iloc[1:].map(label)
        delta = df.diff(axis=0).iloc[1:, :] if axis == "rows" else df.diff(axis=1).iloc[:, 1:]
        return delta.map(label)

    def op_count(self, df, axis="rows"):
        if isinstance(df, (list, pd.Series)):
            size = len(df)
        else:
            size = len(df.columns) if axis == "rows" else len(df.index)
        return pd.Series([size], index=[("Count",)])

    def op_cond(self, df, condition="equal", cond_source="custom", cond_val=None,
                cond_step=None, target_source="custom", target_val=None, target_step=None):
        df = _copy(df)
        source = _const_value(cond_source, cond_val, cond_step)
        target = _const_value(target_source, target_val, target_step)
        masks = {"equal": df == source, "gt": df > source, "lt": df < source,
                 "ge": df >= source, "le": df <= source, "ne": df != source}
        if condition not in masks:
            raise ValueError("Unsupported condition")
        df[masks[condition]] = target
        return df

    def _const_op(self, df, idx, columns, source, val, step, function, table=None):
        df = _copy(df)
        if source == "table":
            return function(df, table)
        selected = _select(df, idx, columns)
        updated = function(selected, _const_value(source, val, step))
        df.update(updated)
        return df

    def op_const_add(self, df, idx=":", columns=":", source="custom", val=None,
                     step=None, table=None):
        return self._const_op(df, idx, columns, source, val, step,
                              lambda left, right: left.add(right), table)

    def op_const_mul(self, df, idx=":", columns=":", source="custom", val="1",
                     step=None, table=None):
        return self._const_op(df, idx, columns, source, val, step,
                              lambda left, right: left.mul(right), table)

    def op_choices(self, _df=None, choice="yes"):
        return choice


def _singleton(value):
    """Unwrap nested one-element containers without coercing real answer lists."""
    while True:
        if isinstance(value, pd.DataFrame):
            values = value.to_numpy().reshape(-1).tolist()
        elif isinstance(value, pd.Series):
            values = value.to_numpy().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return value.item() if isinstance(value, np.generic) else value
        if len(values) != 1:
            return value
        value = values[0]


def result_to_answer(result, label):
    """Convert an executed reference-plan result to the label's public JSON type.

    The label is used only as a type contract. Its value is never inspected to
    select among candidate cells or headers.
    """
    if isinstance(label, list):
        if isinstance(result, pd.DataFrame):
            values = list(result.index)
        elif isinstance(result, pd.Series):
            values = (result.tolist() if isinstance(result.index, pd.RangeIndex)
                      else list(result.index))
        elif isinstance(result, np.ndarray):
            values = result.reshape(-1).tolist()
        elif isinstance(result, (list, tuple)):
            values = list(result)
        else:
            values = [result]
        return [_singleton(value) for value in values]
    if isinstance(label, dict):
        if isinstance(result, pd.DataFrame):
            if result.shape[0] == 1:
                result = result.iloc[0, :]
            elif result.shape[1] == 1:
                result = result.iloc[:, 0]
            else:
                raise ValueError("Cannot convert a matrix result to an answer object")
        if isinstance(result, pd.Series):
            keys_are_strings = all(isinstance(key, str) for key in label)
            return {
                str(_singleton(key)) if keys_are_strings else _singleton(key): _singleton(value)
                for key, value in result.items()
            }
        if isinstance(result, dict):
            return {_singleton(key): _singleton(value) for key, value in result.items()}
        raise ValueError("Cannot convert result to an answer object")
    value = _singleton(result)
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)):
        raise ValueError("Scalar label requires a singleton result")
    return value


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return {"tuple": [_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and np.isnan(value):
        return {"missing": True}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Cannot serialize plan result value: {type(value).__name__}")


def serialize_plan_result(result) -> dict:
    """Serialize an executed result without using or inferring its gold label."""
    if isinstance(result, pd.DataFrame):
        return {
            "kind": "dataframe",
            "index": [_json_value(value) for value in result.index.tolist()],
            "columns": [_json_value(value) for value in result.columns.tolist()],
            "values": [[_json_value(value) for value in row]
                       for row in result.to_numpy().tolist()],
        }
    if isinstance(result, pd.Series):
        return {
            "kind": "series",
            "index": [_json_value(value) for value in result.index.tolist()],
            "values": [_json_value(value) for value in result.tolist()],
        }
    return {"kind": "value", "value": _json_value(result)}
