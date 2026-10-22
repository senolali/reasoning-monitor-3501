"""Explicit model-facing contract for DBQR declarative plans."""

from __future__ import annotations

import hashlib

PROMPT_CONTRACT_VERSION = "DBQR-PROMPT-v1.1"

SYSTEM_PROMPT = """You produce a declarative plan for the current DBQR table question.
The local evaluator, not you, executes the plan against the aliased table.
Do not guess table values or directly answer the question.

OUTPUT FORMAT (strict):
- Write one or more reasoning lines. Every non-empty reasoning line must be exactly
  numbered as `1. ...`, `2. ...`, and so on. Do not write a heading or preamble.
- The last non-empty line must start exactly `FINAL_PLAN_JSON: ` followed by one
  complete JSON object on that same line.
- Do not use Markdown, code fences, comments, NaN, Infinity, or duplicate JSON keys.
- The JSON object must have exactly `version` and `steps`, with version
  `DBQR-PLAN-v1`.

PLAN SHAPE:
- A table-read step is exactly {"id":"s1","op":"query","table":"table_1"}.
- Every later step is exactly
  {"id":"sN","op":"OP","args":[...],"kwargs":{...}}.
- Step ids are sequential strings s1, s2, ... . Refer to an earlier result only as
  {"ref":"sN"}. Never refer forward. The final step is the returned result.
- Use only aliases present in COMPACT_TABLE_SCHEMA_JSON and only operations below.

ALLOWLISTED OPERATIONS AND PARAMETERS:
loc(df, idx=":", columns=":") selects labels.
loc_filter(df, axis="rows", operation="all|any",
  condition="equal|gt|lt|ge|le", source="custom", val=null, step=null,
  idx=":", columns=":") filters by a condition.
merge(df, dfs, axis="rows|columns", ignore_index=false,
  replace_columns=true) concatenates tables.
intersect(df, dfs, axis="rows", level=null) keeps shared labels.
headers(df, axis="rows|columns", level=null) returns axis labels.
abs(df) takes absolute values.
sum(df, axis="rows|columns", level=null, keep_col=false) aggregates.
mean(df, axis="rows|columns", level=null, keep_col=false, skipna=true) averages.
diff(df, axis="rows|columns") computes adjacent differences.
div(df, axis="rows|columns", keep_col=false) computes adjacent ratios.
sort(df, by=null, axis="rows|columns", ascending=false) sorts values.
k_end(df, k, direction="first|last", axis="rows|columns", reverse=false)
  selects an end after sorting.
compare(df, axis="rows|columns", greater="greater", lower="lower",
  equal="equal") labels adjacent comparisons.
count(df, axis="rows|columns") counts labels.
cond(df, condition="equal|gt|lt|ge|le|ne", cond_source="custom",
  cond_val=null, cond_step=null, target_source="custom", target_val=null,
  target_step=null) conditionally replaces cells.
const_add(df, idx=":", columns=":", source="custom", val=null, step=null,
  table=null) adds a constant or referenced result.
const_mul(df, idx=":", columns=":", source="custom", val="1", step=null,
  table=null) multiplies by a constant or referenced result.
choices(df=null, choice="yes") returns a literal choice.

For operation inputs, put the main prior result in args, for example
"args":[{"ref":"s1"}]. Put every named parameter in kwargs. Omitted optional
parameters use the defaults above. JSON null/true/false must be lowercase.

VALID FORMAT EXAMPLE (illustrative only; adapt operations to the question):
1. Read the supplied aliased query-result table.
2. Aggregate its rows and return the resulting typed value.
FINAL_PLAN_JSON: {"version":"DBQR-PLAN-v1","steps":[{"id":"s1","op":"query","table":"table_1"},{"id":"s2","op":"sum","args":[{"ref":"s1"}],"kwargs":{"axis":"rows","level":null,"keep_col":false}}]}
"""


def prompt_contract_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
