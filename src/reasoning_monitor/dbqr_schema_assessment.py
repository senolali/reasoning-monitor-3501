"""Call-free leakage and boundedness audit for DBQR compact schemas."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath

from reasoning_monitor.dbqr_audit import MAX_BYTES, SOURCE_SHA256, restricted_load
from reasoning_monitor.dbqr_schema import MAX_SCHEMA_CHARS, compact_schema


def _resolved_strings(value):
    if isinstance(value, str) and len(value) >= 4:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _resolved_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _resolved_strings(item)


def assess_compact_schemas(blob: bytes, expected_sha256: str = SOURCE_SHA256) -> dict:
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")
    rows = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        questions = sorted(name for name in members if "/questions/" in name)
        for question_name in questions:
            question = json.load(archive.extractfile(members[question_name]))
            table_name = question_name.replace("/questions/", "/tables/").replace(
                ".json", ".pkl")
            tables, _globals = restricted_load(archive.extractfile(members[table_name]).read())
            schema, aliases = compact_schema(tables)
            repeated, repeated_aliases = compact_schema(tables)
            rendered = json.dumps(schema, ensure_ascii=True, sort_keys=True)
            resolved = sorted(set(_resolved_strings(question.get("vars", {}))))
            leaks = [value for value in resolved if value.casefold() in rendered.casefold()]
            rows.append({
                "qid": f"{PurePosixPath(question_name).parent.name}/"
                       f"{PurePosixPath(question_name).stem}",
                "tables": len(tables),
                "schema_characters": len(rendered),
                "deterministic": schema == repeated,
                "alias_order_deterministic": list(aliases) == list(repeated_aliases),
                "private_keys_absent": all(key not in rendered for key in tables),
                "program_targets_absent": "var_q" not in rendered,
                "resolved_string_leaks": leaks,
            })
    checks = {
        "expected_questions": len(rows) == 50,
        "deterministic": all(row["deterministic"] and row["alias_order_deterministic"]
                             for row in rows),
        "bounded": bool(rows) and max(row["schema_characters"] for row in rows)
                   <= MAX_SCHEMA_CHARS,
        "private_keys_absent": all(row["private_keys_absent"] for row in rows),
        "program_targets_absent": all(row["program_targets_absent"] for row in rows),
        "resolved_strings_absent": all(not row["resolved_string_leaks"] for row in rows),
    }
    return {
        "protocol_id": "DBQR-RQ-v1",
        "stage": "practice_compact_schema_audit",
        "no_model_calls": True,
        "source_sha256": digest,
        "questions": len(rows),
        "maximum_schema_characters": max(row["schema_characters"] for row in rows),
        "checks": checks,
        "pass": all(checks.values()),
        "rows": rows,
    }
