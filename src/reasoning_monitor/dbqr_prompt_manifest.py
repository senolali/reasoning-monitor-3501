"""Sanitized DBQR prompt manifests for label-blind acquisition runners."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from pathlib import PurePosixPath

from reasoning_monitor.dbqr_audit import restricted_load
from reasoning_monitor.dbqr_schema import compact_schema

ALLOWED_CASE_KEYS = {
    "dialogue_id", "turn", "qid", "question", "schema", "table_member",
}
FORBIDDEN_KEYS = {"answer", "program", "vars", "queries", "gold", "reference_answer"}


def _natural(text: str):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _contains_forbidden_key(value) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_KEYS & set(value)) or any(
            _contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def build_prompt_manifest(blob: bytes, *, split: str, expected_sha256: str,
                          maximum_bytes: int) -> dict:
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > maximum_bytes or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch")
    cases = []
    root = split
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        names = sorted((name for name in members if f"{root}/questions/" in name), key=_natural)
        for name in names:
            question = json.load(archive.extractfile(members[name]))
            table_name = name.replace("/questions/", "/tables/").replace(".json", ".pkl")
            tables, _globals = restricted_load(archive.extractfile(members[table_name]).read())
            schema, _aliases = compact_schema(tables)
            path = PurePosixPath(name)
            cases.append({
                "dialogue_id": path.parent.name,
                "turn": int(question["questionID"]),
                "qid": f"{path.parent.name}/{path.stem}",
                "question": str(question["question"]),
                "schema": schema,
                "table_member": table_name,
            })
    body = {
        "protocol_id": "DBQR-RQ-v1",
        "split": split,
        "source_sha256": digest,
        "label_blind": True,
        "cases": cases,
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    validate_prompt_manifest(body)
    return body


def validate_prompt_manifest(manifest: dict) -> None:
    if set(manifest) != {
        "protocol_id", "split", "source_sha256", "label_blind", "cases",
        "manifest_sha256",
    } or manifest.get("protocol_id") != "DBQR-RQ-v1" or manifest.get("label_blind") is not True:
        raise ValueError("Invalid prompt manifest envelope")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Prompt manifest has no cases")
    if any(set(case) != ALLOWED_CASE_KEYS or _contains_forbidden_key(case) for case in cases):
        raise ValueError("Prompt manifest contains forbidden or unknown fields")
    unsigned = dict(manifest)
    expected = unsigned.pop("manifest_sha256")
    actual = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "ascii")
    ).hexdigest()
    if actual != expected:
        raise ValueError("Prompt manifest hash mismatch")


def load_prompt_manifest(path) -> dict:
    """Load the only dataset metadata format accepted by acquisition code."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_prompt_manifest(manifest)
    return manifest
