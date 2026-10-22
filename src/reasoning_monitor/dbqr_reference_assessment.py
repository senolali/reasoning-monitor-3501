"""Call-free reproduction assessment for pinned DBQR reference plans."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import PurePosixPath

from reasoning_monitor.dbqr_answers import typed_answer_equal
from reasoning_monitor.dbqr_audit import MAX_BYTES, SOURCE_SHA256, restricted_load
from reasoning_monitor.dbqr_reference_interpreter import (
    ReferencePlanInterpreter,
    result_to_answer,
)


def assess_reference_plans(blob: bytes, expected_sha256: str = SOURCE_SHA256) -> dict:
    """Execute every Practice reference plan against its pinned query tables."""
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")

    rows = []
    interpreter = ReferencePlanInterpreter()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        question_names = sorted(
            name for name in members
            if PurePosixPath(name).parts[1:2] == ("questions",)
            and name.endswith(".json")
        )
        for question_name in question_names:
            question = json.load(archive.extractfile(members[question_name]))
            table_name = question_name.replace("/questions/", "/tables/").replace(
                ".json", ".pkl")
            tables, _globals = restricted_load(archive.extractfile(members[table_name]).read())
            row = {
                "qid": f"{PurePosixPath(question_name).parent.name}/"
                       f"{PurePosixPath(question_name).stem}",
                "section_id": question["sectionID"],
                "turn": question["questionID"],
                "reference_answer": question["answer"],
            }
            try:
                result, _environment = interpreter.execute(question["program"], tables)
                answer = result_to_answer(result, question["answer"])
                row.update({
                    "executed": True,
                    "interpreted_answer": answer,
                    "typed_match": typed_answer_equal(answer, question["answer"]),
                })
            except Exception as exc:  # assessment records failures instead of hiding them
                row.update({
                    "executed": False,
                    "typed_match": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            rows.append(row)

    section_counts = {}
    for section in sorted({row["section_id"] for row in rows}):
        selected = [row for row in rows if row["section_id"] == section]
        section_counts[str(section)] = {
            "questions": len(selected),
            "executed": sum(row["executed"] for row in selected),
            "matched": sum(row["typed_match"] for row in selected),
        }
    checks = {
        "expected_questions": len(rows) == 50,
        "all_plans_executed": bool(rows) and all(row["executed"] for row in rows),
        "all_labels_reproduced": bool(rows) and all(row["typed_match"] for row in rows),
    }
    return {
        "protocol_id": "DBQR-RQ-v1",
        "stage": "practice_reference_plan_reproduction",
        "no_model_calls": True,
        "source_sha256": digest,
        "questions": len(rows),
        "plans_executed": sum(row["executed"] for row in rows),
        "labels_reproduced": sum(row["typed_match"] for row in rows),
        "section_counts": section_counts,
        "checks": checks,
        "pass": all(checks.values()),
        "rows": rows,
    }
