"""Diagnose the frozen failed DBQR Practice review without model calls."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results/dbqr_v1_practice_transport/data/records.jsonl"
OUTPUT = ROOT / "results/dbqr_v1_practice_transport/failure_diagnostic.json"


def main() -> int:
    rows = [
        json.loads(line)
        for line in RECORDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = collections.Counter(row["plan_error"] for row in rows)
    by_model = {
        model: dict(collections.Counter(
            row["plan_error"] for row in rows if row["model"] == model))
        for model in sorted({row["model"] for row in rows})
    }
    report = {
        "analysis": "dbqr_v1_failed_practice_transport_prompt_contract_diagnostic",
        "no_model_calls": True,
        "records": len(rows),
        "records_sha256": hashlib.sha256(RECORDS.read_bytes()).hexdigest(),
        "valid_plans": sum(bool(row["ok"]) for row in rows),
        "error_counts": dict(errors),
        "errors_by_model": by_model,
        "responses_with_marker_text": sum(
            "FINAL_PLAN_JSON:" in row["raw_text"] for row in rows),
        "responses_with_markdown_fence": sum("```" in row["raw_text"] for row in rows),
        "responses_with_numbered_reasoning": sum(
            any(line.lstrip().startswith("1.") for line in row["raw_text"].splitlines())
            for row in rows),
        "responses_mentioning_plan_version": sum(
            "DBQR-PLAN-v1" in row["raw_text"] for row in rows),
        "responses_mentioning_steps_key": sum(
            '"steps"' in row["raw_text"] for row in rows),
        "diagnosis": (
            "Shared prompt-contract failure: the paid prompt requested a final marker but "
            "did not expose the required DBQR-PLAN-v1 envelope, reference syntax, operation "
            "allowlist, or a valid example. The records are not evidence about plan-solving "
            "quality or early warning and cannot be reparsed into executable plans."
        ),
        "decision": (
            "Preserve the failed gate, keep the remaining 336 Practice calls locked, and "
            "require a separately approved minimal format-contract probe before any fresh "
            "Practice review."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
