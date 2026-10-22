"""Audit compact DBQR Practice schemas without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_schema_assessment import assess_compact_schemas  # noqa: E402


def main() -> int:
    report = assess_compact_schemas((ROOT / "data/dbqr/practice.tar.gz").read_bytes())
    output = ROOT / "results/dbqr_v1_audit/compact_schema_assessment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: report[key] for key in (
        "protocol_id", "stage", "no_model_calls", "questions",
        "maximum_schema_characters", "checks", "pass")}
    print(json.dumps(summary, indent=2))
    print(f"report: {output.relative_to(ROOT)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
