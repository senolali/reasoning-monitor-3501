"""Run and assess the DBQR Practice fixture replay without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_fixture_assessment import assess_fixture_replay  # noqa: E402
from reasoning_monitor.dbqr_fixture_replay import load_practice_fixture_cases  # noqa: E402
from reasoning_monitor.progress import ProgressReporter, count_jsonl_records  # noqa: E402


def main() -> int:
    cases, source_sha256 = load_practice_fixture_cases(
        (ROOT / "data/dbqr/practice.tar.gz").read_bytes())
    checkpoint = ROOT / "results/dbqr_v1_fixture/data/records_DBQR_fixture.jsonl"
    completed = count_jsonl_records(checkpoint)
    progress = ProgressReporter(total=100, label="DBQR fixture", completed=completed)
    progress.print(completed)

    def update(_record):
        nonlocal completed
        completed += 1
        progress.print(completed, final=completed == 100)

    from reasoning_monitor.dbqr_fixture_replay import run_fixture_replay
    run_fixture_replay(
        cases,
        source_sha256=source_sha256,
        checkpoint_path=checkpoint,
        checkpoint_callback=update,
    )
    if completed == count_jsonl_records(checkpoint) and completed == 100:
        progress.print(completed, final=True)
    report = assess_fixture_replay(cases, source_sha256, checkpoint)
    output = ROOT / "results/dbqr_v1_fixture/assessment.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output.relative_to(ROOT)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
