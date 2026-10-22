"""Assessment for the call-free DBQR Practice fixture replay."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from reasoning_monitor.dbqr_fixture_replay import (
    CONDITIONS,
    PLAN_MARKER,
    run_fixture_replay,
)


def assess_fixture_replay(cases, source_sha256: str, checkpoint_path: str | Path) -> dict:
    checkpoint = Path(checkpoint_path)
    first = run_fixture_replay(
        cases, source_sha256=source_sha256, checkpoint_path=checkpoint)
    resume_callbacks = []
    resumed = run_fixture_replay(
        cases,
        source_sha256=source_sha256,
        checkpoint_path=checkpoint,
        checkpoint_callback=resume_callbacks.append,
    )
    counts = Counter((record["dialogue_id"], record["condition"]) for record in resumed)
    branch_transport = True
    for record in resumed:
        prior_assistants = [
            message["content"] for message in record["messages"][:-1]
            if message["role"] == "assistant"
        ]
        if record["turn"] > 1:
            if record["condition"] == "self_history":
                branch_transport &= all(PLAN_MARKER in text for text in prior_assistants)
            else:
                branch_transport &= all(
                    PLAN_MARKER not in text and "FINAL_ANSWER_JSON:" in text
                    for text in prior_assistants)
    checks = {
        "expected_records": len(resumed) == 100,
        "all_dialogue_condition_cells_complete": (
            len(counts) == 5 * len(CONDITIONS) and set(counts.values()) == {10}),
        "all_fixture_plans_execute_correctly": all(record["correct"] for record in resumed),
        "no_model_calls": all(record["no_model_calls"] for record in resumed),
        "history_branches_isolated": branch_transport,
        "resume_reuses_every_record": first == resumed and not resume_callbacks,
    }
    return {
        "protocol_id": "DBQR-RQ-v1",
        "stage": "practice_call_free_fixture_replay",
        "no_model_calls": True,
        "source_sha256": source_sha256,
        "records": len(resumed),
        "dialogues": len({record["dialogue_id"] for record in resumed}),
        "conditions": list(CONDITIONS),
        "correct_records": sum(record["correct"] for record in resumed),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checks": checks,
        "pass": all(checks.values()),
    }
