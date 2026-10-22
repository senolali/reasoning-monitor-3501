"""Call-free exploratory readiness-gap analysis for LIC-RQ-v2 integration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.lost_conversation_semantic_rescore import (  # noqa: E402
    plot_readiness_gap,
    readiness_gap,
    semantic_records,
    write_json,
)


def main() -> int:
    checkpoint = ROOT / "results/lost_conversation_v2_integration/data/records.jsonl"
    raw = checkpoint.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    rescored = semantic_records(records)
    frame, models = readiness_gap(rescored)
    output = ROOT / "results/lost_conversation_v2_integration/semantic_rescore/readiness"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "readiness_gap.csv", index=False)
    figures = plot_readiness_gap(frame, output)
    report = {
        "analysis": "LIC-RQ-v2 exploratory resolution-readiness gap",
        "no_model_calls": True,
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "definition": (
            "fresh-prefix primary answer readiness minus dialogue-history primary answer "
            "readiness at matched revealed information; +1 means dialogue history lags"
        ),
        "rqeval_metric": False,
        "warning_established": False,
        "models": models,
        "figures": figures,
        "next_use": (
            "Freeze as an auxiliary candidate before, and validate only in, the untouched "
            "48-task science sample."
        ),
    }
    write_json(output / "assessment.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

