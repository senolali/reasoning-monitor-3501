"""Apply the frozen, call-free Section 57 answer-order calibration gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_answer_last_calibration import (  # noqa: E402
    assess_proposal_s1_answer_last_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="results/proposal_s1_answer_last_calibration",
    )
    parser.add_argument(
        "--gate",
        default="configs/proposal_s1_answer_last_calibration_gate.yaml",
    )
    args = parser.parse_args()
    result = assess_proposal_s1_answer_last_calibration(args.result_dir, args.gate)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

