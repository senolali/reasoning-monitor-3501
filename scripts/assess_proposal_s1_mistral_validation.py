"""Apply the frozen, call-free Section 62 ASU Mistral validation gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_answer_last_validation import (  # noqa: E402
    assess_proposal_s1_answer_last_validation,
)


def main() -> int:
    result = assess_proposal_s1_answer_last_validation(
        "results/proposal_s1_mistral_validation",
        "configs/proposal_s1_mistral_validation_gate.yaml",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
