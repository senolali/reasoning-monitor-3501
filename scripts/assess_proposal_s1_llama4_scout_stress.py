"""Apply the frozen, call-free Section 60 Llama-4 Scout stress gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_format_stress import (  # noqa: E402
    assess_proposal_s1_format_stress,
)


def main() -> int:
    result = assess_proposal_s1_format_stress(
        "results/proposal_s1_llama4_scout_stress",
        "configs/proposal_s1_llama4_scout_stress_gate.yaml",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
