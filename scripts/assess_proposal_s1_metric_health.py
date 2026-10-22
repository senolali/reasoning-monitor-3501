"""Apply the frozen, call-free Section 64 metric-health gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_metric_health import (  # noqa: E402
    assess_metric_health,
)


def main() -> int:
    result = assess_metric_health(
        ROOT / "results" / "proposal_s1_metric_health_gpt_llama4",
        ROOT / "configs" / "proposal_s1_metric_health_gpt_llama4_gate.yaml",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["overall_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

