"""Run the call-free RAT-v1 structural and freeze preflight."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.risk_adjusted_trajectory_preflight import (  # noqa: E402
    run_preflight,
)


def main() -> int:
    result = run_preflight(
        ROOT,
        ROOT / "configs" / "risk_adjusted_trajectory_v1.yaml",
        ROOT / "configs" / "risk_adjusted_trajectory_v1_gate.yaml",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
