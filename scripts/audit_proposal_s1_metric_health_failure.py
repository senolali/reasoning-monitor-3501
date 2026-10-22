"""Run the call-free Section 65 diagnostic of the Section 64 failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_metric_health_audit import (  # noqa: E402
    audit_metric_health_failure,
)


def main() -> int:
    result = audit_metric_health_failure(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_source_hashes_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

