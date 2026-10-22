"""Run the call-free accepted-proposal S1 answer observability audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_parsing_audit import (  # noqa: E402
    audit_proposal_s1_parsing,
)


def main() -> int:
    result = audit_proposal_s1_parsing(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
