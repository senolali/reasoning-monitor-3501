"""Generate the terminal Section 69 evidence synthesis without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.final_evidence_synthesis import (  # noqa: E402
    synthesize_final_evidence,
)


def main() -> int:
    result = synthesize_final_evidence(
        ROOT,
        ROOT / "configs" / "final_evidence_sources.yaml",
        ROOT / "results" / "final_evidence_synthesis",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["synthesis_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
