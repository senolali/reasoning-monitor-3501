"""Audit Section 66 RS answers for semantic unit equivalence without calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_paraphrase_semantic_audit import (  # noqa: E402
    audit_paraphrase_semantics,
)


def main() -> int:
    result = audit_paraphrase_semantics(
        ROOT / "results" / "proposal_s1_paraphrase_health_gpt_llama4"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
