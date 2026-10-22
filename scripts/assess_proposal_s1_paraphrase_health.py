"""Apply the frozen Section 66 gate without making model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_paraphrase_health import (  # noqa: E402
    assess_paraphrase_health,
)


def main() -> int:
    result = assess_paraphrase_health(
        ROOT / "results" / "proposal_s1_paraphrase_health_gpt_llama4",
        ROOT / "configs" / "proposal_s1_paraphrase_health_gpt_llama4_gate.yaml",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(not result["overall_gate_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
