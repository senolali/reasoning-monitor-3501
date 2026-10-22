"""Run the call-free Section 63 Mistral format sensitivity audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.proposal_s1_mistral_format_audit import (  # noqa: E402
    audit_mistral_format,
)


def main() -> int:
    result = audit_mistral_format(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(
        not result["source_sha256_matches"]
        or not result["stream_sequence_matches"]
    )


if __name__ == "__main__":
    raise SystemExit(main())

