"""Run the call-free Section 52 SuperGPQA pool audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.supergpqa_pool_audit import (  # noqa: E402
    audit_supergpqa_pool,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/supergpqa_pool_audit.yaml",
    )
    args = parser.parse_args()
    result = audit_supergpqa_pool(ROOT, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

