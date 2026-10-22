"""Run call-free stationary and shuffled controls on a completed S1 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.evaluation.order_controls import (  # noqa: E402
    analyze_s1_order_controls,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--control-seed", type=int, default=10042)
    args = parser.parse_args()
    result = analyze_s1_order_controls(args.result_dir, args.control_seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
