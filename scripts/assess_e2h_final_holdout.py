"""Apply the frozen call-free Section 47 gate to one completed track."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.evaluation.e2h_final_holdout import (  # noqa: E402
    FINAL_HOLDOUT_SPECS,
    assess_e2h_final_holdout,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=tuple(FINAL_HOLDOUT_SPECS))
    parser.add_argument("result_dir", nargs="?", type=Path)
    args = parser.parse_args()
    spec = FINAL_HOLDOUT_SPECS[args.track]
    result_dir = args.result_dir or Path(spec.result_dir)
    result = assess_e2h_final_holdout(result_dir, args.track)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
