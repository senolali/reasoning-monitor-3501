"""Run the preregistered call-free detector-development analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.evaluation.detector_development import (  # noqa: E402
    develop_s1_detectors,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--n-calibration", type=int, default=1000)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--calibration-seed", type=int, default=51042)
    parser.add_argument("--validation-seed", type=int, default=61042)
    args = parser.parse_args()
    result = develop_s1_detectors(
        args.result_dir,
        n_calibration=args.n_calibration,
        n_validation=args.n_validation,
        calibration_seed=args.calibration_seed,
        validation_seed=args.validation_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
