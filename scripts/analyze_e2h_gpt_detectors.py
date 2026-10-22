"""Run the frozen call-free Section 49 GPT detector analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.evaluation.e2h_gpt_detector_analysis import (  # noqa: E402
    analyze_e2h_gpt_detectors,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/e2h_final_holdout_gpt4o_mini"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/e2h_gpt_detector_analysis.yaml"),
    )
    args = parser.parse_args()
    result = analyze_e2h_gpt_detectors(args.result_dir, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
