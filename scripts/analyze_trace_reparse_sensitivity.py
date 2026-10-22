"""Run the frozen Section 51 call-free GPT trace-reparse sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.trace_reparse_sensitivity import (  # noqa: E402
    run_trace_reparse_sensitivity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results" / "e2h_final_holdout_gpt4o_mini",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "e2h_gpt_trace_reparse_sensitivity.yaml",
    )
    args = parser.parse_args()
    result = run_trace_reparse_sensitivity(args.result_dir, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
