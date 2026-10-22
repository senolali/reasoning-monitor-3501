"""Run the preregistered call-free S1 CQ permutation analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.evaluation.permutation_controls import (  # noqa: E402
    analyze_s1_permutations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=21042)
    args = parser.parse_args()
    result = analyze_s1_permutations(
        args.result_dir,
        n_permutations=args.n_permutations,
        permutation_seed=args.permutation_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
