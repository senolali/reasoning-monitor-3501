"""ADWIN-defect sensitivity of the LIC-RQ-v2-SR replay (ADWIN_DEFECT_AUDIT.md section 4).

Runs the frozen replay unchanged except for the detectors that were the
defective historical ADWIN:
  - proposal_mapping CS and SS -> VARIANT ("adwin_v" or "cusum")
  - adwin_all (every metric)    -> "adwin_v" (the corrected generic baseline)
Output goes to results/adwin_sensitivity/lic_replay_<variant>/; the frozen replay
outputs are never touched. Patches are applied at import time so that spawned
worker processes inherit them.

  python scripts\\adwin_sensitivity_lic_replay.py adwin_v
  python scripts\\adwin_sensitivity_lic_replay.py cusum
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reasoning_monitor.evaluation import detector_development as dd  # noqa: E402
from reasoning_monitor.evaluation import stream_replay as sr  # noqa: E402

ENV = "ADWIN_SENSITIVITY_VARIANT"
_original = sr.strategy_detectors


def _patch(variant: str) -> None:
    sr.GRIDS["adwin_v"] = dd.DevelopmentSpec("adwin_v", "delta", sr.GRIDS["adwin"].values)

    def strategy_detectors() -> dict[str, dict[str, str]]:
        table = _original()
        table["proposal_mapping"] = {
            m: (variant if d == "adwin" else d) for m, d in table["proposal_mapping"].items()}
        table["adwin_all"] = {m: "adwin_v" for m in table["adwin_all"]}
        return table

    sr.strategy_detectors = strategy_detectors


if os.environ.get(ENV):
    _patch(os.environ[ENV])


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("adwin_v", "cusum"):
        raise SystemExit("usage: adwin_sensitivity_lic_replay.py {adwin_v|cusum} [--workers N]")
    variant = sys.argv.pop(1)
    os.environ[ENV] = variant
    _patch(variant)
    import replay_lost_conversation_v2_streams as replay

    output = ROOT / f"results/adwin_sensitivity/lic_replay_{variant}"
    sys.argv += ["--output", str(output)]
    replay.main()


if __name__ == "__main__":
    main()
