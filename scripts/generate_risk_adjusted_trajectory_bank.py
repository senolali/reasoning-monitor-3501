"""Generate the deterministic RAT-v1 controlled problem bank without calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.controlled_trajectories import (  # noqa: E402
    bank_sha256,
    generate_controlled_bank,
    write_controlled_bank,
)


def main() -> int:
    output = ROOT / "configs" / "selections" / "risk_adjusted_trajectory_v1_items.json"
    bank = generate_controlled_bank()
    write_controlled_bank(output, bank)
    report = {
        "protocol_id": bank["protocol_id"],
        "path": str(output.relative_to(ROOT)),
        "families": len({item["family_id"] for item in bank["items"]}),
        "items": len(bank["items"]),
        "sha256": bank_sha256(bank),
        "no_model_calls": True,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
