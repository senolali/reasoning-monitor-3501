"""Materialize the deterministic CDR-v1.4 replication bank."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.dialogue_streams import write_dialogue_bank  # noqa: E402
from reasoning_monitor.dialogue_replication import (  # noqa: E402
    generate_replication_bank,
    validate_replication_bank,
)

OUTPUT = ROOT / "configs" / "selections" / "controlled_dialogue_replication_v1.json"


def main() -> int:
    bank = generate_replication_bank()
    problems = validate_replication_bank(bank)
    if problems:
        raise ValueError(f"invalid replication bank: {problems}")
    write_dialogue_bank(OUTPUT, bank)
    report = {
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "dialogues": len(bank["dialogues"]),
        "turns": sum(len(dialogue["turns"]) for dialogue in bank["dialogues"]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
