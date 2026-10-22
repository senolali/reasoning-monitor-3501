"""Generate the deterministic CDR-v1 longitudinal dialogue bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.dialogue_streams import (  # noqa: E402
    dialogue_bank_sha256,
    generate_dialogue_bank,
    validate_dialogue_bank,
    write_dialogue_bank,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="configs/selections/controlled_dialogue_stream_v1.json",
    )
    args = parser.parse_args()
    bank = generate_dialogue_bank()
    problems = validate_dialogue_bank(bank)
    if problems:
        raise ValueError(f"dialogue bank validation failed: {problems[:10]}")
    path = ROOT / args.output
    write_dialogue_bank(path, bank)
    report = {
        "path": str(path.relative_to(ROOT)),
        "sha256": dialogue_bank_sha256(bank),
        "dialogues": len(bank["dialogues"]),
        "turns": sum(len(dialogue["turns"]) for dialogue in bank["dialogues"]),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
