"""Estimate the frozen CDR-v1.4 replication without model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "controlled_dialogue_v1_4_replication_asu.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    report = {
        "protocol_id": config["protocol_id"],
        "models": len(config["models"]),
        "dialogues": len(config["selected_dialogue_ids"]),
        "conditions": len(config["conditions"]),
        "turns_per_dialogue": int(config["turns_per_dialogue"]),
        "calls_per_model": int(config["expected_calls_per_model"]),
        "total_calls": int(config["expected_total_calls"]),
        "max_output_tokens_per_call": int(config["max_output_tokens_per_call"]),
        "maximum_output_token_ceiling": int(
            config["maximum_output_token_ceiling"]
        ),
        "paid_calls_authorized": False,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

