"""Estimate the frozen CDR-v1.3 validation without model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "controlled_dialogue_v1_3_primary_validation_asu.yaml"
)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = {
        "dialogues": len(config["selected_dialogue_ids"]),
        "turns_per_dialogue": int(config["turns_per_dialogue"]),
        "conditions": len(config["conditions"]),
        "models": len(config["models"]),
        "model_calls": int(config["expected_total_calls"]),
        "max_output_tokens_per_call": int(config["max_output_tokens_per_call"]),
        "max_output_tokens": int(config["maximum_output_token_ceiling"]),
        "conditional_self_history_calls_deferred": 216,
        "currency_cost": "unknown (provider pricing/quota not inferred)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

