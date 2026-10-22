"""Estimate the fresh CDR-v1.1 pilot without issuing model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_1_pilot_asu.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = {
        "dialogues": len(config["selected_dialogue_ids"]),
        "turns_per_dialogue": int(config["turns_per_dialogue"]),
        "conditions": len(config["conditions"]),
        "models": len(config["models"]),
        "calls_per_turn": 1 + int(config["l_paraphrases"]),
        "model_calls": int(config["expected_total_calls"]),
        "max_output_tokens": int(config["maximum_output_token_ceiling"]),
        "currency_cost": "unknown (provider pricing/quota not inferred)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

