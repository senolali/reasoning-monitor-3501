"""Estimate the CDR-v1.2 user-only mechanism arm without model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_user_only_probe_asu.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = {
        "dialogues": len(config["selected_dialogue_ids"]),
        "turns_per_dialogue": int(config["turns_per_dialogue"]),
        "conditions": 1,
        "models": len(config["models"]),
        "model_calls": int(config["expected_total_calls"]),
        "max_output_tokens": int(config["maximum_output_token_ceiling"]),
        "reused_existing_calls": 144,
        "currency_cost": "unknown (provider pricing/quota not inferred)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

