"""Estimate the CDR-v1.2a exact-message retry without model calls."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "controlled_dialogue_v1_2_capacity_retry_asu.yaml"


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = {
        "models": len(config["models"]),
        "model_calls": int(config["expected_total_calls"]),
        "retry_max_tokens": int(config["retry_max_tokens"]),
        "max_output_tokens": int(config["maximum_output_token_ceiling"]),
        "reused_original_calls": 206,
        "currency_cost": "unknown (provider pricing/quota not inferred)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
