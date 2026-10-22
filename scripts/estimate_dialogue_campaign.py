"""Estimate the frozen CDR-v1 pilot without issuing model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/controlled_dialogue_v1_pilot_asu.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    dialogues = len(config["selected_dialogue_ids"])
    turns = int(config["turns_per_dialogue"])
    conditions = len(config["conditions"])
    models = len(config["models"])
    calls_per_turn = int(config["k_repeats"]) + int(config["l_paraphrases"])
    calls = dialogues * turns * conditions * models * calls_per_turn
    report = {
        "dialogues": dialogues,
        "turns_per_dialogue": turns,
        "conditions": conditions,
        "models": models,
        "calls_per_turn": calls_per_turn,
        "model_calls": calls,
        "max_output_tokens": calls * int(config["max_output_tokens_per_call"]),
        "currency_cost": "unknown (provider pricing/quota not inferred)",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
