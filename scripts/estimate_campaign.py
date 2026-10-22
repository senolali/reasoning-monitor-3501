"""Estimate model-call and output-token ceilings from an experiment YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def estimate(raw: dict, root: str | Path | None = None) -> dict[str, int]:
    seeds = raw.get("seeds") or [raw.get("seed", 42)]
    scenarios = raw.get("scenarios") or []
    models = raw.get("models") or []
    windows = raw.get("window_sizes") or [20]
    cap = raw.get("max_queries_per_scenario")
    per_regime = raw.get("queries_per_regime")
    proposal_per_dataset = raw.get("proposal_queries_per_dataset")
    regime_counts = {"ST2": 3, "ST6": 3}
    if cap:
        queries = len(scenarios) * int(cap)
    elif raw.get("stream_design") == "proposal_multidataset":
        queries = len(scenarios) * 3 * int(proposal_per_dataset or 50)
    elif raw.get("stream_design") == "proposal_answer_order_calibration":
        queries = len(scenarios) * 2 * int(proposal_per_dataset or 10)
    elif raw.get("stream_design") == "proposal_frozen_selection":
        queries = len(scenarios) * int(raw.get("frozen_selection_size") or 1)
    elif raw.get("stream_design") == "controlled_trajectory":
        bank_path = Path(str(raw.get("controlled_trajectory_bank") or ""))
        if not bank_path.is_absolute():
            bank_path = Path(root or Path.cwd()) / bank_path
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        split = raw.get("controlled_trajectory_split")
        queries = sum(item.get("split") == split for item in bank.get("items", []))
    elif per_regime:
        queries = sum(regime_counts.get(name, 1) * int(per_regime) for name in scenarios)
    else:
        queries = len(scenarios) * 150
    calls_per_query = int(raw.get("k_repeats", 5)) + int(raw.get("l_paraphrases", 3))
    cells = len(seeds) * len(models) * len(windows)
    calls = cells * queries * calls_per_query
    max_tokens = max((int(model.get("max_tokens") or 0) for model in models), default=0)
    return {
        "seeds": len(seeds),
        "scenarios": len(scenarios),
        "models": len(models),
        "windows": len(windows),
        "queries_per_model_seed_window": queries,
        "calls_per_query": calls_per_query,
        "model_calls": calls,
        "max_output_tokens": calls * max_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    project_root = (
        args.config.resolve().parent.parent
        if args.config.resolve().parent.name == "configs"
        else Path.cwd()
    )
    result = estimate(raw, root=project_root)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("currency_cost: unknown (provider pricing/quota not inferred)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
