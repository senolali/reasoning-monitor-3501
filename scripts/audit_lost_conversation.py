"""Audit LIC inputs and propose sample/call budgets, without any model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.lost_conversation_audit import (  # noqa: E402
    inside_root,
    run_audit,
    write_artifacts,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lost_conversation_v1_audit.yaml")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    config = inside_root(ROOT, args.config)
    artifacts = run_audit(ROOT, config)
    if args.write:
        settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        write_artifacts(inside_root(ROOT, settings["output_dir"]), artifacts)
    print(json.dumps(artifacts["assessment.json"], indent=2, allow_nan=False), flush=True)
    plan = artifacts["sample_options.json"]
    print(json.dumps({"eligible_tasks": plan["eligible_unmatched_candidate_tasks"],
                      "integration": plan["integration_budget"],
                      "options": [{k: v for k, v in option.items() if k != "science_task_ids"}
                                  for option in plan["options"]]}, indent=2), flush=True)
    print("No model calls. Options are provisional, not an execution authorization.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
