"""Call-free integration preflight for the frozen SuperGPQA pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.loaders import load_all  # noqa: E402
from reasoning_monitor.experiment import (  # noqa: E402
    ExperimentConfig,
    _configured_stream,
    _validate_scenarios,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/supergpqa_gpt4o_mini_pilot_asu.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    _validate_scenarios(cfg.scenarios, cfg.stream_design)
    datasets = load_all(
        cfg.datasets,
        limit_per_dataset=cfg.items_per_dataset,
        seed=cfg.seed,
        allow_synthetic=cfg.allow_synthetic_datasets,
        selection_manifest=cfg.dataset_selection_manifest,
        selection_split=cfg.dataset_selection_split,
        prompt_mode=cfg.dataset_prompt_mode,
    )
    spec, stream, _plan = _configured_stream(
        cfg, cfg.scenarios[0], datasets, cfg.seeds[0]
    )
    counts = Counter(query.difficulty for query in stream)
    problems = []
    if len(stream) != 60 or len({query.qid for query in stream}) != 60:
        problems.append("stream must contain 60 unique questions")
    if counts != {"easy": 20, "medium": 20, "hard": 20}:
        problems.append(f"unexpected tier counts: {dict(counts)}")
    if any(query.gold_answer not in "ABCDEFGHIJ" for query in stream):
        problems.append("gold answers must be one letter from A through J")
    if any("FINAL_CHOICE: X" not in query.prompt for query in stream):
        problems.append("prompt final-choice instruction is missing")
    if cfg.dataset_prompt_mode == "choice_first" and any(
        "first output line must be exactly: FINAL_CHOICE: X" not in query.prompt
        for query in stream
    ):
        problems.append("answer-first instruction is missing")

    report = {
        "no_model_calls": True,
        "config": str(config_path),
        "dataset_source": datasets["supergpqa"].source,
        "dataset_hash": datasets["supergpqa"].content_hash,
        "scenario": spec.name,
        "queries": len(stream),
        "unique_queries": len({query.qid for query in stream}),
        "tier_counts": dict(counts),
        "change_points": spec.change_points,
        "problems": problems,
        "pass": not problems,
    }
    print(json.dumps(report, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
