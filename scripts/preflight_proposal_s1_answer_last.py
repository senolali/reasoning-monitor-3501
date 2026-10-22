"""Call-free integration preflight for the Section 57 answer-order calibration."""

from __future__ import annotations

import argparse
import hashlib
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

EXPECTED_MODELS = ("gpt4o_mini", "llama3_8b")
GATE_PATH = ROOT / "configs" / "proposal_s1_answer_last_calibration_gate.yaml"


def _sequence_sha256(rows: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{dataset}|{qid}" for dataset, qid in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_preflight(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
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

    problems: list[str] = []
    counts = Counter(query.dataset for query in stream)
    sequence = [(query.dataset, query.qid) for query in stream]
    sequence_hash = _sequence_sha256(sequence)
    if len(stream) != 20 or len({query.qid for query in stream}) != 20:
        problems.append("stream must contain 20 unique questions")
    if counts != {"gsm8k": 10, "math": 10}:
        problems.append(f"unexpected dataset counts: {dict(counts)}")
    if sequence_hash != gate["expected_stream_sequence_sha256"]:
        problems.append("stream sequence differs from the frozen gate hash")
    if cfg.k_repeats != 1 or cfg.l_paraphrases != 0:
        problems.append("answer-order calibration must use K=1 and L=0")
    if cfg.use_rqeval:
        problems.append("answer-order calibration must not load unused rqeval scorers")
    if not cfg.resume:
        problems.append("resume must be enabled")
    if cfg.allow_synthetic_datasets:
        problems.append("synthetic dataset fallback must be disabled")
    if cfg.dataset_prompt_mode != "answer_last_bounded":
        problems.append("dataset_prompt_mode must be answer_last_bounded")
    for query in stream:
        if "at most six short numbered reasoning steps" not in query.prompt:
            problems.append(f"bounded-reasoning instruction missing for {query.qid}")
            break
        if "final output line must have exactly this form: FINAL_ANSWER" not in query.prompt:
            problems.append(f"answer-last instruction missing for {query.qid}")
            break

    model_names = tuple(model.get("name") for model in cfg.models)
    if model_names != EXPECTED_MODELS or list(model_names) != gate["expected_models"]:
        problems.append(f"unexpected models: {model_names}")
    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    for model in cfg.models:
        if int(model.get("max_tokens") or 0) != int(gate["max_output_tokens"]):
            problems.append(f"{model.get('name')} max_tokens differs from the gate")
        if model.get("request_options") != expected_options:
            problems.append(f"{model.get('name')} gateway controls differ from the frozen payload")
        if "FINAL_ANSWER line last" not in str(model.get("system_prompt", "")):
            problems.append(f"{model.get('name')} system prompt lacks answer-last control")
    if cfg.seeds != [int(gate["expected_seed"])]:
        problems.append("config seed differs from the frozen gate")
    if cfg.window_sizes != [int(gate["expected_window_size"])]:
        problems.append("config window size differs from the frozen gate")
    if cfg.proposal_queries_per_dataset != int(gate["expected_queries_per_dataset"]):
        problems.append("config dataset sample size differs from the frozen gate")

    return {
        "no_model_calls": True,
        "config": str(config_path.relative_to(ROOT)),
        "scenario": spec.name,
        "queries_per_model": len(stream),
        "planned_model_calls": len(stream) * len(cfg.models),
        "unique_queries": len({query.qid for query in stream}),
        "dataset_counts": dict(counts),
        "dataset_sources": {name: dataset.source for name, dataset in datasets.items()},
        "dataset_hashes": {name: dataset.content_hash for name, dataset in datasets.items()},
        "stream_sequence_sha256": sequence_hash,
        "change_points": spec.change_points,
        "problems": problems,
        "pass": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/proposal_s1_answer_last_calibration_asu.yaml",
    )
    args = parser.parse_args()
    report = run_preflight(args.config)
    print(json.dumps(report, indent=2))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())

