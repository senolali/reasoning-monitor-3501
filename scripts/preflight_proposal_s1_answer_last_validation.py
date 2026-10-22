"""Call-free integration preflight for full answer-last S1 validations."""

from __future__ import annotations

import argparse
import csv
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

SOURCE_STREAM = (
    ROOT
    / "results"
    / "proposal_s1_kl_validation_gpt_llama"
    / "data"
    / "stream_S1_seed42.csv"
)
GATE_PATH = ROOT / "configs" / "proposal_s1_answer_last_validation_gate.yaml"


def stream_sequence_sha256(rows: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{dataset}|{qid}" for dataset, qid in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_preflight(
    config_path: str | Path,
    gate_path: str | Path = GATE_PATH,
) -> dict:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate_path = Path(gate_path)
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    _validate_scenarios(cfg.scenarios, cfg.stream_design)
    datasets = load_all(
        cfg.datasets,
        limit_per_dataset=cfg.items_per_dataset,
        seed=cfg.seed,
        allow_synthetic=cfg.allow_synthetic_datasets,
        prompt_mode=cfg.dataset_prompt_mode,
    )
    spec, stream, _plan = _configured_stream(
        cfg,
        cfg.scenarios[0],
        datasets,
        cfg.seeds[0],
    )

    problems: list[str] = []
    counts = Counter(query.dataset for query in stream)
    if len(stream) != 90 or len({query.qid for query in stream}) != 90:
        problems.append("stream must contain 90 unique questions")
    if counts != {"gsm8k": 30, "arc_challenge": 30, "math": 30}:
        problems.append(f"unexpected dataset counts: {dict(counts)}")
    if cfg.k_repeats != 1 or cfg.l_paraphrases != 0:
        problems.append("full answer-last validation must use K=1 and L=0")
    if cfg.use_rqeval:
        problems.append("CQ-only validation must not load unused rqeval scorers")
    if not cfg.resume:
        problems.append("resume must be enabled")
    if cfg.allow_synthetic_datasets:
        problems.append("synthetic dataset fallback must be disabled")
    expected_prompt_mode = str(
        gate.get("expected_prompt_mode", "answer_last_bounded")
    )
    if cfg.dataset_prompt_mode != expected_prompt_mode:
        problems.append(f"dataset_prompt_mode must be {expected_prompt_mode}")

    for query in stream:
        if expected_prompt_mode == "answer_last_stress":
            marker_name = (
                "FINAL_CHOICE"
                if query.dataset == "arc_challenge"
                else "FINAL_ANSWER"
            )
            final_form = (
                "FINAL_CHOICE: X"
                if query.dataset == "arc_challenge"
                else "FINAL_ANSWER: <answer>"
            )
            required = (
                "no more than six numbered lines",
                "Each numbered line must contain exactly one short sentence.",
                "Do not continue reasoning after line 6.",
                "best answer even if uncertain",
                f"plain text exactly in this form: {final_form}",
                f"Do not put the {marker_name} line inside LaTeX",
            )
            terminal = f"Do not write anything after the {marker_name} line."
            if any(fragment not in query.prompt for fragment in required):
                problems.append(f"strict stress instruction missing for {query.qid}")
                break
        else:
            if "at most six short numbered reasoning steps" not in query.prompt:
                problems.append(
                    f"bounded-reasoning instruction missing for {query.qid}"
                )
                break
            if query.dataset == "arc_challenge":
                marker = "Your final output line must be exactly: FINAL_CHOICE: X"
                terminal = "Do not write anything after the FINAL_CHOICE line."
            else:
                marker = (
                    "Your final output line must have exactly this form: "
                    "FINAL_ANSWER: <answer>"
                )
                terminal = "Do not write anything after the FINAL_ANSWER line."
            if marker not in query.prompt:
                problems.append(f"answer-last instruction missing for {query.qid}")
                break
        if not query.prompt.endswith(terminal):
            problems.append(f"answer-last instruction missing for {query.qid}")
            break

    model_names = tuple(model.get("name") for model in cfg.models)
    expected_models = tuple(gate["expected_models"])
    if model_names != expected_models:
        problems.append(f"expected models {expected_models}, found {model_names}")
    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    for model in cfg.models:
        if int(model.get("max_tokens") or 0) != int(gate["max_output_tokens"]):
            problems.append(f"{model.get('name')} max_tokens differs from the gate")
        if model.get("request_options") != expected_options:
            problems.append(
                f"{model.get('name')} gateway controls differ from the frozen payload"
            )
        system_prompt = str(model.get("system_prompt", ""))
        if expected_prompt_mode == "answer_last_stress":
            if "six numbered reasoning lines" not in system_prompt:
                problems.append(
                    f"{model.get('name')} lacks the line-budget instruction"
                )
            if "plain-text FINAL_ANSWER or FINAL_CHOICE line last" not in system_prompt:
                problems.append(
                    f"{model.get('name')} lacks the plain-text final marker"
                )
        elif "FINAL_ANSWER or FINAL_CHOICE line last" not in system_prompt:
            problems.append(
                f"{model.get('name')} system prompt lacks the answer-last contract"
            )
        expected_provider_model = gate.get("expected_provider_model")
        if expected_provider_model and model.get("model") != expected_provider_model:
            problems.append(
                f"{model.get('name')} provider model differs from the frozen gate"
            )

    with SOURCE_STREAM.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    source_sequence = [(row["dataset"], row["qid"]) for row in source_rows]
    current_sequence = [(query.dataset, query.qid) for query in stream]
    sequence_hash = stream_sequence_sha256(current_sequence)
    if current_sequence != source_sequence:
        problems.append(
            "question IDs/order differ from the historical accepted-proposal S1 stream"
        )
    if sequence_hash != gate["expected_stream_sequence_sha256"]:
        problems.append("stream sequence differs from the frozen gate hash")
    if list(model_names) != gate["expected_models"]:
        problems.append("config models differ from the frozen gate")
    if cfg.seeds != [int(gate["expected_seed"])]:
        problems.append("config seed differs from the frozen gate")
    if cfg.window_sizes != [int(gate["expected_window_size"])]:
        problems.append("config window size differs from the frozen gate")
    if cfg.proposal_queries_per_dataset != int(
        gate["expected_queries_per_dataset"]
    ):
        problems.append("config dataset sample size differs from the frozen gate")

    return {
        "no_model_calls": True,
        "config": str(config_path.relative_to(ROOT)),
        "scenario": spec.name,
        "queries_per_model": len(stream),
        "planned_model_calls": len(stream) * len(cfg.models),
        "unique_queries": len({query.qid for query in stream}),
        "dataset_counts": dict(counts),
        "dataset_sources": {
            name: dataset.source for name, dataset in datasets.items()
        },
        "dataset_hashes": {
            name: dataset.content_hash for name, dataset in datasets.items()
        },
        "stream_sequence_sha256": sequence_hash,
        "matches_historical_s1_stream": current_sequence == source_sequence,
        "change_points": spec.change_points,
        "problems": problems,
        "pass": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/proposal_s1_answer_last_validation_asu.yaml",
    )
    parser.add_argument(
        "--gate",
        default="configs/proposal_s1_answer_last_validation_gate.yaml",
    )
    args = parser.parse_args()
    report = run_preflight(args.config, args.gate)
    print(json.dumps(report, indent=2))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
