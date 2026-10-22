"""Call-free integration preflight for the Section 59 format stress pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.loaders import load_all  # noqa: E402
from reasoning_monitor.evaluation.proposal_s1_answer_last_validation import (  # noqa: E402
    _strict_final_line_answer as section58_strict_answer,
)
from reasoning_monitor.experiment import (  # noqa: E402
    ExperimentConfig,
    _configured_stream,
    _validate_scenarios,
)

SECTION58_MODELS = ("gpt4o_mini", "llama3_8b")
SELECTION_PATH = ROOT / "configs" / "selections" / "proposal_s1_format_stress.json"
GATE_PATH = ROOT / "configs" / "proposal_s1_format_stress_gate.yaml"
SECTION58_DATA = ROOT / "results" / "proposal_s1_answer_last_validation" / "data"


def _sequence_sha256(rows: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{dataset}|{qid}" for dataset, qid in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section58_failures() -> tuple[set[str], list[dict]]:
    failures: set[str] = set()
    reference_rows: list[dict] = []
    for model in SECTION58_MODELS:
        path = SECTION58_DATA / f"records_S1_{model}_seed42_W10.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs = [
            row
            for row in rows
            if row.get("role") == "run" and int(row.get("call_index", -1)) == 0
        ]
        if model == SECTION58_MODELS[0]:
            reference_rows = runs
        for row in runs:
            missing = section58_strict_answer(
                str(row.get("raw_text") or ""),
                str(row.get("dataset")),
            ) is None
            truncated = (
                row.get("finish_reason") == "length"
                or int(row.get("n_tokens") or 0) >= 2048
            )
            if missing or truncated:
                failures.add(str(row["qid"]))
    return failures, reference_rows


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
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
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
    spec, stream, plan = _configured_stream(
        cfg,
        cfg.scenarios[0],
        datasets,
        cfg.seeds[0],
    )

    problems: list[str] = []
    sequence = [(query.dataset, query.qid) for query in stream]
    sequence_hash = _sequence_sha256(sequence)
    failure_qids, reference_rows = _section58_failures()
    expected_failures = set(gate["expected_failure_qids"])
    expected_controls = set(gate["expected_control_qids"])
    selected_qids = {query.qid for query in stream}

    if len(stream) != 10 or len(selected_qids) != 10:
        problems.append("stream must contain 10 unique questions")
    if any(query.dataset != "math" for query in stream):
        problems.append("format stress stream must contain only MATH questions")
    if failure_qids != expected_failures:
        problems.append("failure QIDs differ from the call-free Section 58 diagnosis")
    source_math = [
        row for row in reference_rows if row.get("dataset") == "math"
    ]
    first_controls = [
        str(row["qid"])
        for row in source_math
        if str(row["qid"]) not in failure_qids
    ][:6]
    if first_controls != gate["expected_control_qids"]:
        problems.append("controls are not the first six non-failure MATH QIDs")
    if selected_qids != expected_failures | expected_controls:
        problems.append("selected QID set differs from the frozen gate")
    if sequence_hash != gate["expected_sequence_sha256"]:
        problems.append("selected sequence differs from the frozen gate hash")
    if sequence_hash != selection["selected_sequence_sha256"]:
        problems.append("selection manifest hash differs from the built stream")
    roles = {row["qid"]: row["selection_role"] for row in plan}
    if any(roles[qid] != "section58_failure" for qid in expected_failures):
        problems.append("failure selection roles are incorrect")
    if any(roles[qid] != "control" for qid in expected_controls):
        problems.append("control selection roles are incorrect")

    if cfg.k_repeats != 1 or cfg.l_paraphrases != 0:
        problems.append("format stress pilot must use K=1 and L=0")
    if cfg.use_rqeval:
        problems.append("format stress pilot must not load rqeval scorers")
    if not cfg.resume:
        problems.append("resume must be enabled")
    if cfg.allow_synthetic_datasets:
        problems.append("synthetic dataset fallback must be disabled")
    if cfg.dataset_prompt_mode != "answer_last_stress":
        problems.append("dataset_prompt_mode must be answer_last_stress")
    if cfg.frozen_selection_size != int(gate["expected_queries_per_model"]):
        problems.append("frozen selection size differs from the gate")
    for query in stream:
        required = (
            "no more than six numbered lines",
            "Do not continue reasoning after line 6.",
            "best answer even if uncertain",
            "plain text exactly in this form: FINAL_ANSWER",
            "Do not put the FINAL_ANSWER line inside LaTeX",
        )
        if any(fragment not in query.prompt for fragment in required):
            problems.append(f"strict stress instruction missing for {query.qid}")
            break

    model_names = tuple(model.get("name") for model in cfg.models)
    expected_models = tuple(gate["expected_models"])
    if model_names != expected_models:
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
            problems.append(f"{model.get('name')} gateway controls differ from the gate")
        system_prompt = str(model.get("system_prompt", ""))
        if "six numbered reasoning lines" not in system_prompt:
            problems.append(f"{model.get('name')} lacks the line-budget instruction")
        if "plain-text FINAL_ANSWER line last" not in system_prompt:
            problems.append(f"{model.get('name')} lacks the plain-text final marker")
        expected_provider_model = gate.get("expected_provider_model")
        if expected_provider_model and model.get("model") != expected_provider_model:
            problems.append(
                f"{model.get('name')} provider model differs from the frozen gate"
            )
    if cfg.seeds != [int(gate["expected_seed"])]:
        problems.append("config seed differs from the gate")
    if cfg.window_sizes != [int(gate["expected_window_size"])]:
        problems.append("config window size differs from the gate")

    return {
        "no_model_calls": True,
        "config": str(config_path.relative_to(ROOT)),
        "scenario": spec.name,
        "queries_per_model": len(stream),
        "planned_model_calls": len(stream) * len(cfg.models),
        "dataset_source": datasets["math"].source,
        "dataset_hash": datasets["math"].content_hash,
        "failure_qids": sorted(failure_qids),
        "control_qids": first_controls,
        "stream_sequence_sha256": sequence_hash,
        "problems": problems,
        "pass": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/proposal_s1_format_stress_asu.yaml",
    )
    parser.add_argument(
        "--gate",
        default="configs/proposal_s1_format_stress_gate.yaml",
    )
    args = parser.parse_args()
    report = run_preflight(args.config, args.gate)
    print(json.dumps(report, indent=2))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
