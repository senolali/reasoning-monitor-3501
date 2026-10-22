"""Run the frozen LIC-RQ-v2 science campaign; default is a call-free plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_six_metrics import StrictScorers  # noqa: E402
from reasoning_monitor.lost_conversation import sha256_file  # noqa: E402
from reasoning_monitor.lost_conversation_runner import (  # noqa: E402
    operational_report,
    score_records,
)
from reasoning_monitor.lost_conversation_science import (  # noqa: E402
    cumulative_resolution_lag,
    load_science_plan,
    plot_crl,
)
from reasoning_monitor.lost_conversation_science_acquisition import (  # noqa: E402
    acquire_science,
)
from reasoning_monitor.lost_conversation_semantic_rescore import (  # noqa: E402
    contract_summary,
    final_endpoint,
    metric_summary,
    plot_metrics,
    semantic_records,
)
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter, count_jsonl_records  # noqa: E402

CONFIG_PATH = ROOT / "configs/lost_conversation_v2_science_asu.yaml"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _build_clients(config: dict) -> dict:
    return {
        model["name"]: build_client({
            "backend": "openai_compatible",
            "name": model["name"],
            "model": model["provider_model"],
            "base_url": config["base_url"],
            "api_key_env": config["api_key_env"],
            "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens"],
            "request_options": config["request_options"],
            "max_retries": config["max_retries"],
            "backoff_base": 1.0,
        })
        for model in config["models"]
    }


def _score(records: list[dict], tasks: dict, config: dict, output: Path,
           *, no_model_calls: bool) -> dict:
    print("Applying the frozen semantic parser...", flush=True)
    semantic = semantic_records(records)
    print("Scoring with real local NLI and BERTScore...", flush=True)
    scorers = StrictScorers(with_ss=True)
    frame = score_records(semantic, tasks, scorers, k=config["k_repeats"],
                          n_paraphrases=config["l_paraphrases"])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)
    seven_figures = plot_metrics(frame, output / "figures")

    trajectory, crl = cumulative_resolution_lag(semantic)
    trajectory.to_csv(output / "crl_trajectory.csv", index=False)
    crl.to_csv(output / "crl_by_task.csv", index=False)
    crl_figures = plot_crl(trajectory, crl, output / "figures")
    report = {
        "protocol_id": config["protocol_id"],
        "stage": "frozen_science_scoring",
        "no_model_calls": no_model_calls,
        "records_sha256": sha256_file(output / "data/records.jsonl"),
        "semantic_parser_sha256": sha256_file(
            ROOT / "src/reasoning_monitor/lost_conversation_semantic_rescore.py"),
        "contract_summary": contract_summary(semantic),
        "primary_endpoint": final_endpoint(frame),
        "metric_summary": metric_summary(frame),
        "crl_by_task": json.loads(crl.to_json(orient="records")),
        "figures": seven_figures + crl_figures,
        "analysis_contract": config["analysis"],
        "early_warning_established": False,
        "note": (
            "This file records frozen outcomes and support. Confirmatory paired, "
            "model-stratified and task-clustered tests are reported separately."
        ),
    }
    _write_json(output / "assessment.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true",
                      help="load real local scorers without ASU calls")
    mode.add_argument("--execute-science", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)

    config, _manifest, tasks, task_ids, plan = load_science_plan(ROOT, CONFIG_PATH)
    output = ROOT / config["output_dir"]
    checkpoint = output / "data/records.jsonl"
    summary = {
        "protocol_id": plan["protocol_id"],
        "stage": config["stage"],
        "science_tasks": plan["science_tasks"],
        "calls": plan["expected_calls"],
        "maximum_output_tokens": plan["maximum_output_tokens"],
        "execution_ready": plan["execution_ready"],
        "execution_authorized": plan["execution_authorized"],
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "completed_records": count_jsonl_records(checkpoint),
        "acquisition_fingerprint": plan["acquisition_fingerprint"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not any(vars(args).values()):
        print("Plan only: no ASU model call was made.", flush=True)
        return 0

    if args.preflight:
        print("Loading real local NLI and BERTScore scorers; no ASU calls...", flush=True)
        scorers = StrictScorers(with_ss=True)
        report = {
            **summary,
            "no_model_calls": True,
            "nli_loaded": scorers.nli is not None,
            "bertscore_loaded": scorers.bert is not None,
            "pass": bool(plan["execution_ready"] and scorers.nli is not None
                         and scorers.bert is not None),
        }
        _write_json(ROOT / config["preflight_output_dir"] / "real_scorer_assessment.json",
                    report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["pass"] else 1

    expected = int(plan["expected_calls"])
    if args.score_only:
        if not checkpoint.is_file() or count_jsonl_records(checkpoint) != expected:
            raise ValueError("Science checkpoint is absent or incomplete")
        report = _score(_read_records(checkpoint), tasks, config, output,
                        no_model_calls=True)
        print(json.dumps({"assessment": str(output / 'assessment.json'),
                          "figures": report["figures"]}, indent=2), flush=True)
        return 0

    if not plan["execution_ready"]:
        raise ValueError("Science preflight contract is not ready")
    if not (config["execution_authorized"] and config["science_execution_authorized"]):
        raise ValueError("Science execution is locked pending explicit approval")
    if not os.environ.get(config["api_key_env"]):
        raise ValueError(f"Missing {config['api_key_env']}")

    print("Loading real local scorers before paid calls...", flush=True)
    StrictScorers(with_ss=True)
    completed = count_jsonl_records(checkpoint)
    reporter = ProgressReporter(total=expected, label="LIC-RQ-v2 science",
                                completed=completed)
    reporter.print(completed)
    records = acquire_science(
        config=config,
        tasks=tasks,
        task_ids=task_ids,
        clients=_build_clients(config),
        checkpoint=checkpoint,
        config_hash=plan["acquisition_fingerprint"],
        callback=reporter.print,
    )
    reporter.print(len(records), final=True)
    operation = operational_report(records, expected)
    operation.update({"science_tasks": len(task_ids),
                      "acquisition_fingerprint": plan["acquisition_fingerprint"],
                      "empty_primary_responses": sum(
                          row["role"] == "primary" and not str(row.get("raw_text", "")).strip()
                          for row in records),
                      "empty_history_transport": config["transport"]})
    _write_json(output / "operational_report.json", operation)
    if not operation["complete"]:
        return 1
    _score(records, tasks, config, output, no_model_calls=False)
    print(json.dumps(operation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
