"""Locked ASU runner for the staged DBQR Practice transport check."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_acquisition import (  # noqa: E402
    CausalOracleBroker,
    load_oracle_vault,
    run_acquisition,
)
from reasoning_monitor.dbqr_audit import restricted_load  # noqa: E402
from reasoning_monitor.dbqr_campaign_contract import load_budget_contract  # noqa: E402
from reasoning_monitor.dbqr_prompt_manifest import load_prompt_manifest  # noqa: E402
from reasoning_monitor.dbqr_schema import compact_schema  # noqa: E402
from reasoning_monitor.dialogue_six_metrics import StrictScorers  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter, count_jsonl_records  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_probes(config: dict) -> None:
    for model in config["models"]:
        path = ROOT / model["probe"]
        probe = json.loads(path.read_text(encoding="utf-8"))
        if (probe.get("model") != model["provider_model"]
                or not probe.get("results")
                or not all(result.get("accepted") for result in probe["results"])):
            raise ValueError(f"ASU probe is not accepted for {model['name']}")


def _load_tables(manifest: dict, selected_ids: list[str]):
    blob = (ROOT / "data/dbqr/practice.tar.gz").read_bytes()
    tables = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for case in manifest["cases"]:
            if case["dialogue_id"] not in selected_ids:
                continue
            private, _globals = restricted_load(
                archive.extractfile(members[case["table_member"]]).read())
            schema, aliases = compact_schema(private)
            if schema != case["schema"]:
                raise ValueError(f"Schema changed for {case['qid']}")
            tables[case["qid"]] = aliases
    return tables


def _clients(config: dict):
    return {
        model["name"]: build_client({
            "backend": "openai_compatible",
            "name": model["name"],
            "model": model["provider_model"],
            "base_url": config["base_url"],
            "api_key_env": config["api_key_env"],
            "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens_per_call"],
            "request_options": config["request_options"],
            "max_retries": config["max_retries"],
            "backoff_base": 1.0,
        })
        for model in config["models"]
    }


def _stage_report(records: list[dict], config_hash: str) -> dict:
    by_model = {}
    for model, rows in itertools.groupby(
            sorted(records, key=lambda row: row["model"]), key=lambda row: row["model"]):
        selected = list(rows)
        valid = sum(row["ok"] for row in selected)
        truncations = sum(
            row.get("finish_reason") == "length"
            or row.get("n_tokens", 0) >= 2048 for row in selected)
        by_model[model] = {
            "records": len(selected),
            "valid_plan_rate": valid / len(selected),
            "truncations": truncations,
            "distinct_raw_responses": len({row["raw_text"] for row in selected}),
            "pass": valid / len(selected) >= 0.95 and truncations == 0,
        }
    checks = {
        "expected_records": len(records) == 84,
        "seven_records_per_cell": set(Counter(
            (row["model"], row["condition"], row["turn"]) for row in records).values()) == {7},
        "model_operational_gates": len(by_model) == 3
                                   and all(row["pass"] for row in by_model.values()),
    }
    return {
        "protocol_id": "DBQR-RQ-v1",
        "stage": "paid_practice_transport_review_stop",
        "config_sha256": config_hash,
        "records": len(records),
        "models": by_model,
        "checks": checks,
        "pass": all(checks.values()),
        "authorizes": "resume the same Practice checkpoint to 420 records only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--stage-review", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "configs/dbqr_rq_v1_budget.yaml"
    config, budget = load_budget_contract(config_path)
    manifest_path = ROOT / "configs/selections/dbqr_practice_prompt_manifest.json"
    manifest = load_prompt_manifest(manifest_path)
    vault_path = ROOT / "results/dbqr_v1_contract/practice_oracle_vault.json"
    answers = load_oracle_vault(
        vault_path, expected_source_sha256=manifest["source_sha256"])
    selected_ids = config["stages"]["practice_transport"]["selected_dialogues"]
    selected_cases = [case for case in manifest["cases"]
                      if case["dialogue_id"] in selected_ids]
    tables = _load_tables(manifest, selected_ids)
    config_hash = hashlib.sha256(
        (_digest(config_path) + _digest(manifest_path) + _digest(vault_path)
         + _digest(Path(__file__))).encode("ascii")
    ).hexdigest()
    _validate_probes(config)

    summary = {
        "config_sha256": config_hash,
        "stage_review_authorized": config["stages"]["practice_transport"][
            "stage_review_authorized"],
        "full_execution_authorized": config["stages"]["practice_transport"][
            "full_execution_authorized"],
        "selected_dialogues": selected_ids,
        "stage_review_calls": budget["practice_staged_review_calls_included_in_total"],
        "full_calls": budget["stages"]["practice_transport"]["logical_calls"],
        "checkpoint": "results/dbqr_v1_practice_transport/data/records.jsonl",
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not any(vars(args).values()):
        return 0
    if args.preflight:
        print("Loading real local NLI and BERTScore scorers...", flush=True)
        StrictScorers(with_ss=True)
        print("Preflight passed. No ASU model call was made.")
        return 0
    practice = config["stages"]["practice_transport"]
    if args.stage_review and not practice["stage_review_authorized"]:
        raise ValueError("Practice stage review is locked in the frozen config")
    if args.execute and not practice["full_execution_authorized"]:
        raise ValueError("Full Practice execution is locked pending separate approval")
    if not os.environ.get(config["api_key_env"]):
        raise ValueError(f"Missing {config['api_key_env']}")

    output = ROOT / "results/dbqr_v1_practice_transport"
    checkpoint = output / "data/records.jsonl"
    completed = count_jsonl_records(checkpoint)
    progress_total = (
        budget["practice_staged_review_calls_included_in_total"]
        if args.stage_review
        else budget["stages"]["practice_transport"]["logical_calls"]
    )
    reporter = ProgressReporter(
        total=progress_total, label="DBQR Practice calls", completed=completed)
    reporter.print(completed)

    def callback(_record):
        nonlocal completed
        completed += 1
        reporter.print(completed, final=completed == progress_total)

    kwargs = {
        "cases": selected_cases,
        "table_loader": lambda case: tables[case["qid"]],
        "oracle_broker_factory": lambda order: CausalOracleBroker(
            {qid: answers[qid] for qid in order}, order),
        "clients": _clients(config),
        "config": config,
        "checkpoint_path": checkpoint,
        "config_sha256": config_hash,
        "selected_dialogue_ids": selected_ids,
        "checkpoint_callback": callback,
    }
    stage_report_path = output / "stage_review_assessment.json"
    if args.stage_review:
        records = run_acquisition(**kwargs, max_turn=2)
        report = _stage_report(records, config_hash)
        output.mkdir(parents=True, exist_ok=True)
        stage_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1

    if not stage_report_path.exists():
        raise ValueError("Run and pass --stage-review before full execution")
    stage_report = json.loads(stage_report_path.read_text(encoding="utf-8"))
    if not stage_report.get("pass") or stage_report.get("config_sha256") != config_hash:
        raise ValueError("Stage-review gate does not match the frozen run")
    records = run_acquisition(**kwargs)
    reporter.print(len(records), final=True)
    print(json.dumps({"records": len(records), "complete": len(records) == 420}, indent=2))
    return 0 if len(records) == 420 else 1


if __name__ == "__main__":
    raise SystemExit(main())
