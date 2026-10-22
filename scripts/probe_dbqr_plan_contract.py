"""Minimal paid probe for the corrected DBQR model-facing plan contract."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_audit import restricted_load  # noqa: E402
from reasoning_monitor.dbqr_json_plan import (  # noqa: E402
    execute_json_plan,
    parse_plan_response,
)
from reasoning_monitor.dbqr_prompt_contract import (  # noqa: E402
    SYSTEM_PROMPT,
    prompt_contract_sha256,
)
from reasoning_monitor.dbqr_prompt_manifest import load_prompt_manifest  # noqa: E402
from reasoning_monitor.dbqr_reference_interpreter import serialize_plan_result  # noqa: E402
from reasoning_monitor.dbqr_schema import compact_schema  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _validate_probes(config: dict) -> None:
    for model in config["models"]:
        probe = json.loads((ROOT / model["probe"]).read_text(encoding="utf-8"))
        if (probe.get("model") != model["provider_model"]
                or not probe.get("results")
                or not all(row.get("accepted") for row in probe["results"])):
            raise ValueError(f"ASU probe is not accepted for {model['name']}")


def _load_case_and_tables(config: dict):
    manifest_path = ROOT / "configs/selections/dbqr_practice_prompt_manifest.json"
    manifest = load_prompt_manifest(manifest_path)
    case = next((row for row in manifest["cases"] if row["qid"] == config["qid"]), None)
    if case is None:
        raise ValueError("Probe qid is absent from the frozen manifest")
    archive_path = ROOT / "data/dbqr/practice.tar.gz"
    blob = archive_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Probe Practice archive hash changed")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        member = archive.getmember(case["table_member"])
        private, _globals = restricted_load(archive.extractfile(member).read())
    schema, aliases = compact_schema(private)
    if schema != case["schema"]:
        raise ValueError("Probe schema changed")
    return case, aliases


def _client(model: dict, config: dict):
    return build_client({
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


def _messages(case: dict, framing: str) -> list[dict[str, str]]:
    prompt = (
        f"QUESTION:\n{case['question']}\n\n"
        f"COMPACT_TABLE_SCHEMA_JSON:\n{_canonical(case['schema'])}"
    )
    if framing != "base":
        prompt = f"{framing}\n\n{prompt}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/dbqr_rq_v1_1_contract_probe_asu.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if len(config["models"]) * len(config["framing_variants"]) != config["expected_calls"]:
        raise ValueError("Probe call budget changed")
    _validate_probes(config)
    case, tables = _load_case_and_tables(config)
    config_hash = hashlib.sha256(
        (
            _digest(config_path)
            + prompt_contract_sha256()
            + _digest(ROOT / "configs/selections/dbqr_practice_prompt_manifest.json")
            + _digest(ROOT / "data/dbqr/practice.tar.gz")
            + _digest(Path(__file__))
        )
        .encode("ascii")
    ).hexdigest()
    output = ROOT / "results/dbqr_v1_1_contract_probe"
    checkpoint = output / "data/records.jsonl"
    summary = {
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "prompt_contract_sha256": prompt_contract_sha256(),
        "execution_authorized": config["execution_authorized"],
        "expected_calls": config["expected_calls"],
        "qid": config["qid"],
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    example = SYSTEM_PROMPT.split("VALID FORMAT EXAMPLE", maxsplit=1)[1]
    parse_plan_response(example.split("\n", maxsplit=1)[1].strip())
    if args.preflight:
        print("Preflight passed. No ASU model call was made.")
        return 0
    if not args.execute:
        return 0
    if not config["execution_authorized"]:
        raise ValueError("Corrected DBQR contract probe is locked")
    if not os.environ.get(config["api_key_env"]):
        raise ValueError(f"Missing {config['api_key_env']}")

    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["config_sha256"] != config_hash:
                    raise ValueError("Probe checkpoint belongs to another contract")
                saved[(row["model"], row["framing_index"])] = row
    progress = ProgressReporter(
        total=config["expected_calls"], label="DBQR contract probe", completed=len(saved))
    progress.print(len(saved))
    records = []
    with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
        for model_index, model in enumerate(config["models"]):
            client = _client(model, config)
            for framing_index, framing in enumerate(config["framing_variants"]):
                key = (model["name"], framing_index)
                record = saved.get(key)
                if record is None:
                    messages = _messages(case, framing)
                    sample = client.generate_messages(
                        messages, seed=20261011 + model_index * 10 + framing_index)
                    plan_error = None
                    result = None
                    if sample.ok:
                        try:
                            _reasoning, plan = parse_plan_response(sample.raw_text)
                            value, _env = execute_json_plan(plan, tables)
                            result = serialize_plan_result(value)
                        except Exception as exc:
                            plan_error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "protocol_id": config["protocol_id"],
                        "config_sha256": config_hash,
                        "model": model["name"],
                        "framing_index": framing_index,
                        "raw_text": sample.raw_text,
                        "executed_result": result,
                        "plan_error": plan_error,
                        "ok": bool(sample.ok and plan_error is None),
                        "n_tokens": sample.n_tokens,
                        "finish_reason": sample.finish_reason,
                        "latency_s": sample.latency_s,
                    }
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    handle.flush()
                    saved[key] = record
                    progress.print(len(saved), final=len(saved) == config["expected_calls"])
                records.append(record)

    counts = Counter(row["model"] for row in records if row["ok"])
    truncations = sum(row["finish_reason"] == "length" for row in records)
    checks = {
        "expected_records": len(records) == config["expected_calls"],
        "all_models_meet_valid_plan_minimum": all(
            counts[model["name"]] >= config["minimum_valid_plans_per_model"]
            for model in config["models"]),
        "no_truncation": truncations <= config["maximum_truncations"],
    }
    report = {
        **summary,
        "valid_plans_by_model": dict(counts),
        "truncations": truncations,
        "checks": checks,
        "pass": all(checks.values()),
        "authorizes": "preparation of a fresh 84-call v1.1 Practice review only",
    }
    report_path = output / "assessment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
