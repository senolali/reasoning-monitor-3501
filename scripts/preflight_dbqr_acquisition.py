"""Full 420-record DBQR acquisition preflight with fixture clients only."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_acquisition import (  # noqa: E402
    CausalOracleBroker,
    run_acquisition,
)
from reasoning_monitor.dbqr_campaign_contract import load_budget_contract  # noqa: E402
from reasoning_monitor.dbqr_fixture_replay import load_practice_fixture_cases  # noqa: E402
from reasoning_monitor.dbqr_json_plan import PLAN_MARKER  # noqa: E402
from reasoning_monitor.dbqr_prompt_manifest import load_prompt_manifest  # noqa: E402
from reasoning_monitor.progress import ProgressReporter, count_jsonl_records  # noqa: E402
from reasoning_monitor.types import Sample  # noqa: E402


class FixturePlanClient:
    def __init__(self, plans_by_question):
        self.plans_by_question = plans_by_question
        self.calls = 0

    def generate_messages(self, messages, seed=0, gold=""):
        if gold:
            raise ValueError("Acquisition client received forbidden gold")
        current = messages[-1]["content"]
        matches = [plan for question, plan in self.plans_by_question.items()
                   if f"QUESTION:\n{question}\n" in current]
        if len(matches) != 1:
            raise ValueError("Fixture plan could not be selected from current question")
        self.calls += 1
        raw = (
            "1. Apply the fixture plan to validate acquisition transport.\n"
            "2. Preserve the typed local execution result.\n"
            f"{PLAN_MARKER} "
            + json.dumps(matches[0], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )
        return Sample(answer="", reasoning_steps=(), latency_s=0.0,
                      n_tokens=max(1, len(raw) // 4), raw_text=raw,
                      finish_reason="stop")


def main() -> int:
    config, budget = load_budget_contract(ROOT / "configs/dbqr_rq_v1_budget.yaml")
    manifest = load_prompt_manifest(
        ROOT / "configs/selections/dbqr_practice_prompt_manifest.json")
    fixture_cases, source_sha256 = load_practice_fixture_cases(
        (ROOT / "data/dbqr/practice.tar.gz").read_bytes())
    fixture_by_qid = {case["qid"]: case for case in fixture_cases}
    selected_ids = config["stages"]["practice_transport"]["selected_dialogues"]
    selected_cases = [case for case in manifest["cases"]
                      if case["dialogue_id"] in selected_ids]
    answers = {case["qid"]: fixture_by_qid[case["qid"]]["gold_answer"]
               for case in selected_cases}
    plans = {case["question"]: fixture_by_qid[case["qid"]]["fixture_plan"]
             for case in selected_cases}
    clients = {model["name"]: FixturePlanClient(plans) for model in config["models"]}
    combined_hash = hashlib.sha256(
        (budget["config_sha256"] + manifest["manifest_sha256"]).encode("ascii")
    ).hexdigest()
    checkpoint = ROOT / "results/dbqr_v1_acquisition_preflight/data/records.jsonl"
    completed = count_jsonl_records(checkpoint)
    reporter = ProgressReporter(total=420, label="DBQR acquisition preflight", completed=completed)
    reporter.print(completed)

    def callback(_record):
        nonlocal completed
        completed += 1
        reporter.print(completed, final=completed == 420)

    kwargs = {
        "cases": selected_cases,
        "table_loader": lambda case: fixture_by_qid[case["qid"]]["tables"],
        "oracle_broker_factory": lambda order: CausalOracleBroker(answers, order),
        "clients": clients,
        "config": config,
        "checkpoint_path": checkpoint,
        "config_sha256": combined_hash,
        "selected_dialogue_ids": selected_ids,
        "fixture_mode": True,
        "checkpoint_callback": callback,
    }
    staged = run_acquisition(**kwargs, max_turn=2)
    complete = run_acquisition(**kwargs)
    resume_callbacks = []
    resumed = run_acquisition(**{**kwargs, "checkpoint_callback": resume_callbacks.append})
    reporter.print(len(complete), final=True)
    counts = Counter((row["model"], row["condition"], row["turn"]) for row in complete)
    checks = {
        "included_review_stop_is_84_records": len(staged) == 84,
        "full_contract_is_420_records": len(complete) == 420,
        "seven_records_per_cell": len(counts) == 60 and set(counts.values()) == {7},
        "all_fixture_plans_execute": all(row["ok"] for row in complete),
        "no_gold_in_checkpoint_records": all("gold_answer" not in row for row in complete),
        "all_records_are_call_free_fixtures": all(row["fixture_mode"] for row in complete),
        "resume_reuses_all_records": resumed == complete and not resume_callbacks,
        "input_ceiling_enforced": max(row["input_characters"] for row in complete)
                                  <= config["max_input_characters_per_call"],
        "execution_still_not_authorized": not budget["execution_authorized"],
    }
    report = {
        "protocol_id": "DBQR-RQ-v1",
        "stage": "call_free_acquisition_runner_preflight",
        "no_model_calls": True,
        "execution_authorized": False,
        "source_sha256": source_sha256,
        "config_sha256": combined_hash,
        "records": len(complete),
        "included_review_stop_records": len(staged),
        "fixture_client_invocations_this_run": sum(client.calls for client in clients.values()),
        "maximum_input_characters_observed": max(
            row["input_characters"] for row in complete),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    output = ROOT / "results/dbqr_v1_acquisition_preflight/assessment.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
