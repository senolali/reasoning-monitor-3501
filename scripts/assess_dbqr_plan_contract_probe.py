"""Score the completed DBQR v1.1 contract probe against its Practice label."""

from __future__ import annotations

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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_dbqr_plan_contract import _load_case_and_tables  # noqa: E402

from reasoning_monitor.dbqr_answers import typed_answer_equal  # noqa: E402
from reasoning_monitor.dbqr_json_plan import (  # noqa: E402
    execute_json_plan,
    parse_plan_response,
)
from reasoning_monitor.dbqr_reference_interpreter import result_to_answer  # noqa: E402

RECORDS = ROOT / "results/dbqr_v1_1_contract_probe/data/records.jsonl"
OUTPUT = ROOT / "results/dbqr_v1_1_contract_probe/scientific_assessment.json"


def main() -> int:
    config = yaml.safe_load((
        ROOT / "configs/dbqr_rq_v1_1_contract_probe_asu.yaml"
    ).read_text(encoding="utf-8"))
    _case, tables = _load_case_and_tables(config)
    vault = json.loads((
        ROOT / "results/dbqr_v1_contract/practice_oracle_vault.json"
    ).read_text(encoding="utf-8"))
    gold = vault["answers"][config["qid"]]
    rows = [
        json.loads(line)
        for line in RECORDS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assessed = []
    for row in rows:
        answer = None
        error = None
        try:
            _reasoning, plan = parse_plan_response(row["raw_text"])
            result, _env = execute_json_plan(plan, tables)
            answer = result_to_answer(result, gold)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        assessed.append({
            "model": row["model"],
            "framing_index": row["framing_index"],
            "executable": error is None,
            "answer": answer,
            "correct": error is None and typed_answer_equal(answer, gold),
            "error": error,
        })
    executable = Counter(row["model"] for row in assessed if row["executable"])
    correct = Counter(row["model"] for row in assessed if row["correct"])
    report = {
        "analysis": "dbqr_rq_v1_1_contract_probe_scientific_assessment",
        "no_model_calls": True,
        "records": len(rows),
        "records_sha256": hashlib.sha256(RECORDS.read_bytes()).hexdigest(),
        "gold_answer": gold,
        "executable_plans_by_model": dict(executable),
        "correct_plans_by_model": dict(correct),
        "correct_plans": sum(row["correct"] for row in assessed),
        "rows": assessed,
        "pass": len(assessed) == 6 and all(row["correct"] for row in assessed),
        "decision": (
            "The corrected prompt improved syntax but did not establish correct plan "
            "synthesis. Do not authorize a fresh 84-call review."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
