"""Call-free preflight for the prospectively frozen Section 68 policy."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.experiment import ExperimentConfig  # noqa: E402
from reasoning_monitor.metrics.semantic_answers import (  # noqa: E402
    load_semantic_policy,
)

GATE_PATH = ROOT / "configs" / "semantic_answer_policy_v1_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    policy_path = ROOT / gate["policy_path"]
    cases_path = ROOT / gate["cases_path"]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    problems: list[str] = []

    policy_hash = _sha256(policy_path)
    cases_hash = _sha256(cases_path)
    if policy_hash != gate["expected_policy_sha256"]:
        problems.append("semantic policy hash differs from the frozen gate")
    if cases_hash != gate["expected_cases_sha256"]:
        problems.append("semantic policy case hash differs from the frozen gate")

    try:
        policy = load_semantic_policy(str(policy_path))
    except Exception as exc:
        problems.append(f"semantic policy could not be loaded: {exc}")
        policy = None

    outcomes = []
    if policy is not None:
        if policy.name != gate["expected_policy_name"]:
            problems.append("semantic policy name differs from the frozen gate")
        if policy.require_explicit_units_on_both is not bool(
            gate["require_explicit_units_on_both"]
        ):
            problems.append("explicit-unit requirement differs from the gate")
        for index, case in enumerate(cases):
            observed = policy.equivalent(
                str(case["left"]), str(case["right"]), str(case["dataset"])
            )
            passed = observed is bool(case["expected"])
            outcomes.append(
                {
                    "index": index,
                    "dataset": case["dataset"],
                    "expected": case["expected"],
                    "observed": observed,
                    "pass": passed,
                }
            )
            if not passed:
                problems.append(f"semantic validation case {index} failed")

    if len(cases) != int(gate["expected_case_count"]):
        problems.append(f"expected {gate['expected_case_count']} cases; found {len(cases)}")
    default_policy = ExperimentConfig().semantic_answer_policy
    if gate["historical_default_must_remain_none"] and default_policy is not None:
        problems.append("historical semantic policy default is no longer None")

    report = {
        "analysis": "section68_semantic_answer_policy_preflight",
        "no_model_calls": True,
        "policy_path": str(policy_path.relative_to(ROOT)),
        "policy_sha256": policy_hash,
        "cases_path": str(cases_path.relative_to(ROOT)),
        "cases_sha256": cases_hash,
        "case_count": len(cases),
        "case_passes": sum(outcome["pass"] for outcome in outcomes),
        "historical_default_policy": default_policy,
        "problems": problems,
        "pass": not problems,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
