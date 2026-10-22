"""Call-free preflight for the Section 62 full ASU Mistral validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_proposal_s1_answer_last_validation import (  # noqa: E402
    run_preflight,
)

CONFIG_PATH = "configs/proposal_s1_mistral_validation_asu.yaml"
GATE_PATH = "configs/proposal_s1_mistral_validation_gate.yaml"
PROBE_PATH = ROOT / "results" / "probe_mistral_7b_production.json"
PROVIDER_MODEL = "aws/mistral-7b"


def main() -> int:
    report = run_preflight(CONFIG_PATH, GATE_PATH)
    problems = list(report["problems"])
    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    results = probe.get("results") or []
    accepted = any(
        result.get("name") == "disable_all" and result.get("accepted") is True
        for result in results
    )
    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    if probe.get("model") != PROVIDER_MODEL:
        problems.append("gateway probe used a different provider model")
    if probe.get("executed") is not True or not accepted:
        problems.append("disable_all gateway probe was not executed and accepted")
    if not any(
        result.get("name") == "disable_all"
        and result.get("request_options") == expected_options
        for result in results
    ):
        problems.append("gateway probe controls differ from the validation config")

    report["gateway_probe"] = {
        "path": str(PROBE_PATH.relative_to(ROOT)),
        "provider_model": probe.get("model"),
        "executed": probe.get("executed"),
        "disable_all_accepted": accepted,
    }
    report["known_asu_limitation"] = (
        "Prior Mistral K=5 runs had zero verbatim repeat diversity; Section 62 "
        "uses K=1/L=0 and cannot clear that limitation."
    )
    report["problems"] = problems
    report["pass"] = not problems
    print(json.dumps(report, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
