"""Call-free preflight for the Section 60 Llama-4 Scout stress arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_proposal_s1_format_stress import run_preflight  # noqa: E402

CONFIG_PATH = "configs/proposal_s1_llama4_scout_stress_asu.yaml"
GATE_PATH = "configs/proposal_s1_llama4_scout_stress_gate.yaml"
PROBE_PATH = ROOT / "results" / "probe_llama4_scout_production.json"
PROVIDER_MODEL = "aws/llama4_scout-17b"


def main() -> int:
    report = run_preflight(CONFIG_PATH, GATE_PATH)
    problems = list(report["problems"])
    probe = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    results = probe.get("results") or []
    accepted = any(
        result.get("name") == "disable_all" and result.get("accepted") is True
        for result in results
    )
    if probe.get("model") != PROVIDER_MODEL:
        problems.append("gateway probe used a different provider model")
    if probe.get("executed") is not True or not accepted:
        problems.append("disable_all gateway probe was not executed and accepted")
    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    if not any(
        result.get("name") == "disable_all"
        and result.get("request_options") == expected_options
        for result in results
    ):
        problems.append("gateway probe controls differ from the stress config")

    report["gateway_probe"] = {
        "path": str(PROBE_PATH.relative_to(ROOT)),
        "provider_model": probe.get("model"),
        "executed": probe.get("executed"),
        "disable_all_accepted": accepted,
    }
    report["problems"] = problems
    report["pass"] = not problems
    print(json.dumps(report, indent=2))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
