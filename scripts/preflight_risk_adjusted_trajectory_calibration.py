"""Call-free operational preflight for the frozen RAT-v1 calibration run."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from reasoning_monitor.data.controlled_trajectories import OUTPUT_CONTRACT  # noqa: E402
from reasoning_monitor.evaluation.risk_adjusted_trajectory_preflight import (  # noqa: E402
    run_preflight,
)
from reasoning_monitor.experiment import (  # noqa: E402
    ExperimentConfig,
    _configured_stream,
    _make_monitor,
    _validate_scenarios,
)
from reasoning_monitor.models.clients import make_paraphrases  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_calibration_asu.yaml"
GATE_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_calibration_gate.yaml"
PROTOCOL_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1.yaml"
STRUCTURAL_GATE_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_gate.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    problems: list[str] = []

    structural = run_preflight(
        ROOT,
        PROTOCOL_PATH,
        STRUCTURAL_GATE_PATH,
    )
    if not structural["pass"]:
        problems.append("RAT-v1 structural preflight no longer passes")
    if _sha256(CONFIG_PATH) != gate["expected_config_sha256"]:
        problems.append("calibration config hash differs from the frozen gate")

    bank_path = ROOT / str(raw["controlled_trajectory_bank"])
    if _sha256(bank_path) != gate["expected_bank_sha256"]:
        problems.append("controlled bank hash differs from the frozen gate")
    policy_path = ROOT / str(raw["semantic_answer_policy"])
    if _sha256(policy_path) != gate["expected_semantic_policy_sha256"]:
        problems.append("semantic answer policy hash differs from the frozen gate")

    try:
        _validate_scenarios(cfg.scenarios, cfg.stream_design)
        spec, stream, plan = _configured_stream(cfg, "RAT", {}, cfg.seed)
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(str(exc))
        spec, stream, plan = None, [], []

    template_item_counts = Counter(query.domain for query in stream if query.family_id)
    template_family_counts = {
        template: len(
            {query.family_id for query in stream if query.domain == template}
        )
        for template in template_item_counts
    }
    levels = sorted({query.trajectory_level for query in stream})
    families = {query.family_id for query in stream}
    prompt_contract_ok = all(
        query.prompt.endswith(OUTPUT_CONTRACT)
        and len(make_paraphrases(query, cfg.l_paraphrases))
        == gate["expected_l_paraphrases"]
        for query in stream
    )
    stream_checks = {
        "scenario": spec is not None and spec.name == "RAT",
        "split": raw.get("controlled_trajectory_split") == gate["expected_split"],
        "query_count": len(stream) == gate["expected_queries_per_model"],
        "plan_count": len(plan) == gate["expected_queries_per_model"],
        "family_count": len(families) == gate["expected_families"],
        "families_per_template": template_family_counts
        == {
            "arithmetic_chain": gate["expected_families_per_template"],
            "shortest_path": gate["expected_families_per_template"],
            "precedence_schedule": gate["expected_families_per_template"],
        },
        "levels": levels == gate["expected_levels"],
        "family_metadata": all(
            query.family_id
            and query.trajectory_level
            and query.trajectory_split == gate["expected_split"]
            for query in stream
        ),
        "prompt_contract": prompt_contract_ok,
        "resume": cfg.resume is True,
        "rqeval": cfg.use_rqeval is True,
        "nli": cfg.coherence_scorer == "nli",
        "bertscore": cfg.stability_scorer == "bertscore",
        "semantic_policy": cfg.semantic_answer_policy
        == "configs/semantic_answer_policy_v1.yaml",
    }
    problems.extend(name for name, passed in stream_checks.items() if not passed)

    expected_options = gate["expected_request_options"]
    providers = gate["expected_provider_models"]
    model_names = [str(model["name"]) for model in cfg.models]
    model_checks = {
        "model_names": model_names == list(providers),
        "provider_models": all(
            model.get("model") == providers.get(str(model["name"]))
            for model in cfg.models
        ),
        "request_options": all(
            model.get("request_options") == expected_options for model in cfg.models
        ),
        "temperature": all(float(model.get("temperature", -1)) == 0.7 for model in cfg.models),
        "token_ceiling": all(
            int(model.get("max_tokens", -1)) == gate["expected_max_tokens_per_call"]
            for model in cfg.models
        ),
        "k_repeats": cfg.k_repeats == gate["expected_k_repeats"],
        "l_paraphrases": cfg.l_paraphrases == gate["expected_l_paraphrases"],
    }
    problems.extend(name for name, passed in model_checks.items() if not passed)

    probe_report = {}
    for name, probe_spec in gate["gateway_probes"].items():
        probe_path = ROOT / probe_spec["path"]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        accepted = any(
            result.get("name") == "disable_all"
            and result.get("accepted") is True
            and result.get("request_options") == expected_options
            for result in probe.get("results") or []
        )
        hash_ok = _sha256(probe_path) == probe_spec["sha256"]
        model_ok = probe.get("model") == providers[name]
        if not (accepted and hash_ok and model_ok):
            problems.append(f"{name} frozen production probe does not match")
        probe_report[name] = {
            "path": probe_spec["path"],
            "sha256": _sha256(probe_path),
            "provider_model": probe.get("model"),
            "accepted": accepted,
        }

    scorer_report = {}
    try:
        monitor = _make_monitor(
            {"backend": "simulated", "name": "rat-v1-scorer-preflight"},
            cfg.window_sizes[0],
            cfg,
        )
        scorer_report = monitor.preflight(strict=True)
    except Exception as exc:
        problems.append(f"real scorer preflight failed: {exc}")

    planned_calls = len(stream) * len(cfg.models) * (
        cfg.k_repeats + cfg.l_paraphrases
    )
    max_output_tokens = planned_calls * gate["expected_max_tokens_per_call"]
    budget_checks = {
        "calls_per_model": len(stream) * (cfg.k_repeats + cfg.l_paraphrases)
        == gate["expected_calls_per_model"],
        "total_calls": planned_calls == gate["expected_total_calls"],
        "max_output_tokens": max_output_tokens == gate["expected_max_output_tokens"],
        "paid_calls_blocked": gate["paid_calls_authorized"] is False,
    }
    problems.extend(name for name, passed in budget_checks.items() if not passed)

    report = {
        "analysis": "rat_v1_calibration_preflight",
        "no_model_calls": True,
        "execution_authorized": False,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG_PATH),
        "structural_preflight_pass": structural["pass"],
        "queries_per_model": len(stream),
        "families": len(families),
        "template_item_counts": dict(template_item_counts),
        "template_family_counts": template_family_counts,
        "levels": levels,
        "models": model_names,
        "calls_per_model": (
            len(stream) * (cfg.k_repeats + cfg.l_paraphrases)
        ),
        "planned_model_calls": planned_calls,
        "max_output_tokens": max_output_tokens,
        "stream_checks": stream_checks,
        "model_checks": model_checks,
        "budget_checks": budget_checks,
        "gateway_probes": probe_report,
        "scorer_preflight": scorer_report,
        "problems": problems,
        "pass": not problems,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
