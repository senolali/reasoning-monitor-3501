"""Call-free preflight for the Section 64 GPT/Llama-4 metric-health pilot."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.loaders import load_all  # noqa: E402
from reasoning_monitor.data.proposal_scenarios import (  # noqa: E402
    build_proposal_s1_stream,
)
from reasoning_monitor.evaluation.proposal_s1_metric_health import (  # noqa: E402
    DATASETS,
    sequence_sha256,
    validate_frozen_config,
)
from reasoning_monitor.experiment import (  # noqa: E402
    ExperimentConfig,
    _configured_stream,
    _make_monitor,
    _validate_scenarios,
)

CONFIG_PATH = ROOT / "configs" / "proposal_s1_metric_health_gpt_llama4_asu.yaml"
GATE_PATH = ROOT / "configs" / "proposal_s1_metric_health_gpt_llama4_gate.yaml"
PROBES = {
    "gpt4o_mini": ROOT / "results" / "probe_gpt4o_mini_production.json",
    "llama4_scout": ROOT / "results" / "probe_llama4_scout_production.json",
}


def _derived_disjoint_sequence(datasets: dict, seed: int) -> list[tuple[str, str]]:
    _, expanded, _ = build_proposal_s1_stream(
        datasets, seed=seed, queries_per_dataset=40
    )
    by_dataset = {
        dataset: [query for query in expanded if query.dataset == dataset]
        for dataset in DATASETS
    }
    return [
        (dataset, by_dataset[dataset][rank].qid)
        for rank in range(30, 35)
        for dataset in DATASETS
    ]


def main() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    problems: list[str] = []
    try:
        validate_frozen_config(raw, gate)
        _validate_scenarios(cfg.scenarios, cfg.stream_design)
    except ValueError as exc:
        problems.append(str(exc))

    datasets = load_all(
        cfg.datasets,
        limit_per_dataset=cfg.items_per_dataset,
        seed=cfg.seed,
        allow_synthetic=cfg.allow_synthetic_datasets,
        prompt_mode=cfg.dataset_prompt_mode,
    )
    _, stream, _ = _configured_stream(
        cfg, cfg.scenarios[0], datasets, cfg.seeds[0]
    )
    observed = [(query.dataset, query.qid) for query in stream]
    expected = _derived_disjoint_sequence(datasets, cfg.seed)
    counts = Counter(query.dataset for query in stream)
    if observed != expected:
        problems.append("frozen selection differs from seed-42 ranks 31-35")
    if len(stream) != 15 or len({query.qid for query in stream}) != 15:
        problems.append("selection must contain 15 unique questions")
    if counts != {dataset: 5 for dataset in DATASETS}:
        problems.append(f"unexpected dataset counts: {dict(counts)}")
    sequence_hash = sequence_sha256(observed)
    if sequence_hash != gate["expected_sequence_sha256"]:
        problems.append("selection sequence hash differs from the frozen gate")

    _, historical, _ = build_proposal_s1_stream(
        datasets, seed=cfg.seed, queries_per_dataset=30
    )
    historical_qids = {query.qid for query in historical}
    overlap = sorted(historical_qids & {query.qid for query in stream})
    if overlap:
        problems.append(f"selection overlaps the historical 90 QIDs: {overlap[:3]}")

    for query in stream:
        marker = "FINAL_CHOICE" if query.dataset == "arc_challenge" else "FINAL_ANSWER"
        if "no more than six numbered lines" not in query.prompt:
            problems.append(f"line budget missing for {query.qid}")
            break
        if f"plain text exactly in this form: {marker}:" not in query.prompt:
            problems.append(f"typed final marker missing for {query.qid}")
            break
        if not query.prompt.endswith(
            f"Do not write anything after the {marker} line."
        ):
            problems.append(f"terminal marker instruction missing for {query.qid}")
            break

    expected_options = {
        "enable_history": False,
        "enable_search": False,
        "enhance_prompt": {"time": False, "date": False},
    }
    probe_report = {}
    providers = gate["expected_provider_models"]
    for model in cfg.models:
        name = str(model["name"])
        if model.get("request_options") != expected_options:
            problems.append(f"{name} gateway controls differ from the gate")
        if "six numbered reasoning lines" not in str(model.get("system_prompt", "")):
            problems.append(f"{name} system prompt lacks the line budget")
        probe = json.loads(PROBES[name].read_text(encoding="utf-8"))
        accepted = any(
            result.get("name") == "disable_all"
            and result.get("accepted") is True
            and result.get("request_options") == expected_options
            for result in probe.get("results") or []
        )
        if probe.get("model") != providers[name] or not accepted:
            problems.append(f"{name} accepted production probe does not match")
        probe_report[name] = {
            "path": str(PROBES[name].relative_to(ROOT)),
            "provider_model": probe.get("model"),
            "accepted": accepted,
        }

    scorer_report = {}
    try:
        monitor = _make_monitor(
            {"backend": "simulated", "name": "section64-scorer-preflight"},
            cfg.window_sizes[0],
            cfg,
        )
        scorer_report = monitor.preflight(strict=True)
    except Exception as exc:
        problems.append(f"real scorer preflight failed: {exc}")

    report = {
        "no_model_calls": True,
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "queries_per_model": len(stream),
        "planned_model_calls": (
            len(stream)
            * len(cfg.models)
            * (cfg.k_repeats + cfg.l_paraphrases)
        ),
        "dataset_counts": dict(counts),
        "dataset_sources": {
            name: dataset.source for name, dataset in datasets.items()
        },
        "dataset_hashes": {
            name: dataset.content_hash for name, dataset in datasets.items()
        },
        "selection_sequence_sha256": sequence_hash,
        "disjoint_from_historical_90": not overlap,
        "gateway_probes": probe_report,
        "scorer_preflight": scorer_report,
        "problems": problems,
        "pass": not problems,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
