"""Call-free preflight for the Section 66 genuine-paraphrase RS pilot."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data.loaders import load_all  # noqa: E402
from reasoning_monitor.evaluation.proposal_s1_metric_health import (  # noqa: E402
    sequence_sha256,
)
from reasoning_monitor.evaluation.proposal_s1_paraphrase_health import (  # noqa: E402
    DATASETS,
    validate_frozen_config,
)
from reasoning_monitor.experiment import (  # noqa: E402
    ExperimentConfig,
    _configured_stream,
    _make_monitor,
    _validate_scenarios,
)
from reasoning_monitor.models.clients import make_paraphrases  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "proposal_s1_paraphrase_health_gpt_llama4_asu.yaml"
GATE_PATH = ROOT / "configs" / "proposal_s1_paraphrase_health_gpt_llama4_gate.yaml"
PROBES = {
    "gpt4o_mini": ROOT / "results" / "probe_gpt4o_mini_production.json",
    "llama4_scout": ROOT / "results" / "probe_llama4_scout_production.json",
}
DELIMITER = "\nOutput format:\n"
WORD = re.compile(r"[A-Za-z0-9]+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in WORD.findall(text)}


def _choice_block(text: str) -> str | None:
    marker = "\nChoices:\n"
    return text.split(marker, maxsplit=1)[1] if marker in text else None


def main() -> int:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    selection_path = ROOT / str(raw["dataset_selection_manifest"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    try:
        validate_frozen_config(raw, gate)
        _validate_scenarios(cfg.scenarios, cfg.stream_design)
    except ValueError as exc:
        problems.append(str(exc))

    bank_hash = _sha256(selection_path)
    if bank_hash != gate["expected_paraphrase_bank_sha256"]:
        problems.append("paraphrase bank hash differs from the frozen gate")
    if selection.get("authoring_review") != "complete":
        problems.append("paraphrase bank authoring review is not complete")

    datasets = load_all(
        cfg.datasets,
        limit_per_dataset=cfg.items_per_dataset,
        seed=cfg.seed,
        allow_synthetic=cfg.allow_synthetic_datasets,
        prompt_mode=cfg.dataset_prompt_mode,
    )
    _, stream, _ = _configured_stream(cfg, cfg.scenarios[0], datasets, cfg.seeds[0])
    observed = [(query.dataset, query.qid) for query in stream]
    counts = Counter(query.dataset for query in stream)
    if sequence_sha256(observed) != gate["expected_sequence_sha256"]:
        problems.append("selection sequence hash differs from the frozen gate")
    if counts != {dataset: 5 for dataset in DATASETS}:
        problems.append(f"unexpected dataset counts: {dict(counts)}")

    similarities: list[dict] = []
    for query in stream:
        if DELIMITER not in query.prompt:
            problems.append(f"output contract missing for {query.qid}")
            continue
        source_body, source_contract = query.prompt.split(DELIMITER, maxsplit=1)
        variants = make_paraphrases(query, cfg.l_paraphrases)
        if len(variants) != 3:
            problems.append(f"{query.qid} does not have exactly three paraphrases")
            continue
        for index, prompt in enumerate(variants):
            if DELIMITER not in prompt:
                problems.append(f"paraphrase contract missing for {query.qid}:{index}")
                continue
            body, contract = prompt.split(DELIMITER, maxsplit=1)
            if contract != source_contract:
                problems.append(f"output contract changed for {query.qid}:{index}")
            if source_body.strip() in body.strip():
                problems.append(f"wrapper-only paraphrase for {query.qid}:{index}")
            source_tokens = _tokens(source_body)
            variant_tokens = _tokens(body)
            overlap = len(source_tokens & variant_tokens) / max(len(source_tokens), 1)
            similarity = SequenceMatcher(
                None, source_body, body, autojunk=False
            ).ratio()
            if overlap < 0.25:
                problems.append(f"insufficient lexical anchor for {query.qid}:{index}")
            if similarity > 0.95:
                problems.append(f"near-copy paraphrase for {query.qid}:{index}")
            source_choices = _choice_block(source_body)
            if source_choices is not None and _choice_block(body) != source_choices:
                problems.append(f"answer choices changed for {query.qid}:{index}")
            similarities.append(
                {
                    "qid": query.qid,
                    "index": index,
                    "sequence_similarity": similarity,
                    "source_token_overlap": overlap,
                }
            )

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
            {"backend": "simulated", "name": "section66-scorer-preflight"},
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
        "planned_model_calls": len(stream)
        * len(cfg.models)
        * (cfg.k_repeats + cfg.l_paraphrases),
        "dataset_counts": dict(counts),
        "selection_sequence_sha256": sequence_sha256(observed),
        "paraphrase_bank_sha256": bank_hash,
        "paraphrase_prompts": len(similarities),
        "minimum_sequence_similarity": min(
            (row["sequence_similarity"] for row in similarities), default=None
        ),
        "maximum_sequence_similarity": max(
            (row["sequence_similarity"] for row in similarities), default=None
        ),
        "minimum_source_token_overlap": min(
            (row["source_token_overlap"] for row in similarities), default=None
        ),
        "gateway_probes": probe_report,
        "scorer_preflight": scorer_report,
        "independent_human_review": selection.get("independent_human_review"),
        "problems": problems,
        "pass": not problems,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
