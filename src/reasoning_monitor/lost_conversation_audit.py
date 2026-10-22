"""Offline, input-only exposure audit and provisional LIC measurement planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from reasoning_monitor.lost_conversation import MathTask, sha256_file, validate_design


def inside_root(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Audit paths must remain inside the repository")
    return path


def normalized_question(text: str) -> str:
    # Keep punctuation, signs and numbers: different arithmetic is not a match.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _encoding_tolerant(text: str) -> str:
    # Candidate only, NEVER proof of exposure: source has replacement characters.
    return re.sub(r"['\u2018\u2019\ufffd]", "'", normalized_question(text))


def exposure_audit(root: Path, tasks: Sequence[MathTask], campaigns: Sequence[str]) -> dict:
    if not campaigns or len(set(campaigns)) != len(campaigns):
        raise ValueError("A nonempty, unique completed-campaign allowlist is required")
    exact = {t.task_id: normalized_question(t.question) for t in tasks}
    relaxed = {t.task_id: _encoding_tolerant(t.question) for t in tasks}
    hits = {t.task_id: [] for t in tasks}
    candidates = {t.task_id: [] for t in tasks}
    manifest, seen_files = [], set()
    for campaign in campaigns:
        directory = inside_root(root, campaign)
        files = sorted((directory / "data").glob("records*.jsonl"))
        if not files:
            raise ValueError(f"No completed records for allowlisted campaign: {campaign}")
        for path in files:
            path = inside_root(root, str(path))
            if path in seen_files:
                raise ValueError("Duplicate exposure file")
            seen_files.add(path)
            raw = path.read_bytes()
            source = path.relative_to(root).as_posix()
            counts = Counter()
            unique_inputs = set()
            for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["records"] += 1
                if row.get("dataset") != "gsm8k":
                    continue
                counts["gsm8k_records"] += 1
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Missing GSM8K input prompt: {source}:{line_number}")
                prompt = normalized_question(prompt)
                if prompt in unique_inputs:
                    continue
                unique_inputs.add(prompt)
                tolerant = _encoding_tolerant(prompt)
                # Responses, gold, correctness and local qid numbering do not decide matches.
                for task in tasks:
                    evidence = {"path": source, "line": line_number}
                    if exact[task.task_id] in prompt:
                        hits[task.task_id].append(evidence)
                    elif relaxed[task.task_id] in tolerant:
                        candidates[task.task_id].append(evidence)
            manifest.append({"path": source, "sha256": hashlib.sha256(raw).hexdigest(),
                             "records": counts["records"],
                             "gsm8k_records": counts["gsm8k_records"],
                             "unique_gsm8k_prompts": len(unique_inputs)})
    rows = []
    for task in tasks:
        status = ("exact_question_overlap" if hits[task.task_id] else
                  "encoding_variant_candidate" if candidates[task.task_id] else
                  "no_match_in_allowlist")
        rows.append({"task_id": task.task_id, "status": status,
                     "exact_evidence": hits[task.task_id],
                     "candidate_evidence": candidates[task.task_id]})
    return {
        "method": "normalized_full_question_containment_in_recorded_GSM8K_inputs",
        "scope": "explicit_completed_campaigns_only; not pretraining-contamination certification",
        "selection_uses_model_outcomes": False,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "source_files": manifest, "tasks": rows,
    }


def semantic_audit(tasks: Sequence[MathTask], review: Mapping, source_sha256: str) -> dict:
    if review.get("source_sha256") != source_sha256:
        raise ValueError("Semantic review source hash mismatch")
    if (review.get("reviewer") != "assistant_first_pass"
            or review.get("independent_review_complete") is not False):
        raise ValueError("Only explicitly provisional assistant review is supported")
    items = review.get("tasks")
    if not isinstance(items, list):
        raise ValueError("Semantic review tasks must be a list")
    by_id = {r["task_id"]: r for r in items}
    if len(by_id) != len(items) or set(by_id) != {t.task_id for t in tasks}:
        raise ValueError("Review must cover every task exactly once")
    prefixes, reviewed = [], []
    for task in tasks:
        row = by_id[task.task_id]
        n = len(task.shards)
        candidate = row.get("earliest_sufficient_turn_candidate")
        status = row.get("status")
        if status not in {"candidate", "quarantine"} or not row.get("reason"):
            raise ValueError("Every task needs review status and a reason")
        if row.get("total_turns") != n:
            raise ValueError(f"Review shard count mismatch: {task.task_id}")
        if status == "quarantine":
            if candidate is not None:
                raise ValueError("Quarantined tasks cannot have scoring eligibility")
        elif type(candidate) is not int or not 1 <= candidate <= n:
            raise ValueError("Invalid candidate sufficiency turn")
        reviewed.append(dict(row))
        for turn in range(1, n + 1):
            prefixes.append({
                "task_id": task.task_id, "turn": turn, "total_turns": n,
                "review_status": status,
                "candidate_sufficient": candidate is not None and turn >= candidate,
                "baseline_full_information_eligible": status == "candidate" and turn == n,
                "execution_eligibility_authorized": False,
            })
    return {
        "reviewer": review["reviewer"], "independent_review_complete": False,
        "source_sha256": source_sha256,
        "scope": "source semantics only; no model outputs; ordinary arithmetic conventions",
        "status_counts": dict(Counter(r["status"] for r in reviewed)),
        "candidate_early_sufficient_tasks": sum(
            r["status"] == "candidate"
            and r["earliest_sufficient_turn_candidate"] < r["total_turns"] for r in reviewed),
        "candidate_sufficient_partial_prefixes": sum(
            r["candidate_sufficient"] and r["turn"] < r["total_turns"] for r in prefixes),
        "tasks": reviewed, "prefixes": prefixes,
    }


def ranked_sample(tasks: Sequence[MathTask], n: int, seed: str) -> list[MathTask]:
    """Hash-rank whole source tasks without scores, answerability or length enrichment."""
    if type(n) is not int or not 0 <= n <= len(tasks):
        raise ValueError("Requested sample exceeds eligible source-task pool")
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate sampling unit")
    return sorted(tasks, key=lambda t: (
        hashlib.sha256(f"{seed}\n{t.task_id}".encode()).hexdigest(), t.task_id))[:n]


def measurement_budget(tasks: Sequence[MathTask], config: Mapping, semantics: Mapping) -> dict:
    validate_design(config)
    reviews = {r["task_id"]: r for r in semantics["tasks"]}
    if any(reviews[t.task_id]["status"] != "candidate" for t in tasks):
        raise ValueError("Quarantined tasks cannot enter a measurement sample")
    turns = sum(len(t.shards) for t in tasks)
    multiplier = len(config["models"]) * len(config["conditions"])
    calls = turns * multiplier * (config["k_repeats"] + config["l_paraphrases"])
    available = sum(len(t.shards) - reviews[t.task_id]["earliest_sufficient_turn_candidate"] + 1
                    for t in tasks)
    es_available = sum(max(0, len(t.shards) - max(config["es_window"],
                        reviews[t.task_id]["earliest_sufficient_turn_candidate"]) + 1)
                       for t in tasks)
    return {
        "source_tasks": len(tasks), "prefixes_per_model_condition": turns,
        "shards_per_task": dict(sorted(Counter(len(t.shards) for t in tasks).items())),
        "logical_model_calls": calls,
        "max_output_tokens": calls * config["max_output_tokens"],
        "baseline_full_information_prefixes": len(tasks),
        "candidate_gold_sufficient_prefixes_upper_bound": available,
        "candidate_RQ6_prefixes_upper_bound_after_ES_warmup": es_available,
        "candidate_RQ6_coverage_upper_bound": es_available / turns if turns else None,
        "coverage_note": "Upper bound only: requires independent prefix review, answers, traces, "
                         "valid repeats/paraphrases and correct primary for RS; not measured coverage.",
        "currency_cost": None,
        "excluded_costs": ["input_tokens", "retries", "paraphrase_preparation",
                           "auxiliary_classification_or_extraction", "human_review"],
    }


def exact_paired_power(n: int, b: float, c: float, alpha: float = 0.05) -> float:
    """Unconditional exact two-sided McNemar power for hypothetical discordances.

    b=P(control correct, dialogue wrong); c=P(reverse). This plans FINAL task
    success only, not six-metric prediction, multiple tests or turn-level power.
    """
    if (type(n) is not int or n < 1 or not all(math.isfinite(x) for x in (b, c, alpha))
            or b < 0 or c < 0 or b + c > 1 or not 0 < alpha < 1):
        raise ValueError("Invalid paired power arguments")
    q = b + c
    if q == 0:
        return 0.0
    power = 0.0
    for d in range(1, n + 1):
        pd = math.comb(n, d) * q**d * (1 - q)**(n - d)
        for k in range(d + 1):
            p = min(1.0, 2 * sum(math.comb(d, j) for j in range(min(k, d - k) + 1)) / 2**d)
            if p <= alpha:
                power += pd * math.comb(d, k) * (b / q)**k * (c / q)**(d - k)
    return power


def sample_plan(tasks: Sequence[MathTask], exposure: Mapping, semantics: Mapping,
                config: Mapping, audit_config: Mapping) -> dict:
    exposure_by_id = {r["task_id"]: r for r in exposure["tasks"]}
    reviews = {r["task_id"]: r for r in semantics["tasks"]}
    eligible = [t for t in tasks if exposure_by_id[t.task_id]["status"] == "no_match_in_allowlist"
                and reviews[t.task_id]["status"] == "candidate"]
    seed = audit_config["sampling_seed"]
    order = ranked_sample(eligible, len(eligible), seed)
    ni = audit_config["integration_tasks"]
    if type(ni) is not int or not 1 <= ni < len(order):
        raise ValueError("Need disjoint integration and science source tasks")
    integration, pool = order[:ni], order[ni:]
    options = []
    for n in audit_config["science_task_options"]:
        if type(n) is not int or n < 1:
            raise ValueError("Science options must be positive task counts")
        if n > len(pool):
            options.append({"requested_science_tasks": n, "available": False})
            continue
        science = pool[:n]
        options.append({
            "requested_science_tasks": n, "available": True,
            "science_task_ids": [t.task_id for t in science],
            "science_budget": measurement_budget(science, config, semantics),
            "total_with_integration": measurement_budget(integration + science, config, semantics),
            "hypothetical_final_success_power": [
                {"control_only_correct": b, "dialogue_only_correct": c,
                 "net_difference": round(b - c, 2), "two_sided_alpha": 0.05,
                 "power": exact_paired_power(n, b, c)}
                for b, c in ((0.2, 0.1), (0.25, 0.05), (0.35, 0.05))],
        })
    return {
        "status": "provisional_options_not_execution_manifest",
        "selection_uses_model_outcomes": False,
        "sampling_method": "SHA256(seed,newline,source_task_id) ascending; no sufficiency enrichment",
        "sampling_seed": seed, "eligible_unmatched_candidate_tasks": len(eligible),
        "integration_task_ids": [t.task_id for t in integration],
        "integration_budget": measurement_budget(integration, config, semantics),
        "integration_success_criterion": "transport/cache/extraction/scoring integrity, not effect sign",
        "remaining_science_pool": len(pool), "options": options,
        "power_scope": "hypothetical unadjusted paired final-task success, not early-warning power",
        "split_note": "Integration tasks never enter science. Options are nested alternatives, not "
                      "a significance-driven expansion schedule. Select once before model outputs.",
        "inference_note": "No fitted warning thresholds authorized. Source task is the independent "
                          "unit; neither turns, repeats nor the two models double task N.",
        "execution_authorized": False,
    }


def write_artifacts(output: Path, artifacts: Mapping[str, object]) -> None:
    encoded = {name: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
               for name, value in artifacts.items()}
    for name, raw in encoded.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Existing artifact differs; use a new output directory: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in encoded.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)


def run_audit(root: Path, config_path: Path) -> dict[str, object]:
    import yaml

    from reasoning_monitor.lost_conversation import load_math_tasks

    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if settings.get("execution_authorized") is not False:
        raise ValueError("Audit cannot authorize model calls")
    design_path = inside_root(root, settings["design_path"])
    config = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    validate_design(config)
    source = config["source"]
    tasks, _ = load_math_tasks(inside_root(root, source["path"]), source["sha256"])
    review_path = inside_root(root, settings["semantic_review_path"])
    semantics = semantic_audit(tasks, json.loads(review_path.read_text(encoding="utf-8")),
                               source["sha256"])
    exposure = exposure_audit(root, tasks, settings["completed_campaigns"])
    plan = sample_plan(tasks, exposure, semantics, config, settings)
    report = {
        "protocol_id": config["protocol_id"], "analysis": "call_free_input_and_observability_audit",
        "no_model_calls": True, "execution_authorized": False, "execution_ready": False,
        "config_sha256": sha256_file(config_path), "design_sha256": sha256_file(design_path),
        "semantic_review_sha256": sha256_file(review_path), "source_sha256": source["sha256"],
        "math_tasks": len(tasks), "math_prefixes": sum(len(t.shards) for t in tasks),
        "exposure_counts": exposure["status_counts"],
        "historical_record_files_checked": len(exposure["source_files"]),
        "semantic_counts": semantics["status_counts"],
        "candidate_early_sufficient_tasks": semantics["candidate_early_sufficient_tasks"],
        "candidate_sufficient_partial_prefixes": semantics["candidate_sufficient_partial_prefixes"],
        "independent_semantic_review_complete": False,
        "claim": "No degradation or early-warning finding; no new model output collected.",
        "blockers": ["independent source/prefix and frozen paraphrase review",
                     "natural response classification/extraction and adapted ES validation",
                     "choose one sample option and freeze comparisons",
                     "resumable runner and explicit paid-call approval"],
    }
    return {"assessment.json": report, "exposure_audit.json": exposure,
            "semantic_audit.json": semantics, "sample_options.json": plan}
