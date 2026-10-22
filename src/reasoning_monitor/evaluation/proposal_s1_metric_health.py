"""Frozen preflight and post-run gate for the Section 64 metric-health pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from reasoning_monitor.evaluation.kl_validation import repeat_diversity_rate
from reasoning_monitor.evaluation.proposal_s1_answer_last_validation import (
    _strict_final_line_answer,
    _violates_step_contract,
)
from reasoning_monitor.metrics.dimensions import answers_equal
from reasoning_monitor.types import METRIC_ORDER

DATASETS = ("gsm8k", "arc_challenge", "math")
TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens", "model_length"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sequence_sha256(rows: list[tuple[str, str]]) -> str:
    payload = "\n".join(f"{dataset}|{qid}" for dataset, qid in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_sequence(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    primary = sorted(
        (
            record
            for record in records
            if record.get("role") == "run"
            and int(record.get("call_index", -1)) == 0
        ),
        key=lambda record: int(record["t"]),
    )
    return [(str(record["dataset"]), str(record["qid"])) for record in primary]


def _scorer_preflight_pass(manifest: dict[str, Any]) -> bool:
    preflight = manifest.get("preflight") or {}
    scorers = preflight.get("scorers") or {}
    ls = scorers.get("LS") or {}
    ss = scorers.get("SS") or {}
    return bool(
        not preflight.get("problems")
        and ls.get("ready") is True
        and ls.get("scorer") == "nli"
        and ss.get("ready") is True
        and ss.get("scorer") == "bertscore"
    )


def validate_frozen_config(config: dict[str, Any], gate: dict[str, Any]) -> None:
    expected = {
        "output_dir": "results/proposal_s1_metric_health_gpt_llama4",
        "stream_design": "proposal_frozen_selection",
        "datasets": list(DATASETS),
        "allow_synthetic_datasets": False,
        "dataset_prompt_mode": gate["expected_prompt_mode"],
        "scenarios": ["S1"],
        "frozen_selection_size": gate["expected_queries_per_model"],
        "seeds": [gate["expected_seed"]],
        "window_sizes": [gate["expected_window_size"]],
        "k_repeats": gate["expected_k_repeats"],
        "l_paraphrases": gate["expected_l_paraphrases"],
        "use_rqeval": True,
        "coherence_scorer": "nli",
        "stability_scorer": "bertscore",
        "publication_mode": True,
        "resume": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    models = config.get("models") or []
    names = [model.get("name") for model in models]
    if names != gate["expected_models"]:
        mismatches["models"] = (names, gate["expected_models"])
    providers = gate["expected_provider_models"]
    for model in models:
        name = str(model.get("name"))
        if model.get("model") != providers.get(name):
            mismatches[f"provider:{name}"] = (
                model.get("model"),
                providers.get(name),
            )
        if int(model.get("max_tokens") or 0) != int(gate["max_output_tokens"]):
            mismatches[f"max_tokens:{name}"] = (
                model.get("max_tokens"),
                gate["max_output_tokens"],
            )
    if mismatches:
        raise ValueError(f"result does not match frozen Section 64 config: {mismatches}")


def assess_model_records(
    records: list[dict[str, Any]],
    per_query: pd.DataFrame,
    gate: dict[str, Any],
    scorer_preflight_pass: bool,
) -> dict[str, Any]:
    expected_queries = int(gate["expected_queries_per_model"])
    expected_records = int(gate["expected_records_per_model"])
    expected_k = int(gate["expected_k_repeats"])
    expected_l = int(gate["expected_l_paraphrases"])
    max_tokens = int(gate["max_output_tokens"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)

    role_counts_ok = all(
        sum(record.get("role") == "run" for record in rows) == expected_k
        and sum(record.get("role") == "paraphrase" for record in rows) == expected_l
        for rows in grouped.values()
    )
    failures = sum(not bool(record.get("ok")) for record in records)
    truncations = sum(
        str(record.get("finish_reason") or "").strip().lower()
        in TRUNCATION_REASONS
        or int(record.get("n_tokens") or 0) >= max_tokens
        for record in records
    )
    extracted = [
        _strict_final_line_answer(
            str(record.get("raw_text") or ""),
            str(record.get("dataset")),
            bool(gate.get("forbid_code_fences", False)),
        )
        for record in records
    ]
    strict_observable = sum(candidate is not None for candidate in extracted)
    stored_agreement = sum(
        candidate is not None
        and answers_equal(
            str(record.get("answer") or ""),
            candidate,
            str(record.get("dataset")),
        )
        for record, candidate in zip(records, extracted, strict=True)
    )
    step_violations = sum(
        _violates_step_contract(str(record.get("raw_text") or ""))
        for record in records
    )
    diverse, complete, diversity_rate = repeat_diversity_rate(records, expected_k)

    metric_non_nan: dict[str, int] = {}
    metric_unique: dict[str, int] = {}
    metric_ranges: dict[str, list[float] | None] = {}
    metrics_in_range = True
    for metric in METRIC_ORDER:
        column = f"{metric}_raw"
        if column not in per_query:
            metric_non_nan[metric] = 0
            metric_unique[metric] = 0
            metric_ranges[metric] = None
            metrics_in_range = False
            continue
        values = [float(value) for value in per_query[column].dropna().tolist()]
        metric_non_nan[metric] = len(values)
        metric_unique[metric] = len({round(value, 12) for value in values})
        metric_ranges[metric] = [min(values), max(values)] if values else None
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            metrics_in_range = False

    rs_by_dataset = {
        dataset: int(
            per_query.loc[
                per_query["dataset"] == dataset, "RS_raw"
            ].notna().sum()
        )
        for dataset in DATASETS
    }
    required_non_rs = int(gate["required_non_rs_metric_coverage"])
    metric_coverage_pass = all(
        metric_non_nan[metric] == required_non_rs
        for metric in METRIC_ORDER
        if metric != "RS"
    ) and metric_non_nan["RS"] >= int(gate["minimum_rs_evaluable"])
    rs_dataset_pass = all(
        count >= int(gate["minimum_rs_evaluable_per_dataset"])
        for count in rs_by_dataset.values()
    )
    metric_nondegenerate = all(
        metric_unique[metric] >= int(gate["minimum_unique_values_per_metric"])
        for metric in METRIC_ORDER
    )

    primary = [
        record
        for record in records
        if record.get("role") == "run"
        and int(record.get("call_index", -1)) == 0
    ]
    dataset_counts = Counter(str(record.get("dataset")) for record in primary)
    descriptive_correct = {
        dataset: sum(
            answers_equal(
                str(record.get("answer") or ""),
                str(record.get("gold_answer") or ""),
                dataset,
            )
            for record in primary
            if record.get("dataset") == dataset
        )
        for dataset in DATASETS
    }

    success_rate = (len(records) - failures) / max(len(records), 1)
    strict_rate = strict_observable / max(len(records), 1)
    agreement_rate = stored_agreement / max(len(records), 1)
    checks = {
        "record_count": len(records) == expected_records,
        "query_count": len(grouped) == expected_queries,
        "per_query_count": len(per_query) == expected_queries,
        "dataset_counts": dataset_counts
        == {dataset: int(gate["expected_queries_per_dataset"]) for dataset in DATASETS},
        "role_counts": role_counts_ok,
        "execution": success_rate >= float(gate["minimum_success_rate"]),
        "strict_marker": strict_rate >= float(gate["minimum_strict_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "step_contract": step_violations
        <= int(gate["maximum_step_contract_violations"]),
        "repeat_diversity": complete == expected_queries
        and diversity_rate >= float(gate["minimum_k_repeat_diversity_rate"]),
        "metric_coverage": metric_coverage_pass,
        "rs_dataset_coverage": rs_dataset_pass,
        "metrics_in_range": metrics_in_range,
        "metric_nondegeneracy": metric_nondegenerate,
        "real_scorers": scorer_preflight_pass,
    }
    return {
        "n_records": len(records),
        "n_queries": len(grouped),
        "dataset_counts": dict(dataset_counts),
        "success_rate": success_rate,
        "strict_marker_rate": strict_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "step_contract_violations": step_violations,
        "k_repeat_diverse_queries": diverse,
        "k_repeat_complete_queries": complete,
        "k_repeat_diversity_rate": diversity_rate,
        "metric_non_nan": metric_non_nan,
        "metric_unique_values": metric_unique,
        "metric_ranges": metric_ranges,
        "rs_evaluable_by_dataset": rs_by_dataset,
        "descriptive_correct": descriptive_correct,
        "checks": checks,
        "overall_gate_pass": all(checks.values()),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Section 64 GPT-4o-mini + Llama-4 Scout Metric-Health Pilot",
        "",
        "This report applies the gate frozen before any Section 64 response.",
        "Accuracy and detector alarms are descriptive; this is a scorer and",
        "sampling-health pilot on a disjoint 15-question selection.",
        "",
    ]
    for model, values in result["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Records: {values['n_records']}/120",
                f"- Successful calls: {values['success_rate']:.3f}",
                f"- Strict typed markers: {values['strict_marker_rate']:.3f}",
                f"- Truncations: {values['truncations']}",
                f"- K-repeat diversity: {values['k_repeat_diversity_rate']:.3f}",
                f"- RS evaluable: {values['metric_non_nan']['RS']}/15",
                "",
                "| Check | Pass |",
                "| --- | --- |",
            ]
        )
        for name, passed in values["checks"].items():
            lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
        lines.extend(
            [
                "",
                f"Model gate: **{'PASS' if values['overall_gate_pass'] else 'FAIL'}**.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"Overall frozen gate: **{'PASS' if result['overall_gate_pass'] else 'FAIL'}**.",
            "A pass permits only preregistration of a separate full 90-question",
            "GPT-4o-mini plus Llama-4 Scout K=5/L=3 campaign. It does not start",
            "that campaign or establish early warning.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_metric_health(
    result_dir: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    gate_path = Path(gate_path)
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    validate_frozen_config(config, gate)
    scorer_pass = _scorer_preflight_pass(manifest)
    seed = int(gate["expected_seed"])
    window = int(gate["expected_window_size"])
    models: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    observed_sequences: list[str] = []

    for model in gate["expected_models"]:
        tag = f"S1_{model}_seed{seed}_W{window}"
        records_path = result_dir / "data" / f"records_{tag}.jsonl"
        per_query_path = result_dir / "data" / f"per_query_{tag}.csv"
        records = _read_jsonl(records_path)
        per_query = pd.read_csv(per_query_path)
        model_result = assess_model_records(records, per_query, gate, scorer_pass)
        sequence_hash = sequence_sha256(_primary_sequence(records))
        model_result["stream_sequence_sha256"] = sequence_hash
        model_result["checks"]["stream_identity"] = (
            sequence_hash == gate["expected_sequence_sha256"]
        )
        model_result["overall_gate_pass"] = all(model_result["checks"].values())
        models[model] = model_result
        observed_sequences.append(sequence_hash)
        source_hashes[model] = {
            "records_sha256": _sha256(records_path),
            "per_query_sha256": _sha256(per_query_path),
        }

    result = {
        "analysis": "section64_proposal_s1_metric_health_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "models": models,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_hashes": source_hashes,
        "stream_identity_across_models": len(set(observed_sequences)) == 1,
        "overall_gate_pass": bool(models)
        and len(set(observed_sequences)) == 1
        and all(values["overall_gate_pass"] for values in models.values()),
        "gate": gate,
    }
    (result_dir / "metric_health_assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(result_dir / "METRIC_HEALTH_ASSESSMENT.md", result)
    return result

