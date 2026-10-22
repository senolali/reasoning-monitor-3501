"""Call-free assessment of the Section 56 parser-controlled S1 pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import norm

from reasoning_monitor.metrics.dimensions import answers_equal

DATASETS = ("gsm8k", "arc_challenge", "math")
_FIRST_CHOICE = re.compile(
    r"(?i)\A[ \t]*FINAL_CHOICE[ \t]*:[ \t]*([A-D])[ \t]*(?:\r?\n|$)"
)
_FIRST_ANSWER = re.compile(
    r"(?i)\A[ \t]*FINAL_ANSWER[ \t]*:[ \t]*(\S.*?)[ \t]*(?:\r?\n|$)"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_sequence_sha256(records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: int(record["t"]))
    payload = "\n".join(
        f"{record['dataset']}|{record['qid']}" for record in ordered
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def _ordered_trend_test(correct: dict[str, int], counts: dict[str, int]) -> dict[str, float]:
    scores = (0.0, 1.0, 2.0)
    successes = [correct[dataset] for dataset in DATASETS]
    totals = [counts[dataset] for dataset in DATASETS]
    total = sum(totals)
    pooled = sum(successes) / total
    statistic = sum(
        score * (success - group_total * pooled)
        for score, success, group_total in zip(scores, successes, totals, strict=True)
    )
    variance = pooled * (1 - pooled) * (
        sum(group_total * score * score for group_total, score in zip(totals, scores, strict=True))
        - sum(group_total * score for group_total, score in zip(totals, scores, strict=True))
        ** 2
        / total
    )
    if variance <= 0:
        return {"z": 0.0, "one_sided_p": 1.0}
    z_value = statistic / math.sqrt(variance)
    return {"z": z_value, "one_sided_p": float(norm.cdf(z_value))}


def _strict_first_line_answer(raw_text: str, dataset: str) -> str | None:
    pattern = _FIRST_CHOICE if dataset == "arc_challenge" else _FIRST_ANSWER
    match = pattern.match(raw_text)
    return match.group(1).strip() if match else None


def assess_model_records(
    records: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    runs = [
        record
        for record in records
        if record.get("role") == "run" and int(record.get("call_index", -1)) == 0
    ]
    rows = {dataset: [record for record in runs if record.get("dataset") == dataset] for dataset in DATASETS}
    expected_per_dataset = int(gate["expected_queries_per_dataset"])
    expected_total = expected_per_dataset * len(DATASETS)
    max_tokens = int(gate["max_output_tokens"])

    extracted = [
        _strict_first_line_answer(str(record.get("raw_text") or ""), str(record.get("dataset")))
        for record in runs
    ]
    resolved = sum(answer is not None for answer in extracted)
    stored_agreement = sum(
        candidate is not None
        and answers_equal(str(record.get("answer") or ""), candidate, str(record.get("dataset")))
        for record, candidate in zip(runs, extracted, strict=True)
    )
    truncations = sum(
        record.get("finish_reason") == "length"
        or int(record.get("n_tokens") or 0) >= max_tokens
        for record in runs
    )
    success_rate = sum(bool(record.get("ok")) for record in runs) / max(len(runs), 1)
    marker_rate = resolved / max(len(runs), 1)
    agreement_rate = stored_agreement / max(len(runs), 1)

    correct: dict[str, int] = {}
    accuracy: dict[str, float] = {}
    intervals: dict[str, list[float]] = {}
    for dataset in DATASETS:
        dataset_rows = rows[dataset]
        dataset_correct = sum(
            candidate is not None
            and answers_equal(candidate, str(record.get("gold_answer") or ""), dataset)
            for record, candidate in zip(runs, extracted, strict=True)
            if record.get("dataset") == dataset
        )
        correct[dataset] = dataset_correct
        accuracy[dataset] = dataset_correct / max(len(dataset_rows), 1)
        intervals[dataset] = _wilson_interval(dataset_correct, len(dataset_rows))

    counts = {dataset: len(rows[dataset]) for dataset in DATASETS}
    trend = _ordered_trend_test(correct, counts)
    gap = accuracy["gsm8k"] - accuracy["math"]
    checks = {
        "record_count": len(runs) == expected_total,
        "unique_queries": len({record.get("qid") for record in runs}) == expected_total,
        "dataset_counts": all(counts[dataset] == expected_per_dataset for dataset in DATASETS),
        "stream_identity": stream_sequence_sha256(runs) == gate["expected_stream_sequence_sha256"],
        "execution": success_rate >= float(gate["minimum_success_rate"]),
        "first_line_marker": marker_rate >= float(gate["minimum_first_line_marker_rate"]),
        "stored_answer_agreement": agreement_rate >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "gsm8k_math_gap": gap >= float(gate["minimum_gsm8k_math_gap"]),
        "ordered_trend_significance": trend["one_sided_p"] <= float(gate["maximum_one_sided_trend_p"]),
    }
    inference_valid = all(
        checks[name]
        for name in (
            "record_count",
            "unique_queries",
            "dataset_counts",
            "stream_identity",
            "execution",
            "first_line_marker",
            "stored_answer_agreement",
            "no_truncation",
        )
    )
    return {
        "n_run_records": len(runs),
        "n_unique_queries": len({record.get("qid") for record in runs}),
        "dataset_counts": counts,
        "correct": correct,
        "accuracy": accuracy,
        "wilson_95_ci": intervals,
        "gsm8k_math_gap": gap,
        "ordered_trend": trend,
        "success_rate": success_rate,
        "first_line_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "stream_sequence_sha256": stream_sequence_sha256(runs),
        "inference_valid": inference_valid,
        "checks": checks,
        "overall_gate_pass": all(checks.values()),
    }


def _write_report(path: Path, result: dict[str, Any], gate: dict[str, Any]) -> None:
    lines = [
        "# Section 56 Parser-Controlled Accepted-Proposal S1 Pilot",
        "",
        "This assessment is call-free. It applies the gate frozen before the",
        "homogeneous answer-first rerun and does not pool historical records.",
        "",
    ]
    for model, model_result in result["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Successful calls: {model_result['success_rate']:.3f}",
                f"- First-line typed answers: {model_result['first_line_marker_rate']:.3f}",
                f"- Stored/parser agreement: {model_result['stored_answer_agreement_rate']:.3f}",
                f"- Token-ceiling/length truncations: {model_result['truncations']}",
                "",
                "| Dataset | Correct | N | CQ | 95% Wilson CI |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for dataset in DATASETS:
            interval = model_result["wilson_95_ci"][dataset]
            lines.append(
                f"| {dataset} | {model_result['correct'][dataset]} | "
                f"{model_result['dataset_counts'][dataset]} | "
                f"{model_result['accuracy'][dataset]:.3f} | "
                f"[{interval[0]:.3f}, {interval[1]:.3f}] |"
            )
        lines.extend(
            [
                "",
                f"GSM8K-MATH gap: **{model_result['gsm8k_math_gap']:.3f}** "
                f"(required >= {float(gate['minimum_gsm8k_math_gap']):.3f}).",
                f"Ordered trend: z={model_result['ordered_trend']['z']:.3f}, "
                f"one-sided p={model_result['ordered_trend']['one_sided_p']:.4f}.",
                "",
                "| Check | Pass |",
                "| --- | --- |",
            ]
        )
        for name, passed in model_result["checks"].items():
            lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
        lines.extend(["", f"Model gate: **{'PASS' if model_result['overall_gate_pass'] else 'FAIL'}**.", ""])

    lines.extend(
        [
            "## Decision",
            "",
            f"Overall frozen gate: **{'PASS' if result['overall_gate_pass'] else 'FAIL'}**.",
            "Both model gates are conjunctive. A pass permits planning the separate",
            "K=5/L=3 metric-health run; it is not itself early-warning evidence.",
            "A failure remains a recorded pilot result and does not trigger threshold",
            "relaxation or selective reruns.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_proposal_s1_answer_first_pilot(
    result_dir: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    gate_path = Path(gate_path)
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    model_results: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for model in gate["expected_models"]:
        seed = int(gate["expected_seed"])
        window = int(gate["expected_window_size"])
        record_path = result_dir / "data" / f"records_S1_{model}_seed{seed}_W{window}.jsonl"
        records = [
            json.loads(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        model_results[model] = assess_model_records(records, gate)
        source_hashes[str(record_path)] = _sha256(record_path)

    result = {
        "analysis": "section56_proposal_s1_answer_first_pilot_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "models": model_results,
        "source_sha256": source_hashes,
        "overall_gate_pass": all(
            model_result["overall_gate_pass"] for model_result in model_results.values()
        ),
        "gate": gate,
    }
    manifest_path = result_dir / "section56_s1_answer_first_assessment.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(result_dir / "SECTION56_S1_ANSWER_FIRST_ASSESSMENT.md", result, gate)
    return result
