"""Call-free assessment of the frozen Section 52 SuperGPQA pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import fisher_exact, norm

from reasoning_monitor.metrics.dimensions import answers_equal

TIERS = ("easy", "medium", "hard")
_FINAL_CHOICE = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-J])\b")
_FIRST_LINE_CHOICE = re.compile(r"(?i)\A\s*FINAL_CHOICE\s*:\s*([A-J])\b")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
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
    successes = [correct[tier] for tier in TIERS]
    totals = [counts[tier] for tier in TIERS]
    total = sum(totals)
    pooled = sum(successes) / total
    statistic = sum(
        score * (success - tier_total * pooled)
        for score, success, tier_total in zip(scores, successes, totals, strict=True)
    )
    variance = pooled * (1 - pooled) * (
        sum(tier_total * score * score for tier_total, score in zip(totals, scores, strict=True))
        - sum(tier_total * score for tier_total, score in zip(totals, scores, strict=True))
        ** 2
        / total
    )
    z_value = statistic / math.sqrt(variance) if variance > 0 else 0.0
    return {"z": z_value, "one_sided_p": float(norm.cdf(z_value))}


def assess_records(records: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate frozen execution, parsing and CQ range gates."""
    runs = [record for record in records if record.get("role") == "run"]
    expected_per_tier = int(gate["expected_queries_per_tier"])
    expected_total = expected_per_tier * len(TIERS)
    tier_rows = {
        tier: [record for record in runs if record.get("difficulty") == tier]
        for tier in TIERS
    }

    success_rate = sum(bool(record.get("ok")) for record in runs) / max(len(runs), 1)
    valid_choice_rate = sum(
        str(record.get("answer", "")).strip().upper() in "ABCDEFGHIJ"
        and len(str(record.get("answer", "")).strip()) == 1
        for record in runs
    ) / max(len(runs), 1)
    marker_rate = sum(
        bool(_FINAL_CHOICE.search(str(record.get("raw_text", "")))) for record in runs
    ) / max(len(runs), 1)
    first_line_marker_rate = sum(
        bool(_FIRST_LINE_CHOICE.search(str(record.get("raw_text", ""))))
        for record in runs
    ) / max(len(runs), 1)
    max_output_tokens = int(gate["max_output_tokens"])
    truncations = sum(
        record.get("finish_reason") == "length"
        or int(record.get("n_tokens") or 0) >= max_output_tokens
        for record in runs
    )

    accuracy = {}
    correct = {}
    valid_counts = {}
    complete_case_accuracy = {}
    accuracy_upper_bound = {}
    for tier, rows in tier_rows.items():
        count = sum(
            answers_equal(
                str(row.get("answer", "")),
                str(row.get("gold_answer", "")),
                "supergpqa",
            )
            for row in rows
        )
        correct[tier] = count
        accuracy[tier] = count / max(len(rows), 1)
        valid_rows = [
            row
            for row in rows
            if len(str(row.get("answer", "")).strip()) == 1
            and str(row.get("answer", "")).strip().upper() in "ABCDEFGHIJ"
        ]
        valid_counts[tier] = len(valid_rows)
        valid_correct = sum(
            answers_equal(
                str(row.get("answer", "")),
                str(row.get("gold_answer", "")),
                "supergpqa",
            )
            for row in valid_rows
        )
        complete_case_accuracy[tier] = valid_correct / max(len(valid_rows), 1)
        accuracy_upper_bound[tier] = (
            count + len(rows) - len(valid_rows)
        ) / max(len(rows), 1)

    gap = accuracy["easy"] - accuracy["hard"]
    monotonic = accuracy["easy"] >= accuracy["medium"] >= accuracy["hard"]
    range_pass = all(
        float(gate["minimum_tier_accuracy"])
        <= accuracy[tier]
        <= float(gate["maximum_tier_accuracy"])
        for tier in TIERS
    )
    checks = {
        "record_count": len(runs) == expected_total,
        "unique_queries": len({record.get("qid") for record in runs}) == expected_total,
        "tier_counts": all(len(tier_rows[tier]) == expected_per_tier for tier in TIERS),
        "execution": (
            success_rate >= float(gate["minimum_success_rate"])
            and truncations == 0
        ),
        "valid_choice": valid_choice_rate >= float(gate["minimum_valid_choice_rate"]),
        "final_choice_marker": (
            marker_rate >= float(gate["minimum_final_choice_marker_rate"])
        ),
        "monotonic_cq": (
            monotonic if bool(gate["require_monotonic_nonincreasing"]) else True
        ),
        "easy_hard_gap": gap >= float(gate["minimum_easy_hard_gap"]),
        "nondegenerate_tiers": range_pass,
    }
    if "minimum_first_line_marker_rate" in gate:
        checks["first_line_choice_marker"] = (
            first_line_marker_rate >= float(gate["minimum_first_line_marker_rate"])
        )
    trend = _ordered_trend_test(correct, {tier: len(tier_rows[tier]) for tier in TIERS})
    fisher_p = float(
        fisher_exact(
            [
                [correct["easy"], len(tier_rows["easy"]) - correct["easy"]],
                [correct["hard"], len(tier_rows["hard"]) - correct["hard"]],
            ],
            alternative="greater",
        ).pvalue
    )
    if "maximum_one_sided_trend_p" in gate:
        checks["ordered_trend_significance"] = (
            trend["one_sided_p"] <= float(gate["maximum_one_sided_trend_p"])
        )
    inference_valid = (
        checks["execution"]
        and checks["valid_choice"]
        and checks["final_choice_marker"]
    )
    return {
        "n_run_records": len(runs),
        "n_unique_queries": len({record.get("qid") for record in runs}),
        "tier_counts": {tier: len(rows) for tier, rows in tier_rows.items()},
        "correct": correct,
        "accuracy": accuracy,
        "valid_choice_counts": valid_counts,
        "complete_case_accuracy": complete_case_accuracy,
        "accuracy_bounds": {
            tier: [accuracy[tier], accuracy_upper_bound[tier]] for tier in TIERS
        },
        "wilson_95_ci": {
            tier: _wilson_interval(correct[tier], len(tier_rows[tier])) for tier in TIERS
        },
        "ordered_trend": trend,
        "easy_hard_fisher_one_sided_p": fisher_p,
        "inference_valid": inference_valid,
        "easy_hard_gap": gap,
        "success_rate": success_rate,
        "valid_choice_rate": valid_choice_rate,
        "final_choice_marker_rate": marker_rate,
        "first_line_choice_marker_rate": first_line_marker_rate,
        "truncations": truncations,
        "checks": checks,
        "overall_gate_pass": all(checks.values()),
    }


def _write_report(path: Path, result: dict[str, Any], gate: dict[str, Any]) -> None:
    section = int(gate.get("section", 52))
    is_validation = "maximum_one_sided_trend_p" in gate
    run_label = "validation" if is_validation else "pilot"
    lines = [
        f"# Section {section} SuperGPQA GPT-4o-mini Assessment",
        "",
        "This assessment is call-free. It reads the immutable raw records generated",
        f"by the separately authorised {run_label} and applies the gate frozen beforehand.",
        "",
        "## Execution And Parsing",
        "",
        f"- Successful calls: {result['success_rate']:.3f}",
        f"- Truncations: {result['truncations']}",
        f"- Valid extracted A-J choices: {result['valid_choice_rate']:.3f}",
        f"- Anchored `FINAL_CHOICE` markers: {result['final_choice_marker_rate']:.3f}",
        f"- First-line `FINAL_CHOICE` markers: "
        f"{result['first_line_choice_marker_rate']:.3f}",
        f"- Token-ceiling/length truncations: {result['truncations']}",
        "",
        "## CQ Screening",
        "",
        "| Tier | Correct | Valid choices | N | CQ | 95% Wilson CI | CQ bounds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in TIERS:
        lines.append(
            f"| {tier.title()} | {result['correct'][tier]} | "
            f"{result['valid_choice_counts'][tier]} | {result['tier_counts'][tier]} | "
            f"{result['accuracy'][tier]:.3f} | "
            f"[{result['wilson_95_ci'][tier][0]:.3f}, "
            f"{result['wilson_95_ci'][tier][1]:.3f}] | "
            f"[{result['accuracy_bounds'][tier][0]:.3f}, "
            f"{result['accuracy_bounds'][tier][1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            f"Easy-Hard gap: **{result['easy_hard_gap']:.3f}** "
            f"(required >= {float(gate['minimum_easy_hard_gap']):.3f}).",
            f"Ordered trend: z={result['ordered_trend']['z']:.3f}, "
            f"one-sided p={result['ordered_trend']['one_sided_p']:.4f}.",
            f"Easy>Hard Fisher exact one-sided p="
            f"{result['easy_hard_fisher_one_sided_p']:.4f}.",
            f"Inferential statistics valid (no censoring/parser loss): "
            f"**{result['inference_valid']}**.",
            "",
            "## Frozen Gate",
            "",
            "| Check | Pass |",
            "| --- | --- |",
        ]
    )
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"Overall {run_label} gate: "
            f"**{'PASS' if result['overall_gate_pass'] else 'FAIL'}**.",
            "",
        ]
    )
    if result["inference_valid"]:
        lines.extend(
            [
                "All choices are observed, so CQ bounds collapse to the point estimates",
                "and the reported trend/Fisher statistics are evaluable.",
            ]
        )
    else:
        lines.extend(
            [
                "Invalid choices are censored token-ceiling outputs, not observed wrong",
                "choices. Conservative CQ treats them as wrong; the bounds treat all of",
                "them as either wrong or correct. Differential censoring prevents a clean",
                "difficulty interpretation when these intervals overlap substantially.",
                "Trend and Fisher p-values are not scientifically interpretable here;",
                "they are retained only as raw diagnostics.",
            ]
        )
    if is_validation:
        lines.extend(
            [
                f"The frozen validation threshold is one-sided ordered-trend "
                f"p <= {float(gate['maximum_one_sided_trend_p']):.3f}.",
                "All frozen conditions are conjunctive. A failed validation is retained",
                "as a failure; it does not authorise threshold relaxation or automatic",
                "cross-model expansion.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No significance threshold is used for this 20-per-tier screening run.",
                "A pass authorises consideration of the already frozen disjoint validation",
                "pool; it is not itself confirmatory evidence.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_supergpqa_pilot(
    result_dir: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    gate_path = Path(gate_path)
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    record_files = sorted((result_dir / "data").glob("records_ST2_gpt4o_mini_*.jsonl"))
    if len(record_files) != 1:
        raise FileNotFoundError(
            f"expected one GPT pilot record file in {result_dir / 'data'}, "
            f"found {len(record_files)}"
        )
    record_path = record_files[0]
    records = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = assess_records(records, gate)
    section = int(gate.get("section", 52))
    result.update(
        {
            "analysis": f"section{section}_supergpqa_gpt4o_mini_gate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "no_model_calls": True,
            "record_file": str(record_path),
            "record_sha256": _sha256(record_path),
            "gate": gate,
        }
    )
    manifest_path = result_dir / f"section{section}_supergpqa_assessment.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(
        result_dir / f"SECTION{section}_SUPERGPQA_ASSESSMENT.md", result, gate
    )
    return result
