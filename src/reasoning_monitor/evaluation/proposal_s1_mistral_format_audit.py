"""Call-free format sensitivity audit for the Section 62 Mistral records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reasoning_monitor.evaluation.proposal_s1_answer_last_validation import (
    DATASETS,
    _ordered_trend_test,
    _strict_final_line_answer,
    stream_sequence_sha256,
)
from reasoning_monitor.metrics.dimensions import answers_equal

EXPECTED_RECORD_SHA256 = (
    "e28b727e0befed5273a6214c244ccb702b972b528abbab8fca7d577e128a4429"
)
EXPECTED_STREAM_SHA256 = (
    "791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca"
)
_TYPED_LINE = re.compile(
    r"(?im)^[ \t]*(?:FINAL_CHOICE|FINAL_ANSWER)[ \t]*:[^\r\n]*$"
)
_RELAXED_CHOICE = re.compile(r"(?im)FINAL_CHOICE[ \t]*:[ \t]*([A-D])\b")
_RELAXED_ANSWER = re.compile(
    r"(?im)FINAL_ANSWER[ \t]*:[ \t]*([^\r\n]+)"
)
_NUMBERED_MARKER = re.compile(
    r"(?im)^[ \t]*(?:Step[ \t]*)?\d+[.):].*"
    r"FINAL_(?:ANSWER|CHOICE)[ \t]*:"
)
_CHOICE_WITH_EXTRA_TEXT = re.compile(
    r"(?im)^[ \t]*FINAL_CHOICE[ \t]*:[ \t]*[A-D][ \t]*\S+"
)


def extract_relaxed_typed_answer(raw_text: str, dataset: str) -> str | None:
    """Recover the last explicit typed marker without requiring its own line."""
    pattern = _RELAXED_CHOICE if dataset == "arc_challenge" else _RELAXED_ANSWER
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def classify_strict_format_failure(raw_text: str, dataset: str) -> str | None:
    """Name why a response fails the frozen Section 62 strict contract."""
    if _strict_final_line_answer(raw_text, dataset, True) is not None:
        return None
    if "```" in raw_text:
        return "code_fence"
    if _NUMBERED_MARKER.search(raw_text):
        return "marker_inside_numbered_step"
    if dataset == "arc_challenge" and _CHOICE_WITH_EXTRA_TEXT.search(raw_text):
        return "choice_marker_has_extra_text"
    typed_lines = list(_TYPED_LINE.finditer(raw_text.rstrip()))
    if len(typed_lines) > 1:
        return "multiple_typed_markers"
    if typed_lines:
        return "typed_marker_not_strict_final"
    if re.search(r"(?i)FINAL_(?:ANSWER|CHOICE)[ \t]*:", raw_text):
        return "embedded_marker"
    return "no_typed_marker"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the frozen strict parser with an explicit-marker sensitivity."""
    runs = [
        record
        for record in records
        if record.get("role") == "run"
        and int(record.get("call_index", -1)) == 0
    ]
    category_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    per_dataset: dict[str, dict[str, Any]] = {}
    relaxed_correct: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}

    for dataset in DATASETS:
        rows = [record for record in runs if record.get("dataset") == dataset]
        dataset_counts[dataset] = len(rows)
        strict_observable = 0
        strict_correct = 0
        relaxed_observable = 0
        relaxed_dataset_correct = 0
        dataset_categories: Counter[str] = Counter()

        for record in rows:
            raw_text = str(record.get("raw_text") or "")
            gold = str(record.get("gold_answer") or "")
            strict = _strict_final_line_answer(raw_text, dataset, True)
            relaxed = extract_relaxed_typed_answer(raw_text, dataset)
            strict_observable += strict is not None
            strict_correct += strict is not None and answers_equal(
                strict, gold, dataset
            )
            relaxed_observable += relaxed is not None
            relaxed_is_correct = relaxed is not None and answers_equal(
                relaxed, gold, dataset
            )
            relaxed_dataset_correct += relaxed_is_correct

            category = classify_strict_format_failure(raw_text, dataset)
            if category is not None:
                category_counts[category] += 1
                dataset_categories[category] += 1
                details.append(
                    {
                        "qid": str(record.get("qid")),
                        "dataset": dataset,
                        "category": category,
                        "gold_answer": gold,
                        "stored_answer": str(record.get("answer") or ""),
                        "relaxed_answer": relaxed,
                        "relaxed_correct": relaxed_is_correct,
                    }
                )

        relaxed_correct[dataset] = relaxed_dataset_correct
        per_dataset[dataset] = {
            "n": len(rows),
            "strict_observable": strict_observable,
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / max(len(rows), 1),
            "relaxed_observable": relaxed_observable,
            "relaxed_correct": relaxed_dataset_correct,
            "relaxed_accuracy": relaxed_dataset_correct / max(len(rows), 1),
            "strict_failure_categories": dict(dataset_categories),
        }

    relaxed_accuracy = {
        dataset: relaxed_correct[dataset] / max(dataset_counts[dataset], 1)
        for dataset in DATASETS
    }
    return {
        "n_run_records": len(runs),
        "strict_observable": sum(
            values["strict_observable"] for values in per_dataset.values()
        ),
        "relaxed_observable": sum(
            values["relaxed_observable"] for values in per_dataset.values()
        ),
        "strict_format_failures": sum(category_counts.values()),
        "failure_categories": dict(category_counts),
        "datasets": per_dataset,
        "relaxed_accuracy": relaxed_accuracy,
        "relaxed_gsm8k_math_gap": (
            relaxed_accuracy["gsm8k"] - relaxed_accuracy["math"]
        ),
        "relaxed_ordered_trend": _ordered_trend_test(
            relaxed_correct, dataset_counts
        ),
        "relaxed_monotonic_nonincreasing": (
            relaxed_accuracy["gsm8k"]
            >= relaxed_accuracy["arc_challenge"]
            >= relaxed_accuracy["math"]
        ),
        "failure_details": details,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Section 63 Mistral Format Sensitivity Audit",
        "",
        "This call-free audit preserves the immutable Section 62 records and",
        "compares the frozen strict final-line contract with a deterministic",
        "sensitivity parser that reads only explicit `FINAL_*` markers.",
        "",
        "## Observability",
        "",
        f"- Strictly observable: {result['strict_observable']}/90",
        f"- Explicit-marker sensitivity observable: {result['relaxed_observable']}/90",
        f"- Strict format failures: {result['strict_format_failures']}",
        "",
        "| Dataset | Strict observable | Strict correct | Sensitivity observable | Sensitivity correct | Sensitivity CQ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        values = result["datasets"][dataset]
        lines.append(
            f"| {dataset} | {values['strict_observable']}/{values['n']} | "
            f"{values['strict_correct']}/{values['n']} | "
            f"{values['relaxed_observable']}/{values['n']} | "
            f"{values['relaxed_correct']}/{values['n']} | "
            f"{values['relaxed_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Strict Failure Modes",
            "",
            "| Failure mode | Count |",
            "| --- | ---: |",
        ]
    )
    for category, count in sorted(result["failure_categories"].items()):
        lines.append(f"| {category} | {count} |")

    trend = result["relaxed_ordered_trend"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Sensitivity GSM8K-MATH gap: {result['relaxed_gsm8k_math_gap']:.3f}.",
            f"Sensitivity ordered trend: z={trend['z']:.3f}, "
            f"one-sided p={trend['one_sided_p']:.6f}.",
            "The dataset accuracies are not monotonic because ARC-Challenge is",
            "higher than GSM8K for this model.",
            "",
            "Section 62 remains a formal failure. The sensitivity result shows",
            "that its reported ARC CQ was strongly confounded by output format,",
            "but post-hoc recovery cannot satisfy the frozen gate. The failed",
            "Section 62 gate does not authorise the ASU K=5 repeat-diversity",
            "capability check or a K=5/L=3 campaign.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_mistral_format(project_root: str | Path) -> dict[str, Any]:
    """Run and persist the immutable Section 62 call-free format audit."""
    project_root = Path(project_root)
    source = (
        project_root
        / "results"
        / "proposal_s1_mistral_validation"
        / "data"
        / "records_S1_mistral_7b_seed42_W10.jsonl"
    )
    source_sha256 = _sha256(source)
    records = _load_records(source)
    result = audit_records(records)
    result.update(
        {
            "analysis": "section63_proposal_s1_mistral_format_sensitivity",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "no_model_calls": True,
            "source_file": str(source.relative_to(project_root)),
            "source_sha256": source_sha256,
            "source_sha256_matches": source_sha256 == EXPECTED_RECORD_SHA256,
            "stream_sequence_sha256": stream_sequence_sha256(records),
            "stream_sequence_matches": (
                stream_sequence_sha256(records) == EXPECTED_STREAM_SHA256
            ),
            "section62_decision_unchanged": True,
            "k5_capability_check_authorized": False,
        }
    )

    output_dir = project_root / "results" / "proposal_s1_mistral_format_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output_dir / "ASSESSMENT.md", result)
    return result

