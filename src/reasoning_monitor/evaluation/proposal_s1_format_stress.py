"""Call-free assessment of the Section 59 answer-format stress pilot."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from reasoning_monitor.metrics.dimensions import answers_equal

_FINAL_ANSWER_LINE = re.compile(
    r"(?im)^[ \t]*FINAL_ANSWER[ \t]*:[ \t]*(\S[^\r\n]*?)[ \t]*$"
)
_NUMBERED_LINE = re.compile(r"(?im)^[ \t]*(?:Step[ \t]*)?(\d+)[.):][ \t]+")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_sequence_sha256(records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: int(record["t"]))
    payload = "\n".join(
        f"{record['dataset']}|{record['qid']}" for record in ordered
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_final_line_answer(raw_text: str) -> str | None:
    text = raw_text.rstrip()
    matches = list(_FINAL_ANSWER_LINE.finditer(text))
    if "```" in text or len(matches) != 1 or matches[0].end() != len(text):
        return None
    return matches[0].group(1).strip()


def _violates_step_contract(raw_text: str) -> bool:
    step_numbers = [int(value) for value in _NUMBERED_LINE.findall(raw_text)]
    return len(step_numbers) > 6 or any(value > 6 for value in step_numbers)


def assess_model_records(
    records: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    runs = [
        record
        for record in records
        if record.get("role") == "run"
        and int(record.get("call_index", -1)) == 0
    ]
    expected_total = int(gate["expected_queries_per_model"])
    max_tokens = int(gate["max_output_tokens"])
    failure_qids = set(gate["expected_failure_qids"])
    control_qids = set(gate["expected_control_qids"])
    extracted = [
        _strict_final_line_answer(str(record.get("raw_text") or ""))
        for record in runs
    ]

    success_rate = sum(bool(record.get("ok")) for record in runs) / max(
        len(runs), 1
    )
    marker_rate = sum(value is not None for value in extracted) / max(
        len(runs), 1
    )
    agreement_rate = sum(
        candidate is not None
        and answers_equal(
            str(record.get("answer") or ""),
            candidate,
            "math",
        )
        for record, candidate in zip(runs, extracted, strict=True)
    ) / max(len(runs), 1)
    truncations = sum(
        record.get("finish_reason") == "length"
        or int(record.get("n_tokens") or 0) >= max_tokens
        for record in runs
    )
    step_contract_violations = sum(
        _violates_step_contract(str(record.get("raw_text") or ""))
        for record in runs
    )

    def subgroup(qids: set[str]) -> dict[str, Any]:
        paired = [
            (record, candidate)
            for record, candidate in zip(runs, extracted, strict=True)
            if record.get("qid") in qids
        ]
        correct = sum(
            candidate is not None
            and answers_equal(
                candidate,
                str(record.get("gold_answer") or ""),
                "math",
            )
            for record, candidate in paired
        )
        return {
            "n": len(paired),
            "correct": correct,
            "accuracy": correct / max(len(paired), 1),
            "observable": sum(candidate is not None for _, candidate in paired),
        }

    observed_qids = {str(record.get("qid")) for record in runs}
    checks = {
        "record_count": len(runs) == expected_total,
        "unique_queries": len(observed_qids) == expected_total,
        "math_only": all(record.get("dataset") == "math" for record in runs),
        "selected_qids": observed_qids == failure_qids | control_qids,
        "stream_identity": (
            stream_sequence_sha256(runs) == gate["expected_sequence_sha256"]
        ),
        "execution": success_rate >= float(gate["minimum_success_rate"]),
        "final_line_marker": (
            marker_rate >= float(gate["minimum_final_line_marker_rate"])
        ),
        "stored_answer_agreement": (
            agreement_rate
            >= float(gate["minimum_stored_answer_agreement_rate"])
        ),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "step_contract": (
            step_contract_violations
            <= int(gate["maximum_step_contract_violations"])
        ),
    }
    return {
        "n_run_records": len(runs),
        "n_unique_queries": len(observed_qids),
        "success_rate": success_rate,
        "final_line_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "step_contract_violations": step_contract_violations,
        "subgroups": {
            "section58_failures": subgroup(failure_qids),
            "controls": subgroup(control_qids),
        },
        "stream_sequence_sha256": stream_sequence_sha256(runs),
        "checks": checks,
        "overall_gate_pass": all(checks.values()),
    }


def _write_report(
    path: Path,
    result: dict[str, Any],
    section: int,
) -> None:
    title = (
        "Answer-Format Stress Pilot"
        if section == 59
        else "Llama-4 Scout Substitution Stress Arm"
    )
    decision = (
        "A pass permits planning a new homogeneous 90-question validation "
        "with this prompt; it does not alter or pass Section 58."
        if section == 59
        else "A pass permits planning a full 90-question Llama-4 Scout "
        "K=1/L=0 validation; it does not replace Llama-3 results."
    )
    lines = [
        f"# Section {section} {title}",
        "",
        "This call-free assessment applies the operational gate frozen before",
        "the enriched development pilot. Accuracy is descriptive only.",
        "",
    ]
    for model, model_result in result["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Successful calls: {model_result['success_rate']:.3f}",
                f"- Strict final-line markers: {model_result['final_line_marker_rate']:.3f}",
                f"- Stored/parser agreement: {model_result['stored_answer_agreement_rate']:.3f}",
                f"- Truncations: {model_result['truncations']}",
                f"- Step-contract violations: {model_result['step_contract_violations']}",
                "",
                "| Subgroup | Correct | N | CQ | Observable |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, subgroup in model_result["subgroups"].items():
            lines.append(
                f"| {name} | {subgroup['correct']} | {subgroup['n']} | "
                f"{subgroup['accuracy']:.3f} | {subgroup['observable']} |"
            )
        lines.extend(["", "| Check | Pass |", "| --- | --- |"])
        for name, passed in model_result["checks"].items():
            lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
        lines.extend(
            [
                "",
                f"Model gate: **{'PASS' if model_result['overall_gate_pass'] else 'FAIL'}**.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"Overall frozen gate: **{'PASS' if result['overall_gate_pass'] else 'FAIL'}**.",
            decision,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def assess_proposal_s1_format_stress(
    result_dir: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    gate_path = Path(gate_path)
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    section = int(gate["section"])
    seed = int(gate["expected_seed"])
    window = int(gate["expected_window_size"])
    model_results: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for model in gate["expected_models"]:
        record_path = (
            result_dir
            / "data"
            / f"records_S1_{model}_seed{seed}_W{window}.jsonl"
        )
        records = [
            json.loads(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        model_results[model] = assess_model_records(records, gate)
        source_hashes[str(record_path)] = _sha256(record_path)

    result = {
        "analysis": f"section{section}_proposal_s1_format_stress_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "models": model_results,
        "source_sha256": source_hashes,
        "overall_gate_pass": all(
            model_result["overall_gate_pass"]
            for model_result in model_results.values()
        ),
        "gate": gate,
    }
    (result_dir / f"section{section}_format_stress_assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(
        result_dir / f"SECTION{section}_FORMAT_STRESS_ASSESSMENT.md",
        result,
        section,
    )
    return result
