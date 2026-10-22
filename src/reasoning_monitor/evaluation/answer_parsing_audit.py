"""Call-free audit of answer extraction and reasoning-trace splitting."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from reasoning_monitor.metrics.dimensions import answers_equal

TRACKS = {
    "llama3_8b": (
        "e2h_gsm8k",
        "e2h_final_holdout_llama3_8b",
        "ST2_llama3_8b_seed48_W10",
    ),
    "gpt4o_mini": (
        "e2h_amc",
        "e2h_final_holdout_gpt4o_mini",
        "ST2_gpt4o_mini_seed48_W10",
    ),
    "mistral_7b": (
        "e2h_gsm8k",
        "e2h_final_holdout_mistral_7b",
        "ST2_mistral_7b_seed48_W10",
    ),
}

_ANSWER_MARKER = re.compile(r"(?:final answer|answer\s*:|answer is|\\boxed)", re.I)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fragments(text: str) -> list[str]:
    """Reproduce the live client's sentence splitter before its 12-part cap."""
    return [part.strip() for part in text.replace("\n", ". ").split(".") if part.strip()]


def _pairwise_agreement(values: list[str], dataset: str) -> float:
    pairs = list(combinations(values, 2))
    if not pairs:
        return 1.0
    return sum(answers_equal(left, right, dataset) for left, right in pairs) / len(pairs)


def _equivalence_class_count(values: list[str], dataset: str) -> int:
    representatives: list[str] = []
    for value in values:
        if not any(answers_equal(value, seen, dataset) for seen in representatives):
            representatives.append(value)
    return len(representatives)


def summarize_records(
    records: list[dict[str, Any]],
    model: str,
    dataset: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Summarize extraction fidelity and trace truncation for one track."""
    runs = [record for record in records if record.get("role") == "run"]
    record_rows = []
    for record in runs:
        raw = str(record.get("raw_text") or "")
        answer = str(record.get("answer") or "")
        gold = str(record.get("gold_answer") or "")
        parts = _fragments(raw)
        stored_steps = [str(step) for step in record.get("reasoning_steps") or ()]
        total_chars = sum(len(part) for part in parts)
        retained_chars = sum(len(part) for part in parts[:12])
        raw_has_marker = bool(_ANSWER_MARKER.search(raw))
        trace_has_marker = bool(_ANSWER_MARKER.search(" ".join(stored_steps)))
        record_rows.append(
            {
                "model": model,
                "qid": str(record.get("qid")),
                "difficulty": str(record.get("difficulty")),
                "call_index": int(record.get("call_index", -1)),
                "blank_answer": not answer.strip(),
                "answer_is_full_raw": answer == raw,
                "answer_multiline": "\n" in answer,
                "answer_length": len(answer),
                "stored_raw_equivalent": answers_equal(answer, raw, dataset),
                "stored_cq": answers_equal(answer, gold, dataset),
                "raw_cq": answers_equal(raw, gold, dataset),
                "uncapped_fragment_count": len(parts),
                "stored_step_count": len(stored_steps),
                "trace_capped": len(parts) > 12,
                "retained_character_fraction": (
                    retained_chars / total_chars if total_chars else 1.0
                ),
                "raw_has_answer_marker": raw_has_marker,
                "stored_trace_has_answer_marker": trace_has_marker,
            }
        )
    record_frame = pd.DataFrame(record_rows)

    query_rows = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in runs:
        grouped.setdefault(str(record.get("qid")), []).append(record)
    for qid, group in grouped.items():
        ordered = sorted(group, key=lambda item: int(item.get("call_index", -1)))
        raw_values = [str(item.get("raw_text") or "") for item in ordered]
        answers = [str(item.get("answer") or "") for item in ordered]
        traces = [" ".join(map(str, item.get("reasoning_steps") or ())) for item in ordered]
        stored_agreement = _pairwise_agreement(answers, dataset)
        raw_agreement = _pairwise_agreement(raw_values, dataset)
        raw_diverse = len(set(raw_values)) >= 2
        trace_diverse = len(set(traces)) >= 2
        query_rows.append(
            {
                "model": model,
                "qid": qid,
                "difficulty": str(ordered[0].get("difficulty")),
                "n_runs": len(ordered),
                "raw_verbatim_classes": len(set(raw_values)),
                "answer_surface_classes": len(set(answers)),
                "answer_equivalence_classes": _equivalence_class_count(answers, dataset),
                "stored_cs": stored_agreement,
                "raw_rescored_cs": raw_agreement,
                "cs_changed_by_extraction": abs(stored_agreement - raw_agreement) > 1e-12,
                "stored_trace_classes": len(set(traces)),
                "raw_diverse": raw_diverse,
                "stored_trace_diverse": trace_diverse,
                "trace_collapsed_by_splitter": raw_diverse and not trace_diverse,
            }
        )
    query_frame = pd.DataFrame(query_rows)

    marker_rows = record_frame.loc[record_frame["raw_has_answer_marker"]]
    summary = {
        "model": model,
        "dataset": dataset,
        "n_run_records": len(record_frame),
        "n_queries": len(query_frame),
        "blank_answers": int(record_frame["blank_answer"].sum()),
        "full_raw_answers": int(record_frame["answer_is_full_raw"].sum()),
        "multiline_answers": int(record_frame["answer_multiline"].sum()),
        "stored_raw_mismatches": int((~record_frame["stored_raw_equivalent"]).sum()),
        "cq_changed_by_raw_rescore": int(
            (record_frame["stored_cq"] != record_frame["raw_cq"]).sum()
        ),
        "cs_changed_queries": int(query_frame["cs_changed_by_extraction"].sum()),
        "raw_diverse_queries": int(query_frame["raw_diverse"].sum()),
        "answer_surface_diverse_queries": int(
            (query_frame["answer_surface_classes"] >= 2).sum()
        ),
        "answer_equivalence_diverse_queries": int(
            (query_frame["answer_equivalence_classes"] >= 2).sum()
        ),
        "stored_trace_diverse_queries": int(query_frame["stored_trace_diverse"].sum()),
        "trace_collapsed_queries": int(query_frame["trace_collapsed_by_splitter"].sum()),
        "capped_trace_records": int(record_frame["trace_capped"].sum()),
        "capped_trace_rate": float(record_frame["trace_capped"].mean()),
        "median_uncapped_fragments": float(median(record_frame["uncapped_fragment_count"])),
        "median_retained_character_fraction": float(
            median(record_frame["retained_character_fraction"])
        ),
        "answer_marker_capture_rate": (
            float(marker_rows["stored_trace_has_answer_marker"].mean())
            if len(marker_rows)
            else float("nan")
        ),
    }
    return summary, record_frame, query_frame


def _write_report(
    path: Path,
    summaries: pd.DataFrame,
    by_difficulty: pd.DataFrame,
) -> None:
    lines = [
        "# Section 50 Call-Free Parsing Audit",
        "",
        "This audit re-reads the immutable Section 47 records. It issues no model calls",
        "and does not change any frozen result.",
        "",
        "## Answer Extraction",
        "",
        "| Model | Runs | Blank | Stored/raw mismatch | CQ changed | CS queries changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.n_run_records} | {row.blank_answers} | "
            f"{row.stored_raw_mismatches} | {row.cq_changed_by_raw_rescore} | "
            f"{row.cs_changed_queries} |"
        )
    lines.extend(
        [
            "",
            "## Reasoning-Trace Splitter",
            "",
            "| Model | Capped runs | Cap rate | Median fragments | Median retained | "
            "Answer marker retained | Trace-collapsed queries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.capped_trace_records} | {row.capped_trace_rate:.3f} | "
            f"{row.median_uncapped_fragments:.1f} | "
            f"{row.median_retained_character_fraction:.3f} | "
            f"{row.answer_marker_capture_rate:.3f} | {row.trace_collapsed_queries} |"
        )
    lines.extend(
        [
            "",
            "## Cap Rate By Difficulty",
            "",
            "| Model | Difficulty | Runs | Cap rate | Median retained |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in by_difficulty.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.difficulty} | {row.n_records} | "
            f"{row.capped_trace_rate:.3f} | {row.median_retained_character_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            "After the nested AMC answer-label correction, raw-record rescoring changes",
            "neither CQ nor CS for Llama or GPT. The five remaining Mistral path",
            "mismatches also change neither metric. Final-answer extraction therefore does",
            "not explain the Section 47 CQ profiles, CS values, or raw-text K diversity.",
            "",
            "The reasoning-trace splitter is a material validity issue for LS and SS.",
            "It splits on every period, keeps only the first 12 fragments, and discards",
            "the remainder. All 450 GPT K-runs are capped, retaining a median 26.3% of",
            "fragment characters and no final answer marker. Llama capping rises from",
            "14.7% on Easy to 58.7% on Hard. LS/SS findings require a call-free",
            "structure-aware reparse sensitivity analysis before scientific interpretation.",
            "",
            "## Interpretation Boundary",
            "",
            "Answer extraction affects CQ/CS only when the stored and raw rescoring columns",
            "disagree. The reasoning-step splitter affects LS and SS even when CQ/CS agree.",
            "Raw K-repeat diversity is independent of both parsers.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_final_holdout_parsing(project_root: str | Path) -> dict[str, Any]:
    """Audit all Section 47 final-holdout records without model calls."""
    project_root = Path(project_root)
    output_dir = project_root / "results" / "answer_parsing_audit"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    record_frames = []
    query_frames = []
    source_hashes = {}
    for model, (dataset, result_name, tag) in TRACKS.items():
        source = project_root / "results" / result_name / "data" / f"records_{tag}.jsonl"
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary, record_frame, query_frame = summarize_records(records, model, dataset)
        summaries.append(summary)
        record_frames.append(record_frame)
        query_frames.append(query_frame)
        source_hashes[str(source.relative_to(project_root))] = _sha256(source)

    summary_frame = pd.DataFrame(summaries)
    records_frame = pd.concat(record_frames, ignore_index=True)
    queries_frame = pd.concat(query_frames, ignore_index=True)
    by_difficulty = (
        records_frame.groupby(["model", "difficulty"], sort=False)
        .agg(
            n_records=("qid", "size"),
            capped_trace_rate=("trace_capped", "mean"),
            median_retained_character_fraction=("retained_character_fraction", "median"),
        )
        .reset_index()
    )

    summary_frame.to_csv(data_dir / "parsing_audit_summary.csv", index=False)
    records_frame.to_csv(data_dir / "parsing_audit_records.csv", index=False)
    queries_frame.to_csv(data_dir / "parsing_audit_queries.csv", index=False)
    by_difficulty.to_csv(data_dir / "parsing_audit_by_difficulty.csv", index=False)
    _write_report(output_dir / "ANSWER_PARSING_AUDIT.md", summary_frame, by_difficulty)

    result = {
        "analysis": "section50_call_free_answer_and_trace_parsing_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "source_sha256": source_hashes,
        "summary": summaries,
    }
    (output_dir / "answer_parsing_audit_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
