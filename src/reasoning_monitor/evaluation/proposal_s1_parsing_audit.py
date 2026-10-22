"""Call-free answer observability audit for the accepted-proposal S1 run."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.metrics.dimensions import answers_equal

MODELS = ("gpt4o_mini", "llama3_8b")
DATASETS = ("gsm8k", "arc_challenge", "math")
MAX_OUTPUT_TOKENS = 512

_ARC_PATTERNS = (
    re.compile(
        r"(?i)answer\s*:\s*(?:answer\s*:\s*)?[\[($]?\s*([A-D])\b"
    ),
    re.compile(r"(?i)final\s+answer\s+is\s*:?\s*[\[($]?\s*([A-D])\b"),
    re.compile(r"(?i)correct\s+answer\s+is\s*:?\s*[\[($]?\s*([A-D])\b"),
    re.compile(r"(?im)^\s*(?:option|choice)\s*[:\-]?\s*([A-D])\b"),
)
_NUMERIC_MARKERS = (
    re.compile(r"(?im)^\s*answer\s*:\s*(.+)$"),
    re.compile(r"(?im)^\s*final\s+answer(?:\s+is)?\s*:\s*(.+)$"),
    re.compile(r"(?im)^\s*####\s*(.+)$"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_balanced_boxed(text: str) -> tuple[str | None, int]:
    marker = r"\boxed{"
    starts = [index for index in range(len(text)) if text.startswith(marker, index)]
    for start in reversed(starts):
        depth = 1
        content_start = start + len(marker)
        for position in range(content_start, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:position].strip(), start
    return None, -1


def extract_strict_final_answer(
    raw_text: str,
    dataset: str,
    ceiling_hit: bool,
) -> tuple[str | None, str]:
    """Extract only an observable final answer, preserving censored outputs."""
    boxed, boxed_position = _last_balanced_boxed(raw_text)

    if dataset == "arc_challenge":
        candidates: list[tuple[int, str, str]] = []
        if boxed and re.fullmatch(r"\s*[A-D]\s*", boxed, re.IGNORECASE):
            prefix = raw_text[max(0, boxed_position - 160) : boxed_position]
            if re.search(r"(?:final|correct)\s+answer", prefix, re.IGNORECASE):
                candidates.append((boxed_position, boxed.strip().upper(), "final_boxed"))
        for pattern in _ARC_PATTERNS:
            candidates.extend(
                (match.start(), match.group(1).upper(), "explicit_choice")
                for match in pattern.finditer(raw_text)
            )
        if not candidates:
            return None, "unresolved_choice"
        _, candidate, method = max(candidates, key=lambda item: item[0])
        return candidate, method

    if boxed is not None:
        prefix = raw_text[max(0, boxed_position - 160) : boxed_position]
        if not ceiling_hit or re.search(
            r"(?:final\s+answer|answer\s*:)", prefix, re.IGNORECASE
        ):
            return boxed, "boxed"

    marked: list[tuple[int, str]] = []
    for pattern in _NUMERIC_MARKERS:
        marked.extend(
            (match.start(), match.group(1).strip()) for match in pattern.finditer(raw_text)
        )
    if marked:
        _, candidate = max(marked, key=lambda item: item[0])
        return candidate.strip().strip("$").rstrip(".;"), "explicit_answer"

    if ceiling_hit:
        return None, "censored_at_ceiling"
    return raw_text, "complete_fallback"


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _audit_model(
    model: str,
    records: list[dict[str, Any]],
    per_query: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    online = per_query.set_index("qid")
    record_rows: list[dict[str, Any]] = []
    for record in records:
        dataset = str(record["dataset"])
        ceiling_hit = int(record.get("n_tokens") or 0) >= MAX_OUTPUT_TOKENS
        candidate, method = extract_strict_final_answer(
            str(record.get("raw_text") or ""), dataset, ceiling_hit
        )
        gold = str(record.get("gold_answer") or "")
        stored = str(record.get("answer") or "")
        record_rows.append(
            {
                "model": model,
                "qid": str(record["qid"]),
                "t": int(record["t"]),
                "dataset": dataset,
                "role": str(record["role"]),
                "call_index": int(record["call_index"]),
                "n_tokens": int(record.get("n_tokens") or 0),
                "ceiling_hit": ceiling_hit,
                "strict_method": method,
                "strict_answer": candidate,
                "strict_resolved": candidate is not None,
                "stored_cq": answers_equal(stored, gold, dataset),
                "raw_current_cq": answers_equal(
                    str(record.get("raw_text") or ""), gold, dataset
                ),
                "strict_cq": (
                    answers_equal(candidate, gold, dataset)
                    if candidate is not None
                    else None
                ),
            }
        )
    record_frame = pd.DataFrame(record_rows)

    primary_rows: list[dict[str, Any]] = []
    primary = record_frame.loc[
        (record_frame["role"] == "run") & (record_frame["call_index"] == 0)
    ]
    for dataset, group in primary.groupby("dataset", sort=False):
        online_values = online.loc[group["qid"], "CQ_raw"].astype(float)
        resolved = group.loc[group["strict_resolved"]]
        strict_correct = int(resolved["strict_cq"].astype(bool).sum())
        unresolved = len(group) - len(resolved)
        primary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n_queries": len(group),
                "ceiling_hits": int(group["ceiling_hit"].sum()),
                "unresolved": unresolved,
                "online_correct": int(online_values.sum()),
                "online_cq": float(online_values.mean()),
                "stored_rescore_correct": int(group["stored_cq"].sum()),
                "stored_rescore_cq": float(group["stored_cq"].mean()),
                "generic_raw_reparse_correct": int(group["raw_current_cq"].sum()),
                "generic_raw_reparse_cq": float(group["raw_current_cq"].mean()),
                "strict_correct": strict_correct,
                "strict_complete_case_cq": (
                    strict_correct / len(resolved) if len(resolved) else float("nan")
                ),
                "strict_cq_lower": strict_correct / len(group),
                "strict_cq_upper": (strict_correct + unresolved) / len(group),
            }
        )

    cs_rows: list[dict[str, Any]] = []
    run_frame = record_frame.loc[record_frame["role"] == "run"]
    for qid, group in run_frame.groupby("qid", sort=False):
        group = group.sort_values("call_index")
        dataset = str(group.iloc[0]["dataset"])
        candidates = group["strict_answer"].tolist()
        complete = all(pd.notna(candidate) for candidate in candidates)
        strict_cs = float("nan")
        if complete:
            observed_candidates = [str(candidate) for candidate in candidates]
            pairs = list(combinations(observed_candidates, 2))
            strict_cs = sum(
                answers_equal(left, right, dataset) for left, right in pairs
            ) / len(pairs)
        online_cs = float(online.loc[qid, "CS_raw"])
        cs_rows.append(
            {
                "model": model,
                "qid": qid,
                "dataset": dataset,
                "complete": complete,
                "online_cs": online_cs,
                "strict_cs": strict_cs,
                "changed": complete and abs(strict_cs - online_cs) > 1e-12,
            }
        )

    return record_frame, pd.DataFrame(primary_rows), pd.DataFrame(cs_rows)


def _write_report(
    path: Path,
    records: pd.DataFrame,
    primary: pd.DataFrame,
    cs: pd.DataFrame,
) -> None:
    run_records = records.loc[records["role"] == "run"]
    lines = [
        "# Section 55 Accepted Proposal S1 Parsing Audit",
        "",
        "This call-free audit re-reads the immutable Section 37-38 S1 records.",
        "It separates parser disagreement from outputs whose final answer was never",
        "observed because generation reached the 512-token ceiling.",
        "",
        "## Run Answer Observability",
        "",
        "| Model | Dataset | Runs | Ceiling hits | Unresolved final answers |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    grouped = (
        run_records.groupby(["model", "dataset"], sort=False)
        .agg(
            runs=("qid", "size"),
            ceiling_hits=("ceiling_hit", "sum"),
            unresolved=("strict_resolved", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dataset} | {row.runs} | "
            f"{row.ceiling_hits} | {row.unresolved} |"
        )

    lines.extend(
        [
            "",
            "## Primary CQ Reassessment",
            "",
            "| Model | Dataset | Online CQ | Stored-answer rescore | Resolved | "
            "Complete-case CQ | CQ bounds |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in primary.itertuples(index=False):
        resolved = row.n_queries - row.unresolved
        lines.append(
            f"| {row.model} | {row.dataset} | {row.online_cq:.3f} | "
            f"{row.stored_rescore_cq:.3f} | {resolved}/{row.n_queries} | "
            f"{row.strict_complete_case_cq:.3f} | "
            f"[{row.strict_cq_lower:.3f}, {row.strict_cq_upper:.3f}] |"
        )

    lines.extend(
        [
            "",
            "## CS Observability",
            "",
            "| Model | Dataset | Complete K=5 queries | CS changed on complete queries |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    cs_summary = (
        cs.groupby(["model", "dataset"], sort=False)
        .agg(
            complete_queries=("complete", "sum"),
            changed_queries=("changed", "sum"),
        )
        .reset_index()
    )
    for row in cs_summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.dataset} | {row.complete_queries}/30 | "
            f"{row.changed_queries} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The original S1 answer stream is not fully observable. MATH censoring is",
            "differential and large, so its CQ level and the timing of CQ/CS changes",
            "cannot be identified from the saved run. Llama ARC also contains genuine",
            "answer-extraction errors when boxed choices were embedded in full prose.",
            "Consequently, the frozen S1 early-warning failure remains an historical",
            "result but is not a clean test of the scientific hypothesis.",
            "",
            "The minimal corrective experiment is a homogeneous answer-first rerun of",
            "all S1 cells with a larger token ceiling. Multiple-choice is useful for",
            "auditing, but changing all datasets is not required: a typed first-line",
            "answer contract can make GSM8K, ARC-Challenge and MATH equally observable",
            "while preserving the accepted-proposal stream.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_proposal_s1_parsing(project_root: str | Path) -> dict[str, Any]:
    """Audit S1 answer observability without issuing model calls."""
    project_root = Path(project_root)
    source_dir = project_root / "results" / "proposal_s1_kl_validation_gpt_llama" / "data"
    output_dir = project_root / "results" / "proposal_s1_parsing_audit"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    record_frames = []
    primary_frames = []
    cs_frames = []
    source_hashes: dict[str, str] = {}
    for model in MODELS:
        records_path = source_dir / f"records_S1_{model}_seed42_W10.jsonl"
        per_query_path = source_dir / f"per_query_S1_{model}_seed42_W10.csv"
        records = _load_records(records_path)
        per_query = pd.read_csv(per_query_path)
        record_frame, primary_frame, cs_frame = _audit_model(
            model, records, per_query
        )
        record_frames.append(record_frame)
        primary_frames.append(primary_frame)
        cs_frames.append(cs_frame)
        for source in (records_path, per_query_path):
            source_hashes[str(source.relative_to(project_root))] = _sha256(source)

    records_frame = pd.concat(record_frames, ignore_index=True)
    primary_frame = pd.concat(primary_frames, ignore_index=True)
    cs_frame = pd.concat(cs_frames, ignore_index=True)
    records_frame.to_csv(data_dir / "s1_parsing_records.csv", index=False)
    primary_frame.to_csv(data_dir / "s1_primary_cq_audit.csv", index=False)
    cs_frame.to_csv(data_dir / "s1_cs_audit.csv", index=False)
    _write_report(
        output_dir / "SECTION55_S1_PARSING_AUDIT.md",
        records_frame,
        primary_frame,
        cs_frame,
    )

    result = {
        "analysis": "section55_proposal_s1_answer_observability_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "source_sha256": source_hashes,
        "primary_cq": primary_frame.to_dict(orient="records"),
    }
    (output_dir / "section55_s1_parsing_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
