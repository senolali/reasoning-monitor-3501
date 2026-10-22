"""Call-free semantic audit of the completed Section 66 RS pilot."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.metrics.dimensions import answers_equal

_TIME_ANSWER = re.compile(
    r"^\s*\$?([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s*$",
    re.IGNORECASE,
)
_TIME_FACTORS = {
    "second": 1.0,
    "sec": 1.0,
    "minute": 60.0,
    "min": 60.0,
    "hour": 3600.0,
    "hr": 3600.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _time_seconds(answer: str) -> float | None:
    match = _TIME_ANSWER.fullmatch(answer.strip())
    if match is None:
        return None
    unit = match.group(2).lower().rstrip("s")
    factor = _TIME_FACTORS.get(unit)
    if factor is None:
        return None
    return float(match.group(1)) * factor


def time_answers_equivalent(left: str, right: str) -> bool:
    """Return true only when two explicit time-unit answers are equivalent."""
    left_seconds = _time_seconds(left)
    right_seconds = _time_seconds(right)
    return bool(
        left_seconds is not None
        and right_seconds is not None
        and math.isclose(left_seconds, right_seconds, rel_tol=1e-9, abs_tol=1e-9)
    )


def audit_model_records(
    records: list[dict[str, Any]],
    per_query: pd.DataFrame,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)
    raw_rs = {
        str(row.qid): float(row.RS_raw)
        for row in per_query.itertuples()
        if not pd.isna(row.RS_raw)
    }
    corrected: dict[str, float] = {}
    corrections: list[dict[str, Any]] = []

    for qid, raw_value in raw_rs.items():
        rows = grouped[qid]
        primary = next(
            row
            for row in rows
            if row.get("role") == "run" and int(row.get("call_index", -1)) == 0
        )
        dataset = str(primary["dataset"])
        paraphrases = sorted(
            (row for row in rows if row.get("role") == "paraphrase"),
            key=lambda row: int(row["call_index"]),
        )
        semantic_correct = 0
        for row in paraphrases:
            answer = str(row.get("answer") or "")
            gold = str(row.get("gold_answer") or "")
            exact_correct = answers_equal(answer, gold, dataset)
            unit_correct = time_answers_equivalent(
                answer, str(primary.get("answer") or "")
            )
            semantic_correct += int(exact_correct or unit_correct)
            if unit_correct and not exact_correct:
                corrections.append(
                    {
                        "qid": qid,
                        "dataset": dataset,
                        "role": "paraphrase",
                        "call_index": int(row["call_index"]),
                        "primary_answer": primary.get("answer"),
                        "paraphrase_answer": answer,
                        "gold_answer": gold,
                        "raw_rs": raw_value,
                    }
                )
        corrected[qid] = semantic_correct / max(len(paraphrases), 1)

    raw_values = list(raw_rs.values())
    corrected_values = list(corrected.values())
    return {
        "rs_evaluable": len(raw_values),
        "raw_rs_unique_values": len({round(value, 12) for value in raw_values}),
        "raw_rs_range": [min(raw_values), max(raw_values)] if raw_values else None,
        "semantic_rs_unique_values": len(
            {round(value, 12) for value in corrected_values}
        ),
        "semantic_rs_range": (
            [min(corrected_values), max(corrected_values)]
            if corrected_values
            else None
        ),
        "unit_equivalence_corrections": corrections,
        "semantic_rs_by_qid": corrected,
        "semantic_nondegeneracy": len(
            {round(value, 12) for value in corrected_values}
        )
        >= 2,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Section 67 Section 66 Semantic-Equivalence Audit",
        "",
        "This call-free audit preserves the frozen Section 66 decision and",
        "checks whether its RS variation represents genuinely different answers.",
        "",
        "| Model | Raw RS unique | Semantic RS unique | Semantic range |",
        "| --- | ---: | ---: | --- |",
    ]
    for model, values in result["models"].items():
        lines.append(
            f"| {model} | {values['raw_rs_unique_values']} | "
            f"{values['semantic_rs_unique_values']} | "
            f"{values['semantic_rs_range']} |"
        )
    lines.extend(
        [
            "",
            "GPT-4o-mini's only raw RS disagreement was `95 minutes` versus",
            "`5700 seconds`; these answers are equivalent. Its corrected RS is",
            "therefore 1.0 for every evaluable query. Llama-4 Scout's `384`",
            "versus gold `480` remains a genuine arithmetic error, so its RS",
            "still has two values.",
            "",
            "The formal Section 66 gate remains passed because it is immutable.",
            "The conjunctive semantic interpretation does not pass, and no full",
            "campaign or fresh holdout is authorised from this result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_paraphrase_semantics(result_dir: str | Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    section66_path = result_dir / "paraphrase_health_assessment.json"
    section66 = json.loads(section66_path.read_text(encoding="utf-8"))
    models: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    for model in section66["gate"]["expected_models"]:
        tag = f"S1_{model}_seed42_W5"
        records_path = result_dir / "data" / f"records_{tag}.jsonl"
        per_query_path = result_dir / "data" / f"per_query_{tag}.csv"
        models[model] = audit_model_records(
            _read_jsonl(records_path), pd.read_csv(per_query_path)
        )
        source_hashes[model] = {
            "records_sha256": _sha256(records_path),
            "per_query_sha256": _sha256(per_query_path),
        }

    result = {
        "analysis": "section67_section66_semantic_equivalence_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "formal_section66_gate_pass": bool(section66["overall_gate_pass"]),
        "models": models,
        "source_section66_assessment_sha256": _sha256(section66_path),
        "source_hashes": source_hashes,
        "semantic_nondegeneracy_all_models": all(
            values["semantic_nondegeneracy"] for values in models.values()
        ),
        "fresh_holdout_authorized": False,
        "full_campaign_authorized": False,
    }
    output_dir = result_dir / "section67_semantic_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output_dir / "ASSESSMENT.md", result)
    return result
