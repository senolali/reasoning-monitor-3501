"""Call-free diagnostic audit of the failed Section 64 metric-health pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MODELS = ("gpt4o_mini", "llama4_scout")
EXPECTED_HASHES = {
    "gpt4o_mini": {
        "records": "66e1318787c087b6a686ae0c8a7e487f2ee5162b40c100a8d08e5b1b22d807c6",
        "per_query": "90f55a6689258a20ca943c234de32052ec99773cb384b57a2d133bf2b1767185",
    },
    "llama4_scout": {
        "records": "a6f6aefa02a2c0810eeb0a75b36bfae4a5dffe9194e434012560709e7200b29e",
        "per_query": "c548ebf2c41f49e89ca05e979b280b080e7b29cd865d1ca9d7e44be3ec24ed40",
    },
}
EXPECTED_MANIFEST_SHA256 = (
    "0ffcdf90209023ed88f81b409e011161f812323fb1a798aa05bb421f41a683d5"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_model(
    records: list[dict[str, Any]],
    per_query: pd.DataFrame,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["qid"])].append(record)

    raw_diverse = 0
    answer_diverse = 0
    raw_identical_by_dataset: Counter[str] = Counter()
    wrapper_exact = 0
    wrapper_total = 0
    diversity_rows = []
    for qid, rows in grouped.items():
        runs = sorted(
            (record for record in rows if record.get("role") == "run"),
            key=lambda record: int(record["call_index"]),
        )
        paraphrases = sorted(
            (record for record in rows if record.get("role") == "paraphrase"),
            key=lambda record: int(record["call_index"]),
        )
        primary_prompt = str(runs[0]["prompt"]).strip()
        expected_prompts = (
            f"Consider the following problem. {primary_prompt}",
            f"{primary_prompt} Please work through this carefully.",
            f"Here is a question that needs solving: {primary_prompt}",
        )
        wrapper_total += len(paraphrases)
        wrapper_exact += sum(
            str(record["prompt"]) == expected
            for record, expected in zip(paraphrases, expected_prompts, strict=True)
        )

        unique_raw = len({str(record.get("raw_text") or "") for record in runs})
        unique_answers = len({str(record.get("answer") or "") for record in runs})
        raw_diverse += unique_raw >= 2
        answer_diverse += unique_answers >= 2
        dataset = str(runs[0]["dataset"])
        if unique_raw == 1:
            raw_identical_by_dataset[dataset] += 1
        diversity_rows.append(
            {
                "qid": qid,
                "dataset": dataset,
                "unique_raw_k": unique_raw,
                "unique_answer_k": unique_answers,
            }
        )

    rs_values = [float(value) for value in per_query["RS_raw"].dropna().tolist()]
    cq_correct = int((per_query["CQ_raw"].astype(float) == 1.0).sum())
    return {
        "n_queries": len(grouped),
        "paraphrase_prompts": wrapper_total,
        "exact_wrapper_prompts": wrapper_exact,
        "all_paraphrases_retain_original_prompt_verbatim": (
            wrapper_exact == wrapper_total
        ),
        "cq_correct_primary": cq_correct,
        "rs_evaluable": len(rs_values),
        "rs_evaluable_equals_cq_correct_primary": len(rs_values) == cq_correct,
        "rs_unique_values": len({round(value, 12) for value in rs_values}),
        "rs_range": [min(rs_values), max(rs_values)] if rs_values else None,
        "rs_all_one": bool(rs_values) and all(value == 1.0 for value in rs_values),
        "k_raw_diverse_queries": raw_diverse,
        "k_answer_diverse_queries": answer_diverse,
        "k_raw_identical_by_dataset": dict(raw_identical_by_dataset),
        "diversity_details": diversity_rows,
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Section 65 Section 64 Failure Diagnostic",
        "",
        "This call-free audit preserves the immutable Section 64 records.",
        "It separates RS perturbation design from K-repeat sampling diversity.",
        "",
        "| Model | RS evaluable | RS unique | RS range | Raw-diverse K | Answer-diverse K |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, values in result["models"].items():
        rs_range = values["rs_range"] or [None, None]
        lines.append(
            f"| {model} | {values['rs_evaluable']}/15 | "
            f"{values['rs_unique_values']} | {rs_range} | "
            f"{values['k_raw_diverse_queries']}/15 | "
            f"{values['k_answer_diverse_queries']}/15 |"
        )

    lines.extend(
        [
            "",
            "All 90 paraphrase prompts are deterministic wrappers that retain",
            "the complete original prompt verbatim. RS is therefore testing",
            "prefix/suffix sensitivity rather than genuine rephrasing. It is",
            "defined exactly on the C-subset and equals 1.0 for every evaluable",
            "query in both models; this is not a parsing failure.",
            "",
            "Llama-4 has six raw-identical K=5 groups: four GSM8K, one",
            "ARC-Challenge and one MATH. Only three of its 15 questions have",
            "more than one distinct extracted K answer.",
            "",
            "Section 64 remains failed and does not authorise the 90-question",
            "campaign. Repeating the same protocol would not repair either",
            "failure. A future development protocol must first freeze genuine",
            "meaning-preserving paraphrases and separately address Llama-4",
            "sampling diversity.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_metric_health_failure(project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root)
    source_dir = project_root / "results" / "proposal_s1_metric_health_gpt_llama4"
    manifest_path = source_dir / "manifest.json"
    models = {}
    source_hashes = {}
    hashes_match = _sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
    for model in MODELS:
        tag = f"S1_{model}_seed42_W5"
        records_path = source_dir / "data" / f"records_{tag}.jsonl"
        per_query_path = source_dir / "data" / f"per_query_{tag}.csv"
        records_hash = _sha256(records_path)
        per_query_hash = _sha256(per_query_path)
        expected = EXPECTED_HASHES[model]
        source_hashes[model] = {
            "records_sha256": records_hash,
            "per_query_sha256": per_query_hash,
            "matches": (
                records_hash == expected["records"]
                and per_query_hash == expected["per_query"]
            ),
        }
        hashes_match = hashes_match and source_hashes[model]["matches"]
        models[model] = audit_model(
            _load_records(records_path), pd.read_csv(per_query_path)
        )

    result = {
        "analysis": "section65_section64_failure_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_hashes": source_hashes,
        "all_source_hashes_match": hashes_match,
        "models": models,
        "section64_decision_unchanged": True,
        "full_campaign_authorized": False,
    }
    output_dir = source_dir / "section65_failure_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output_dir / "ASSESSMENT.md", result)
    return result

