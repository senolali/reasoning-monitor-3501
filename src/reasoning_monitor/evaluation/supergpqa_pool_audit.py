"""Call-free structural audit and deterministic SuperGPQA pool selection."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DIFFICULTIES = ("easy", "middle", "hard")
PROJECT_DIFFICULTY = {"easy": "easy", "middle": "medium", "hard": "hard"}
REQUIRED_FIELDS = (
    "uuid",
    "question",
    "options",
    "answer",
    "answer_letter",
    "discipline",
    "field",
    "subfield",
    "difficulty",
    "is_calculation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _answer_index(letter: Any) -> int | None:
    value = str(letter or "").strip().upper()
    if len(value) != 1 or not "A" <= value <= "Z":
        return None
    return ord(value) - ord("A")


def row_issues(row: dict[str, Any]) -> set[str]:
    """Return machine-checkable structural issues for one source row."""
    issues: set[str] = set()
    if any(field not in row or row[field] in (None, "") for field in REQUIRED_FIELDS):
        issues.add("missing_required")

    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2 or any(not _normalise(x) for x in options):
        issues.add("invalid_options")
        return issues

    if str(row.get("difficulty", "")).lower() not in DIFFICULTIES:
        issues.add("invalid_difficulty")

    answer_index = _answer_index(row.get("answer_letter"))
    if answer_index is None or answer_index >= len(options):
        issues.add("invalid_answer_letter")
    elif _normalise(options[answer_index]) != _normalise(row.get("answer")):
        issues.add("answer_option_mismatch")

    normalised_options = [_normalise(option) for option in options]
    if len(set(normalised_options)) != len(normalised_options):
        issues.add("duplicate_options")
    return issues


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _stable_order(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['uuid']}".encode()
        ).hexdigest(),
    )


def _label_balanced_sample(
    rows: list[dict[str, Any]], n: int, salt: str
) -> list[dict[str, Any]]:
    """Sample deterministically while keeping gold positions as even as possible."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["answer_letter"]).upper()].append(row)
    for label in groups:
        groups[label] = _stable_order(groups[label], f"{salt}:{label}")

    selected: list[dict[str, Any]] = []
    labels = sorted(groups)
    while len(selected) < n:
        progressed = False
        for label in labels:
            if groups[label] and len(selected) < n:
                selected.append(groups[label].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"requested {n} rows but only {len(selected)} are available")
    return _stable_order(selected, f"{salt}:stream")


def _tier_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["difficulty"]).lower() for row in rows)
    return {difficulty: counts[difficulty] for difficulty in DIFFICULTIES}


def _calculation_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    rates = {}
    for difficulty in DIFFICULTIES:
        tier = [row for row in rows if str(row["difficulty"]).lower() == difficulty]
        rates[difficulty] = (
            sum(bool(row.get("is_calculation")) for row in tier) / len(tier)
            if tier
            else 0.0
        )
    return rates


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, result: dict[str, Any]) -> None:
    selected = result["selected_subfield"]
    integrity = result["integrity"]
    clean = result["clean_pool"]
    split = result["selection"]
    lines = [
        "# Section 52 SuperGPQA Structural Pool Audit",
        "",
        "This audit is call-free. It reads the pinned source dataset, validates its",
        "multiple-choice structure, and freezes UUID-only pilot/validation pools.",
        "It does not inspect or generate any target-model response.",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{result['dataset_id']}`",
        f"- Revision: `{result['revision']}`",
        f"- SHA-256: `{result['source_sha256']}`",
        f"- Rows: {integrity['rows']:,}",
        "- Upstream license: ODC-BY; SuperGPQA is a composite dataset and upstream",
        "  source-license attribution remains required for publication/use.",
        "",
        "## Integrity",
        "",
        f"- Unique UUIDs: {integrity['unique_uuids']:,}",
        f"- Unique normalized questions: {integrity['unique_questions']:,}",
        f"- Duplicate-question rows excluded: {integrity['duplicate_question_rows']:,}",
        f"- Rows with any structural issue: {integrity['rows_with_issues']:,}",
        f"- Issue counts: `{json.dumps(integrity['issue_counts'], sort_keys=True)}`",
        "",
        "## Frozen Selection Rule",
        "",
        f"Only clean, unique questions with exactly {clean['required_option_count']} options",
        "are eligible. The selected subfield maximizes its minimum Easy/Middle/Hard",
        "count; ties use total size and then lexical hierarchy. Gold-answer positions",
        "are balanced as far as the available source pool permits.",
        "",
        f"Selected hierarchy: **{selected['discipline']} > {selected['field']} > "
        f"{selected['subfield']}**.",
        "",
        "| Tier | Clean source pool | Calculation rate | Pilot | Validation |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source_tier in DIFFICULTIES:
        project_tier = PROJECT_DIFFICULTY[source_tier]
        lines.append(
            f"| {project_tier.title()} | {selected['tier_counts'][source_tier]} | "
            f"{selected['calculation_rates'][source_tier]:.3f} | "
            f"{split['counts']['pilot'][project_tier]} | "
            f"{split['counts']['validation'][project_tier]} |"
        )
    lines.extend(
        [
            "",
            f"Pilot/validation UUID overlap: **{split['overlap']}**.",
            "",
            "## Decision",
            "",
            f"Structural gate: **{'PASS' if result['structural_gate_pass'] else 'FAIL'}**.",
            "The frozen pool is suitable for an exploratory parser-controlled pilot.",
            "It is not a confirmatory replacement for the primary design.",
            "",
            "## Interpretation Boundaries",
            "",
            "- Multiple choice makes CQ/CS answer extraction auditable, but it does not",
            "  by itself validate LS/SS reasoning-trace measurements.",
            "- SuperGPQA difficulty is partly informed by expert judgment and LLM",
            "  response accuracy; it is not a fully model-independent treatment.",
            "- Calculation prevalence changes sharply across the selected tiers and is",
            "  therefore a measured composition shift, not an isolated causal effect of",
            "  difficulty.",
            "- No paid pilot is authorised by this audit. A run config must remain on",
            "  hold until explicit execution approval.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_supergpqa_pool(
    project_root: str | Path,
    config_path: str | Path = "configs/supergpqa_pool_audit.yaml",
) -> dict[str, Any]:
    """Validate SuperGPQA and freeze disjoint, deterministic UUID pools."""
    project_root = Path(project_root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_path = project_root / config["data_path"]
    output_dir = project_root / config["output_dir"]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = _sha256(data_path)
    if source_sha256 != config["expected_sha256"]:
        raise ValueError(
            f"SuperGPQA SHA-256 mismatch: {source_sha256} != "
            f"{config['expected_sha256']}"
        )
    rows = load_rows(data_path)

    issue_sets = [row_issues(row) for row in rows]
    issue_counts = Counter(issue for issues in issue_sets for issue in issues)
    question_counts = Counter(_normalise(row.get("question")) for row in rows)
    duplicate_questions = {
        question for question, count in question_counts.items() if question and count > 1
    }
    clean_rows = [
        row
        for row, issues in zip(rows, issue_sets, strict=True)
        if not issues and _normalise(row.get("question")) not in duplicate_questions
    ]
    required_option_count = int(config["required_option_count"])
    fixed_option_rows = [
        row for row in clean_rows if len(row["options"]) == required_option_count
    ]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fixed_option_rows:
        hierarchy = (str(row["discipline"]), str(row["field"]), str(row["subfield"]))
        grouped[hierarchy].append(row)

    candidate_rows = []
    for hierarchy, group in grouped.items():
        counts = _tier_counts(group)
        rates = _calculation_rates(group)
        candidate_rows.append(
            {
                "discipline": hierarchy[0],
                "field": hierarchy[1],
                "subfield": hierarchy[2],
                "easy": counts["easy"],
                "middle": counts["middle"],
                "hard": counts["hard"],
                "minimum_tier": min(counts.values()),
                "total": len(group),
                "easy_calculation_rate": rates["easy"],
                "middle_calculation_rate": rates["middle"],
                "hard_calculation_rate": rates["hard"],
            }
        )
    candidate_rows.sort(
        key=lambda item: (
            -item["minimum_tier"],
            -item["total"],
            item["discipline"],
            item["field"],
            item["subfield"],
        )
    )
    if not candidate_rows:
        raise ValueError("no clean SuperGPQA subfield candidates")

    selected_summary = candidate_rows[0]
    hierarchy = (
        selected_summary["discipline"],
        selected_summary["field"],
        selected_summary["subfield"],
    )
    selected_rows = grouped[hierarchy]
    pilot_n = int(config["pilot_queries_per_tier"])
    validation_n = int(config["validation_queries_per_tier"])
    needed = pilot_n + validation_n
    if selected_summary["minimum_tier"] < needed:
        raise ValueError(
            f"selected subfield has only {selected_summary['minimum_tier']} rows in its "
            f"smallest tier; {needed} are required"
        )

    seed = int(config["selection_seed"])
    selected_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        "pilot": {},
        "validation": {},
    }
    selection_rows: list[dict[str, Any]] = []
    for source_tier in DIFFICULTIES:
        project_tier = PROJECT_DIFFICULTY[source_tier]
        tier_rows = [row for row in selected_rows if row["difficulty"] == source_tier]
        pilot = _label_balanced_sample(
            tier_rows, pilot_n, f"section52:{seed}:{source_tier}:pilot"
        )
        pilot_ids = {row["uuid"] for row in pilot}
        remaining = [row for row in tier_rows if row["uuid"] not in pilot_ids]
        validation = _label_balanced_sample(
            remaining,
            validation_n,
            f"section52:{seed}:{source_tier}:validation",
        )
        selected_by_split["pilot"][project_tier] = pilot
        selected_by_split["validation"][project_tier] = validation
        for split_name, split_rows in (("pilot", pilot), ("validation", validation)):
            for order, row in enumerate(split_rows):
                selection_rows.append(
                    {
                        "split": split_name,
                        "difficulty": project_tier,
                        "order": order,
                        "uuid": row["uuid"],
                        "answer_letter": row["answer_letter"],
                        "is_calculation": bool(row["is_calculation"]),
                        "option_count": len(row["options"]),
                    }
                )

    pilot_ids = {
        row["uuid"]
        for tier_rows in selected_by_split["pilot"].values()
        for row in tier_rows
    }
    validation_ids = {
        row["uuid"]
        for tier_rows in selected_by_split["validation"].values()
        for row in tier_rows
    }
    overlap = len(pilot_ids & validation_ids)

    ids_payload = {
        "dataset_id": config["dataset_id"],
        "revision": config["revision"],
        "source_sha256": source_sha256,
        "selection_seed": seed,
        "hierarchy": {
            "discipline": hierarchy[0],
            "field": hierarchy[1],
            "subfield": hierarchy[2],
        },
        "required_option_count": required_option_count,
        "splits": {
            split_name: {
                difficulty: [row["uuid"] for row in tier_rows]
                for difficulty, tier_rows in tiers.items()
            }
            for split_name, tiers in selected_by_split.items()
        },
    }
    ids_path = data_dir / "selected_ids.json"
    ids_path.write_text(
        json.dumps(ids_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _write_csv(
        data_dir / "subfield_candidates.csv",
        candidate_rows,
        [
            "discipline",
            "field",
            "subfield",
            "easy",
            "middle",
            "hard",
            "minimum_tier",
            "total",
            "easy_calculation_rate",
            "middle_calculation_rate",
            "hard_calculation_rate",
        ],
    )
    _write_csv(
        data_dir / "selected_pool_metadata.csv",
        selection_rows,
        [
            "split",
            "difficulty",
            "order",
            "uuid",
            "answer_letter",
            "is_calculation",
            "option_count",
        ],
    )

    split_counts = {
        split_name: {tier: len(values) for tier, values in tiers.items()}
        for split_name, tiers in selected_by_split.items()
    }
    selected_tier_counts = _tier_counts(selected_rows)
    selected_rates = _calculation_rates(selected_rows)
    expected_split_counts = {
        "pilot": {tier: pilot_n for tier in PROJECT_DIFFICULTY.values()},
        "validation": {tier: validation_n for tier in PROJECT_DIFFICULTY.values()},
    }
    structural_gate_pass = (
        source_sha256 == config["expected_sha256"]
        and overlap == 0
        and split_counts == expected_split_counts
        and all(len(row["options"]) == required_option_count for row in selected_rows)
    )
    result = {
        "analysis": "section52_supergpqa_call_free_pool_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "dataset_id": config["dataset_id"],
        "revision": config["revision"],
        "source_sha256": source_sha256,
        "integrity": {
            "rows": len(rows),
            "unique_uuids": len({row.get("uuid") for row in rows}),
            "unique_questions": len(question_counts),
            "duplicate_question_rows": sum(
                count for question, count in question_counts.items()
                if question in duplicate_questions
            ),
            "rows_with_issues": sum(bool(issues) for issues in issue_sets),
            "issue_counts": dict(sorted(issue_counts.items())),
        },
        "clean_pool": {
            "clean_unique_rows": len(clean_rows),
            "required_option_count": required_option_count,
            "fixed_option_rows": len(fixed_option_rows),
            "subfields": len(grouped),
        },
        "selected_subfield": {
            "discipline": hierarchy[0],
            "field": hierarchy[1],
            "subfield": hierarchy[2],
            "tier_counts": selected_tier_counts,
            "calculation_rates": selected_rates,
        },
        "selection": {
            "seed": seed,
            "counts": split_counts,
            "overlap": overlap,
            "ids_sha256": _sha256(ids_path),
        },
        "structural_gate_pass": structural_gate_pass,
    }
    manifest_path = output_dir / "supergpqa_pool_audit_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "SECTION52_SUPERGPQA_POOL_AUDIT.md", result)
    return result
