"""Call-free, hash-verified synthesis of the completed experimental chain."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from reasoning_monitor.experiment import ExperimentConfig
from reasoning_monitor.metrics.semantic_answers import load_semantic_policy

REQUIRED_SOURCE_IDS = {
    "section47_gpt_gate",
    "section47_llama3_gate",
    "section47_mistral_gate",
    "section49_manifest",
    "section49_gate",
    "section51_manifest",
    "section51_gate",
    "section55_parsing_audit",
    "section56_answer_first",
    "section57_answer_last",
    "section58_validation",
    "section59_format_stress",
    "section60_llama4_stress",
    "section61_llama4_validation",
    "section62_mistral_validation",
    "section63_mistral_format",
    "section64_metric_health",
    "section65_failure_audit",
    "section66_paraphrase_health",
    "section67_semantic_audit",
    "section66_llama4_seed_probe",
    "section68_policy_gate",
    "section68_policy",
    "section68_cases",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"unsupported evidence source: {path}")


def verify_sources(
    root: str | Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Verify every frozen source and return parsed documents plus diagnostics."""
    root = Path(root).resolve()
    entries = manifest.get("sources") or []
    ids = [str(entry.get("id")) for entry in entries]
    problems: list[str] = []
    if len(ids) != len(set(ids)):
        problems.append("source IDs are not unique")
    missing = sorted(REQUIRED_SOURCE_IDS - set(ids))
    extra = sorted(set(ids) - REQUIRED_SOURCE_IDS)
    if missing:
        problems.append(f"missing required sources: {missing}")
    if extra:
        problems.append(f"unexpected sources: {extra}")

    documents: dict[str, Any] = {}
    verification: list[dict[str, Any]] = []
    for entry in entries:
        source_id = str(entry["id"])
        relative = Path(str(entry["path"]))
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            problems.append(f"source escapes project root: {relative}")
            continue
        expected = str(entry["sha256"]).lower()
        exists = path.is_file()
        observed = _sha256(path) if exists else None
        matches = observed == expected
        verification.append(
            {
                "id": source_id,
                "path": relative.as_posix(),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": matches,
            }
        )
        if not exists:
            problems.append(f"source does not exist: {relative}")
            continue
        if not matches:
            problems.append(f"source hash mismatch: {relative}")
            continue
        documents[source_id] = _load_source(path)
    return documents, verification, problems


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _float(value: Any) -> float:
    return float(value)


def _first(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 1:
        raise ValueError(f"expected exactly one CSV row; found {len(rows)}")
    return rows[0]


def _semantic_policy_check(root: Path, documents: dict[str, Any]) -> dict[str, Any]:
    gate = documents["section68_policy_gate"]
    cases = documents["section68_cases"]["cases"]
    policy = load_semantic_policy(str(root / gate["policy_path"]))
    outcomes = [
        policy.equivalent(
            str(case["left"]), str(case["right"]), str(case["dataset"])
        )
        is bool(case["expected"])
        for case in cases
    ]
    default_policy = ExperimentConfig().semantic_answer_policy
    return {
        "name": policy.name,
        "case_count": len(cases),
        "case_passes": sum(outcomes),
        "historical_default_policy": default_policy,
        "pass": bool(
            policy.name == gate["expected_policy_name"]
            and len(cases) == int(gate["expected_case_count"])
            and all(outcomes)
            and default_policy is None
        ),
    }


def synthesize_documents(
    documents: dict[str, Any],
    semantic_policy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Derive the frozen scientific decisions from verified evidence documents."""
    problems: list[str] = []

    section47 = {
        "gpt4o_mini": _first(documents["section47_gpt_gate"]),
        "llama3_8b": _first(documents["section47_llama3_gate"]),
        "mistral_7b": _first(documents["section47_mistral_gate"]),
    }
    section49 = documents["section49_manifest"]
    section49_rows = documents["section49_gate"]
    section51 = documents["section51_manifest"]
    section51_rows = documents["section51_gate"]
    section55 = documents["section55_parsing_audit"]
    section56 = documents["section56_answer_first"]
    section57 = documents["section57_answer_last"]
    section58 = documents["section58_validation"]
    section59 = documents["section59_format_stress"]
    section60 = documents["section60_llama4_stress"]
    section61 = documents["section61_llama4_validation"]
    section62 = documents["section62_mistral_validation"]
    section63 = documents["section63_mistral_format"]
    section64 = documents["section64_metric_health"]
    section65 = documents["section65_failure_audit"]
    section66 = documents["section66_paraphrase_health"]
    section67 = documents["section67_semantic_audit"]
    seed_probe = documents["section66_llama4_seed_probe"]

    checks = {
        "section47_cq_passes_all_models": all(
            _bool(row["cq_holdout_gate_pass"]) for row in section47.values()
        ),
        "section47_only_gpt_overall_passes": (
            _bool(section47["gpt4o_mini"]["overall_gate_pass"])
            and not _bool(section47["llama3_8b"]["overall_gate_pass"])
            and not _bool(section47["mistral_7b"]["overall_gate_pass"])
        ),
        "section49_early_warning_fails": (
            section49["overall_early_warning_gate_pass"] is False
            and not section49["passing_metrics"]
            and all(not _bool(row["early_warning_gate_pass"]) for row in section49_rows)
        ),
        "section51_corrected_early_warning_fails": (
            section51["corrected_early_warning_gate_pass"] is False
            and not section51["corrected_passing_metrics"]
            and all(
                not _bool(row["early_warning_gate_pass"])
                for row in section51_rows
                if row["source"] == "corrected"
            )
        ),
        "section55_confounded": len(section55["primary_cq"]) == 6,
        "section56_fails": section56["overall_gate_pass"] is False,
        "section57_passes": section57["overall_gate_pass"] is True,
        "section58_scientific_gradient_operational_failure": (
            section58["overall_gate_pass"] is False
            and all(
                model["checks"]["gsm8k_math_gap"]
                and model["checks"]["ordered_trend_significance"]
                for model in section58["models"].values()
            )
        ),
        "section59_llama3_format_failure": (
            section59["models"]["gpt4o_mini"]["overall_gate_pass"] is True
            and section59["models"]["llama3_8b"]["overall_gate_pass"] is False
        ),
        "section60_llama4_format_pass": section60["overall_gate_pass"] is True,
        "section61_llama4_clean_pass": section61["overall_gate_pass"] is True,
        "section62_mistral_operational_failure": section62["overall_gate_pass"] is False,
        "section63_mistral_nonmonotonic": (
            section63["relaxed_observable"] == 90
            and section63["relaxed_monotonic_nonincreasing"] is False
        ),
        "section64_metric_health_fails": section64["overall_gate_pass"] is False,
        "section65_failure_diagnosed": (
            section65["all_source_hashes_match"] is True
            and section65["full_campaign_authorized"] is False
        ),
        "section66_formal_rs_pass": section66["overall_gate_pass"] is True,
        "section67_semantic_interpretation_fails": (
            section67["semantic_nondegeneracy_all_models"] is False
            and section67["full_campaign_authorized"] is False
        ),
        "llama4_seed_probe_fails": (
            seed_probe["capability_pass"] is False
            and seed_probe["accepted_calls"] == 5
            and seed_probe["unique_raw_responses"] == 1
        ),
        "section68_policy_passes": semantic_policy["pass"] is True,
    }
    problems.extend(name for name, passed in checks.items() if not passed)

    difficulty_profiles = {
        model: {
            "easy": _float(row["easy_cq"]),
            "medium": _float(row["medium_cq"]),
            "hard": _float(row["hard_cq"]),
            "easy_hard_gap": _float(row["easy_hard_gap"]),
            "one_sided_p": _float(row["permutation_p_one_sided"]),
            "cq_gate_pass": _bool(row["cq_holdout_gate_pass"]),
            "overall_gate_pass": _bool(row["overall_gate_pass"]),
            "k_repeat_diversity_rate": _float(row["k_repeat_diversity_rate"]),
        }
        for model, row in section47.items()
    }
    proposal_profiles = {
        "gpt4o_mini": {
            **section58["models"]["gpt4o_mini"]["accuracy"],
            "gap": section58["models"]["gpt4o_mini"]["gsm8k_math_gap"],
            "one_sided_p": section58["models"]["gpt4o_mini"]["ordered_trend"][
                "one_sided_p"
            ],
            "operationally_valid": section58["models"]["gpt4o_mini"][
                "inference_valid"
            ],
        },
        "llama3_8b": {
            **section58["models"]["llama3_8b"]["accuracy"],
            "gap": section58["models"]["llama3_8b"]["gsm8k_math_gap"],
            "one_sided_p": section58["models"]["llama3_8b"]["ordered_trend"][
                "one_sided_p"
            ],
            "operationally_valid": section58["models"]["llama3_8b"][
                "inference_valid"
            ],
        },
        "llama4_scout": {
            **section61["models"]["llama4_scout"]["accuracy"],
            "gap": section61["models"]["llama4_scout"]["gsm8k_math_gap"],
            "one_sided_p": section61["models"]["llama4_scout"]["ordered_trend"][
                "one_sided_p"
            ],
            "operationally_valid": section61["models"]["llama4_scout"][
                "inference_valid"
            ],
        },
    }

    hypotheses = {
        "h1_non_cq_metrics_precede_cq": {
            "status": "not_supported",
            "scope": "the only Section 47 track eligible for frozen detector timing",
            "evidence": {
                "cq_detected_both_transitions": all(
                    _bool(row["cq_detected_both_transitions"])
                    for row in section49_rows
                ),
                "passing_non_cq_metrics": section49["passing_metrics"],
                "section49_gate_pass": section49["overall_early_warning_gate_pass"],
                "section51_corrected_gate_pass": section51[
                    "corrected_early_warning_gate_pass"
                ],
            },
            "interpretation": (
                "CQ detected both transitions, but no non-CQ metric led CQ at both; "
                "the structure-aware LS/SS reparse did not rescue the result."
            ),
        },
        "cq_tracks_difficulty_change": {
            "status": "supported",
            "scope": "frozen E2H holdouts plus the accepted-proposal S1 stream",
            "evidence": {
                "section47_profiles": difficulty_profiles,
                "proposal_s1_profiles": proposal_profiles,
            },
            "interpretation": (
                "CQ showed preregistered difficulty separation in all three Section "
                "47 tracks and a clean gradient in the full Llama-4 S1 validation."
            ),
        },
        "framework_detects_measurement_failures": {
            "status": "supported",
            "scope": "engineering and methodological validity, not early warning",
            "evidence": {
                "answer_first_gsm8k": {
                    model: section56["models"][model]["accuracy"]["gsm8k"]
                    for model in ("gpt4o_mini", "llama3_8b")
                },
                "bounded_answer_last_gsm8k": {
                    model: section57["models"][model]["accuracy"]["gsm8k"]
                    for model in ("gpt4o_mini", "llama3_8b")
                },
                "mistral_strict_to_relaxed_observability": [
                    section63["strict_observable"],
                    section63["relaxed_observable"],
                ],
                "weak_paraphrase_prompts_detected": all(
                    model["all_paraphrases_retain_original_prompt_verbatim"]
                    for model in section65["models"].values()
                ),
                "semantic_unit_artifact_detected": len(
                    section67["models"]["gpt4o_mini"][
                        "unit_equivalence_corrections"
                    ]
                ),
            },
            "interpretation": (
                "The audit chain isolated prompt-order interference, truncation, "
                "format parsing, weak perturbations, unit equivalence and endpoint "
                "determinism before they could be mistaken for model behavior."
            ),
        },
        "cs_ss_cross_model_generalizability": {
            "status": "not_evaluable",
            "scope": "current ASU endpoints and frozen K-repeat gates",
            "evidence": {
                "section47_k_repeat_diversity": {
                    model: profile["k_repeat_diversity_rate"]
                    for model, profile in difficulty_profiles.items()
                },
                "section64_llama4_k_repeat_diversity": section64["models"][
                    "llama4_scout"
                ]["k_repeat_diversity_rate"],
                "llama4_seed_probe_unique_raw": seed_probe["unique_raw_responses"],
            },
            "interpretation": (
                "Llama-3, Llama-4 and Mistral did not provide sufficient repeat "
                "variation for a valid cross-model CS/SS early-warning test."
            ),
        },
        "rs_paraphrase_robustness": {
            "status": "mixed_inconclusive",
            "scope": "15-question development sample with genuine paraphrases",
            "evidence": {
                "formal_section66_gate_pass": section66["overall_gate_pass"],
                "gpt_semantic_unique_values": section67["models"]["gpt4o_mini"][
                    "semantic_rs_unique_values"
                ],
                "llama4_semantic_unique_values": section67["models"][
                    "llama4_scout"
                ]["semantic_rs_unique_values"],
                "all_models_semantically_nondegenerate": section67[
                    "semantic_nondegeneracy_all_models"
                ],
            },
            "interpretation": (
                "GPT's only raw disagreement was unit-equivalent; Llama-4 had one "
                "genuine arithmetic failure. The sample does not establish a stable "
                "cross-model RS signal."
            ),
        },
    }

    return (
        {
            "checks": checks,
            "hypotheses": hypotheses,
            "semantic_answer_policy": semantic_policy,
            "campaign_decision": {
                "full_k5_l3_campaign_authorized": False,
                "additional_paid_pilot_authorized": False,
                "reason": (
                    "The primary early-warning claim is unsupported and the remaining "
                    "non-CQ metric evidence is either endpoint-limited or inconclusive."
                ),
            },
        },
        problems,
    )


def _write_hypothesis_csv(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["hypothesis", "status", "scope", "interpretation"],
        )
        writer.writeheader()
        for name, values in result["hypotheses"].items():
            writer.writerow(
                {
                    "hypothesis": name,
                    "status": values["status"],
                    "scope": values["scope"],
                    "interpretation": values["interpretation"],
                }
            )


def _write_report(path: Path, result: dict[str, Any]) -> None:
    hypotheses = result["hypotheses"]
    lines = [
        "# Section 69 Final Evidence Synthesis",
        "",
        "This is the terminal, call-free synthesis of the completed evidence chain.",
        "It verifies frozen source hashes and does not change any historical gate.",
        "",
        "## Final decisions",
        "",
        "| Claim | Decision | Scope |",
        "| --- | --- | --- |",
    ]
    for name, values in hypotheses.items():
        lines.append(f"| `{name}` | **{values['status']}** | {values['scope']} |")
    lines.extend(
        [
            "",
            "The project's central early-warning claim is **not supported** under the",
            "only track that passed the frozen metric-health gate: CQ detected both",
            "transitions, while no non-CQ metric led CQ in both intervals. Correcting",
            "the reasoning-trace parser did not change that decision.",
            "",
            "CQ difficulty sensitivity is supported. All three Section 47 tracks",
            "passed their preregistered CQ holdout gates, and the clean Section 61",
            "Llama-4 validation reproduced a 0.967/0.900/0.567 S1 profile with a",
            "0.400 GSM8K-MATH gap and one-sided p=0.0000378.",
            "",
            "## What the framework established",
            "",
            "The framework reliably exposed measurement threats that would otherwise",
            "look like model behavior: answer-first prompt interference, token-ceiling",
            "censoring, strict-format parser failures, wrapper-only paraphrases, unit-",
            "equivalent answers and deterministic provider sampling. This is a positive",
            "engineering and methodological result, but it is not evidence that the",
            "non-CQ metrics provide earlier warning than CQ.",
            "",
            "CS/SS cross-model generalizability remains not evaluable on the current",
            "ASU endpoints. Section 47 K-repeat diversity was 1.000 for GPT, 0.667 for",
            "Llama-3 and 0.000 for Mistral; Section 64 Llama-4 diversity was 0.600, and",
            "the five-seed probe still produced one unique raw completion.",
            "",
            "RS evidence is mixed and inconclusive. The genuine-paraphrase arm passed",
            "its formal gate, but semantic auditing removed GPT's only disagreement as",
            "a unit-conversion artifact. Llama-4 retained one genuine arithmetic error.",
            "",
            "## Reporting position",
            "",
            "The project report should retain both the negative central finding and the",
            "positive diagnostic finding. It should not describe failed operational",
            "gates as successful confirmatory tests, and it should distinguish",
            "unsupported from universally disproven.",
            "",
            "## Stop rule",
            "",
            "Section 69 closes this experimental chain. No additional paid pilot, full",
            "K=5/L=3 campaign or Section 70 is authorised. Reopening requires a newly",
            "demonstrated software defect that invalidates immutable evidence used here;",
            "scientific disappointment or a desire for another dataset is not enough.",
            "A future confirmatory study is a new protocol and must begin with the",
            "prospectively frozen `semantic-v1` policy, genuine paraphrases and a backend",
            "that demonstrates usable stochastic repeat diversity.",
            "",
            "## Integrity",
            "",
            f"- Frozen sources verified: {len(result['source_verification'])}",
            f"- All source hashes match: {result['all_source_hashes_match']}",
            f"- Semantic policy cases: {result['semantic_answer_policy']['case_passes']}/"
            f"{result['semantic_answer_policy']['case_count']}",
            f"- Synthesis valid: {result['synthesis_valid']}",
            "- Model calls made by this synthesis: 0",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def synthesize_final_evidence(
    root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify frozen evidence, derive decisions and write the terminal reports."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    documents, verification, source_problems = verify_sources(root, manifest)

    synthesis: dict[str, Any] = {}
    synthesis_problems: list[str] = []
    if not source_problems:
        semantic_policy = _semantic_policy_check(root, documents)
        synthesis, synthesis_problems = synthesize_documents(documents, semantic_policy)

    stop_rule = manifest.get("stop_rule") or {}
    stop_rule_valid = bool(
        int(stop_rule.get("final_section", 0)) == 69
        and stop_rule.get("additional_paid_pilot_authorized") is False
        and stop_rule.get("section70_authorized") is False
    )
    if not stop_rule_valid:
        synthesis_problems.append("invalid Section 69 stop rule")
    problems = [*source_problems, *synthesis_problems]
    result = {
        "analysis": "section69_final_evidence_synthesis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "no_model_calls": True,
        "source_manifest": manifest_path.relative_to(root).as_posix(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_verification": verification,
        "all_source_hashes_match": bool(verification)
        and all(item["matches"] for item in verification),
        **synthesis,
        "stop_rule": stop_rule,
        "problems": problems,
        "synthesis_valid": not problems,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assessment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if result.get("hypotheses"):
        _write_hypothesis_csv(output_dir / "hypothesis_decisions.csv", result)
        _write_report(output_dir / "FINAL_EVIDENCE_SYNTHESIS.md", result)
    return result
