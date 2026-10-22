"""Assessment helpers for the CDR-v1.2a exact-message capacity retry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame
from reasoning_monitor.evaluation.user_only_mechanism import assess_user_only_model


def _replace_censored_records(
    original: Sequence[dict[str, Any]],
    retries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    replacements = {str(record["qid"]): record for record in retries}
    if len(replacements) != len(retries):
        raise ValueError("duplicate retry qid")
    original_qids = {str(record["qid"]) for record in original}
    missing = sorted(set(replacements) - original_qids)
    if missing:
        raise ValueError(f"retry qids absent from original records: {missing}")
    return [replacements.get(str(record["qid"]), record) for record in original]


def assess_capacity_retry_model(
    model: str,
    retries: Sequence[dict[str, Any]],
    original: Sequence[dict[str, Any]],
    existing: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    expected_count = int(gate["expected_retry_counts"][model])
    retry_frame = records_to_turn_frame(retries)
    n_retries = len(retries)
    success_rate = (
        sum(bool(record.get("ok")) for record in retries) / n_retries
        if n_retries
        else 0.0
    )
    marker_rate = (
        float(retry_frame["strict_final_marker"].mean())
        if not retry_frame.empty
        else 0.0
    )
    agreement_rate = (
        float(retry_frame["stored_answer_agreement"].mean())
        if not retry_frame.empty
        else 0.0
    )
    truncations = sum(
        str(record.get("finish_reason") or "").lower() == "length"
        for record in retries
    )
    ceiling_hits = sum(
        int(record.get("n_tokens", 0)) >= int(gate["max_output_tokens"])
        for record in retries
    )
    exact_messages = all(
        record.get("messages_sha256") == record.get("source_messages_sha256")
        for record in retries
    )
    retry_checks = {
        "record_count": n_retries == expected_count,
        "success": success_rate >= float(gate["minimum_success_rate"]),
        "strict_final_marker": marker_rate
        >= float(gate["minimum_strict_final_marker_rate"]),
        "stored_answer_agreement": agreement_rate
        >= float(gate["minimum_stored_answer_agreement_rate"]),
        "no_truncation": truncations <= int(gate["maximum_truncations"]),
        "no_token_ceiling_hit": ceiling_hits
        <= int(gate["maximum_token_ceiling_hits"]),
        "exact_messages": exact_messages,
    }
    retry_checks = {name: bool(value) for name, value in retry_checks.items()}
    adjusted = _replace_censored_records(original, retries)
    adjusted_result = assess_user_only_model(model, adjusted, existing, gate)
    return {
        "model": model,
        "n_retry_records": n_retries,
        "retry_correct": int(retry_frame["CQ_raw"].sum()),
        "success_rate": success_rate,
        "strict_final_marker_rate": marker_rate,
        "stored_answer_agreement_rate": agreement_rate,
        "truncations": truncations,
        "token_ceiling_hits": ceiling_hits,
        "retry_checks": retry_checks,
        "retry_operational_pass": all(retry_checks.values()),
        "capacity_adjusted": adjusted_result,
    }


def assess_capacity_retry(
    retries_by_model: dict[str, Sequence[dict[str, Any]]],
    original_by_model: dict[str, Sequence[dict[str, Any]]],
    existing_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    models = {
        model: assess_capacity_retry_model(
            model,
            retries_by_model.get(model, []),
            original_by_model[model],
            existing_by_model[model],
            gate,
        )
        for model in gate["expected_models"]
    }
    retry_pass = all(result["retry_operational_pass"] for result in models.values())
    qualifying = [
        model
        for model, result in models.items()
        if result["capacity_adjusted"]["mechanism_pass"]
    ]
    return {
        "analysis": "cdr_v1_2a_exact_message_capacity_retry",
        "no_model_calls": True,
        "claim_supporting_analysis": False,
        "original_gate_preserved_as_failed": True,
        "models": models,
        "retry_operational_pass": retry_pass,
        "capacity_adjusted_qualifying_models": qualifying,
        "capacity_adjusted_all_models_qualify": len(qualifying) == len(models),
        "mechanism_resolved": retry_pass,
        "pass": retry_pass,
        "scientific_claim_authorized": False,
        "gate": gate,
    }
