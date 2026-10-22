"""Call-free diagnostics and assessment for CDR dialogue transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reasoning_monitor.dialogue_experiment import records_to_turn_frame


def audit_native_history_transport(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Detect the preregistered pilot's shared history-loss signature."""
    models = {}
    for model, records in records_by_model.items():
        frame = records_to_turn_frame(records)
        native = frame[frame["condition"].isin(["self_history", "oracle_history"])]
        first = native[native["turn"] == 1]
        later = native[native["turn"] > 1]
        summary = frame[frame["condition"] == "state_summary"]
        sequences = {}
        for condition in ("self_history", "oracle_history"):
            rows = frame[frame["condition"] == condition].sort_values(
                ["dialogue_id", "turn"]
            )
            sequences[condition] = rows["CQ_raw"].tolist()
        history_records = [
            record
            for record in records
            if record.get("condition") in {"self_history", "oracle_history"}
            and int(record.get("turn", 0)) > 1
        ]
        recorded_history_present = bool(history_records) and all(
            len(record.get("messages") or []) >= 4 for record in history_records
        )
        first_accuracy = float(first["CQ_raw"].mean()) if not first.empty else 0.0
        later_accuracy = float(later["CQ_raw"].mean()) if not later.empty else 0.0
        summary_accuracy = (
            float(summary["CQ_raw"].mean()) if not summary.empty else 0.0
        )
        sequence_identity = sequences["self_history"] == sequences["oracle_history"]
        signature = bool(
            recorded_history_present
            and first_accuracy == 1.0
            and later_accuracy == 0.0
            and summary_accuracy == 1.0
            and sequence_identity
        )
        models[model] = {
            "n_records": len(records),
            "recorded_multimessage_history_present": recorded_history_present,
            "native_history_turn_1_accuracy": first_accuracy,
            "native_history_post_turn_1_accuracy": later_accuracy,
            "state_summary_accuracy": summary_accuracy,
            "self_oracle_cq_sequences_identical": sequence_identity,
            "history_transport_failure_signature": signature,
        }
    detected = bool(models) and all(
        result["history_transport_failure_signature"] for result in models.values()
    )
    return {
        "analysis": "cdr_v1_native_history_transport_audit",
        "no_model_calls": True,
        "models": models,
        "transport_failure_signature_detected": detected,
        "scientific_interpretation_authorized": False,
        "decision": (
            "integration_invalid_use_packed_transport_probe"
            if detected
            else "signature_not_established_investigate_manually"
        ),
    }


def assess_packed_transport_probe(
    records_by_model: dict[str, Sequence[dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Assess whether packed self/oracle context reaches both ASU models."""
    models = {}
    for model in gate["expected_models"]:
        records = list(records_by_model.get(model, []))
        n_records = len(records)
        success_rate = (
            sum(bool(record.get("ok")) for record in records) / n_records
            if n_records
            else 0.0
        )
        correct = [bool(record.get("correct")) for record in records]
        accuracy = sum(correct) / n_records if n_records else 0.0
        marker_rate = (
            sum(bool(record.get("strict_final_marker")) for record in records)
            / n_records
            if n_records
            else 0.0
        )
        truncations = sum(
            str(record.get("finish_reason") or "").lower() == "length"
            for record in records
        )
        condition_accuracy = {}
        for condition in gate["expected_conditions"]:
            subset = [record for record in records if record.get("condition") == condition]
            condition_accuracy[condition] = (
                sum(bool(record.get("correct")) for record in subset) / len(subset)
                if subset
                else 0.0
            )
        checks = {
            "record_count": n_records == int(gate["expected_records_per_model"]),
            "success": success_rate >= float(gate["minimum_success_rate"]),
            "accuracy": accuracy >= float(gate["minimum_accuracy"]),
            "condition_accuracy": all(
                value >= float(gate["minimum_condition_accuracy"])
                for value in condition_accuracy.values()
            ),
            "strict_final_marker": marker_rate
            >= float(gate["minimum_strict_final_marker_rate"]),
            "no_truncation": truncations <= int(gate["maximum_truncations"]),
        }
        checks = {name: bool(passed) for name, passed in checks.items()}
        models[model] = {
            "n_records": n_records,
            "success_rate": success_rate,
            "accuracy": accuracy,
            "accuracy_by_condition": condition_accuracy,
            "strict_final_marker_rate": marker_rate,
            "truncations": truncations,
            "checks": checks,
            "pass": all(checks.values()),
        }
    passed = all(result["pass"] for result in models.values())
    return {
        "analysis": "cdr_v1_1_packed_history_transport_probe",
        "models": models,
        "pass": passed,
        "pass_authorizes": gate["pass_authorizes"],
        "gate": gate,
    }

