"""Audit CDR-v1.4 formatting and explore next-turn warning without API calls."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.dialogue_experiment import records_to_turn_frame  # noqa: E402
from reasoning_monitor.evaluation.dialogue_warning import (  # noqa: E402
    FEATURE_SETS,
    analyse_warning,
)
from reasoning_monitor.evaluation.primary_dialogue_validation import (  # noqa: E402
    assess_primary_validation,
)

EXPECTED = {
    "gpt4o_mini": "fb21699a1106b3414f2df1f62741eb1fade59751c6e69f98d8656974d3bc6529",
    "llama4_scout": "2a00f004690c63881cb9b670191514fcf54214216388ec35d18c91c1e9eb68c2",
}


def main():
    source = ROOT / "results/controlled_dialogue_v1_4_replication/data"
    assessment_path = source / "replication_assessment.json"
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    records = {}
    for model, digest in EXPECTED.items():
        path = source / f"records_CDR14_{model}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"changed source: {model}")
        records[model] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    reproduced = assess_primary_validation(records, assessment["gate"])
    if reproduced["models"] != assessment["models"] or assessment["pass"] is not False:
        raise ValueError("original failed assessment did not reproduce")
    output = ROOT / "results/controlled_dialogue_v1_4_warning_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis": "cdr14_exploratory_next_turn_warning",
        "no_model_calls": True,
        "post_hoc_calibration_only": True,
        "original_gate_pass": False,
        "claim_supporting_analysis": False,
        "source_sha256": EXPECTED,
        "assessment_sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
        "feature_sets": FEATURE_SETS,
        "target": "error at t+1, predictions issued after response t",
        "fixed_alarm_threshold": 0.5,
        "limitations": [
            "Repeated numeric instances of three templates are not independent domains.",
            "Eight-line prompt and temporal update schedule can influence late errors.",
            "No causal separation of context length, turn position and update type.",
            "Tokens and latency are proxies, not validated reasoning-quality metrics.",
            "Gold is used for outcomes and retrospective first-failure scoring only.",
            "No tuning, significance claims, or access to development/holdout data.",
        ],
        "models": {},
    }
    for model, items in records.items():
        frame = records_to_turn_frame(items)
        bad = frame[~frame.strict_final_marker]
        scores, predictions = analyse_warning(items)
        predictions.to_csv(output / f"predictions_{model}.csv", index=False)
        report["models"][model] = {
            "format_audit": {
                "noncanonical": bad[["qid", "turn", "condition", "CQ_raw"]].to_dict("records"),
                "all_noncanonical_correct_at_baseline": bool(
                    ((bad.turn == 1) & (bad.CQ_raw == 1)).all()
                ),
                "post_baseline_strict_marker_rate": float(
                    frame.loc[frame.turn > 1, "strict_final_marker"].mean()
                ),
            },
            "warning": scores,
        }
        print(f"Completed call-free analysis: {model}", flush=True)
    (output / "warning_assessment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
