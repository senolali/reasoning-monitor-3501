from __future__ import annotations

import numpy as np
import pandas as pd

from reasoning_monitor.evaluation.risk_adjusted_trajectory_ablation import (
    assess_metric_ablation,
)


def test_metric_ablation_qualifies_a_shared_prospective_signal():
    rng = np.random.default_rng(17)
    rows = []
    for family_index in range(60):
        family = f"family-{family_index:02d}"
        event = family_index % 2
        for model in ("gpt4o_mini", "llama4_scout"):
            rows.append(
                {
                    "model": model,
                    "family_id": family,
                    "trajectory_split": "calibration",
                    "task_template": "arithmetic_chain",
                    "trajectory_level": 2,
                    "failure_next": event,
                    "RS_raw": 0.9 if event else 0.1,
                    "LS_raw": float(rng.uniform()),
                    "ES_raw": float(rng.uniform()),
                }
            )
    gate = {
        "expected_models": ["gpt4o_mini", "llama4_scout"],
        "expected_split": "calibration",
        "candidate_metric_subsets": [["RS_raw"]],
        "analysis": {
            "ridge_c_grid": [0.1, 1.0, 10.0],
            "outer_splits": 4,
            "inner_splits": 3,
            "seed": 17,
        },
        "decision": {
            "minimum_combined_mean_improvement": 0.0,
            "minimum_combined_positive_folds": 3,
            "minimum_model_mean_improvement": 0.0,
        },
        "decision_scope": "test only",
    }

    result = assess_metric_ablation(pd.DataFrame(rows), gate)

    assert result["screen_pass"] is True
    assert result["selected_candidate"] == "RS"
    assert result["candidates"]["RS"]["combined"]["positive_folds"] == 4
