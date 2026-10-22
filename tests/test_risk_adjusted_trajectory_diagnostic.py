from __future__ import annotations

import json

from reasoning_monitor.evaluation.risk_adjusted_trajectory_diagnostic import (
    _validation_summary,
)


def test_validation_summary_counts_positive_folds_and_is_json_serializable():
    result = {
        "folds": [
            {"fold": 1, "improvement": 0.2},
            {"fold": 2, "improvement": -0.1},
            {"fold": 3, "improvement": 0.0},
            {"fold": 4, "improvement": 0.05},
        ],
        "mean_improvement": 0.0375,
    }

    summary = _validation_summary(result)

    assert summary["positive_folds"] == 2
    assert summary["n_folds"] == 4
    assert json.loads(json.dumps(summary))["positive_folds"] == 2
