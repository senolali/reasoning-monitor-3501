from pathlib import Path

import numpy as np
import pandas as pd

from reasoning_monitor.dbqr_campaign_contract import (
    METRICS,
    framed_prompts,
    load_budget_contract,
    plot_seven_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_budget_contract_has_exact_frozen_counts():
    config, report = load_budget_contract(ROOT / "configs/dbqr_rq_v1_budget.yaml")
    assert report["calls_per_cell"] == 7
    assert report["practice_staged_review_calls_included_in_total"] == 84
    assert report["stages"]["practice_transport"]["logical_calls"] == 420
    assert report["stages"]["training_calibration"]["logical_calls"] == 8400
    assert report["stages"]["test_confirmation"]["logical_calls"] == 6300
    assert report["training_plus_test_logical_calls"] == 14700
    assert report["all_planned_stages_logical_calls"] == 15120
    assert not report["execution_authorized"]
    assert report["authorized_actions"] == []
    assert not config["stages"]["practice_transport"]["stage_review_authorized"]
    assert not config["stages"]["practice_transport"]["full_execution_authorized"]
    assert not config["stages"]["training_calibration"]["execution_authorized"]
    assert not config["stages"]["test_confirmation"]["execution_authorized"]
    variants = framed_prompts("KEEP THIS EXACT", config)
    assert len(variants) == 2
    assert all("KEEP THIS EXACT" in variant for variant in variants)


def test_plot_contract_always_has_seven_metric_panels(tmp_path):
    rows = []
    for condition in ("self_history", "oracle_answer_history"):
        for turn in range(1, 11):
            row = {"model": "fixture", "condition": condition, "turn": turn}
            row.update({metric: (1.0 if metric == "CQ" else np.nan) for metric in METRICS})
            rows.append(row)
    paths = plot_seven_metrics(pd.DataFrame(rows), tmp_path, title="layout")
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in paths)
