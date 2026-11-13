import numpy as np
import pandas as pd

from reasoning_monitor.evaluation import early_warning_forensics as ew


def test_double_smoothing_has_structural_turn_40_floor():
    assert ew.first_possible_legacy_turn(20, 20) == 40
    values = np.r_[np.ones(30), np.zeros(70)]
    alarm = ew.legacy_window_cusum(values, threshold=0.15)
    assert alarm is None or alarm >= 40


def test_fixed_baseline_path_can_detect_turn_31_precursor():
    values = np.r_[np.ones(30), np.zeros(70)]
    assert ew.fixed_baseline_cusum(values, threshold=2.0, direction="down") <= 33
    assert ew.fixed_baseline_cusum(values, threshold=2.0, direction="up") is None


def test_two_sided_rule_detects_upward_construct_drift():
    values = np.r_[np.full(30, 0.3), np.full(70, 0.9)]
    assert ew.fixed_baseline_cusum(values, threshold=2.0, direction="both") == 31


def test_threshold_selection_controls_small_null_panel():
    nulls = [np.full(100, 0.7) for _ in range(12)]
    threshold, far = ew.select_threshold(nulls, direction="both")
    assert threshold == ew.RESIDUAL_THRESHOLDS[0]
    assert far == 0.0


def test_alarm_summary_counts_true_leads_only_after_change():
    metric = pd.DataFrame([
        {"dialogue_id": "S1/a", "scenario": "S1", "alarm": 35,
         "category": "detection"},
        {"dialogue_id": "S2/a", "scenario": "S2", "alarm": 49,
         "category": "false_alarm"},
        {"dialogue_id": "S0-E/a", "scenario": "S0-E", "alarm": None,
         "category": "null_quiet"},
    ])
    cq = pd.DataFrame([
        {"dialogue_id": "S1/a", "alarm": 40, "category": "detection"},
        {"dialogue_id": "S2/a", "alarm": 60, "category": "detection"},
        {"dialogue_id": "S0-E/a", "alarm": None, "category": "null_quiet"},
    ])
    out = ew.summarize_against_cq(metric, cq)
    assert out["leads_cq"] == 1
    assert out["prechange_false_alarm_rate"] == 0.5


def test_auc_binary_handles_ties():
    assert ew.auc_binary([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert ew.auc_binary([0, 1], [1, 1]) == 0.5
