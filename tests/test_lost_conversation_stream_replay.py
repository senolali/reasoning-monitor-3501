import math
import random

import numpy as np
import pandas as pd
import pytest

from reasoning_monitor.detectors.detectors import build_detector
from reasoning_monitor.evaluation import stream_replay as sr


def _metrics_frame():
    rows = []
    # task A: two turns; task B: one turn. Final CQ of B/dialogue is invalid (NaN).
    spec = {
        ("A", "fresh_prefix"): [(1, 2, np.nan, np.nan, np.nan, 0.6, 0.9),
                                (2, 2, 1.0, 1.0, 1.0, 0.8, 0.7)],
        ("A", "dialogue_history"): [(1, 2, np.nan, np.nan, np.nan, np.nan, 0.5),
                                    (2, 2, 0.0, 0.6, np.nan, 0.4, 0.5)],
        ("B", "fresh_prefix"): [(1, 1, 1.0, np.nan, 1.0, 0.9, 0.8)],
        ("B", "dialogue_history"): [(1, 1, np.nan, np.nan, np.nan, np.nan, np.nan)],
    }
    for (task, condition), turns in spec.items():
        for turn, total, cq, cs, rs, ls, ss in turns:
            rows.append({"model": "m", "task_id": task, "condition": condition,
                         "turn": turn, "total_turns": total, "CQ": cq, "CS": cs,
                         "RS": rs, "LS": ls, "SS": ss})
    return pd.DataFrame(rows)


def test_session_values_follow_frozen_rules():
    table = sr.session_values(_metrics_frame())["m"]
    assert table.tasks == ("A", "B")
    # Order: A fresh, B fresh, A dialogue, B dialogue.
    assert list(table.values["CQ"]) == [1.0, 1.0, 0.0, 0.0]  # invalid final -> 0
    assert math.isnan(table.values["RS"][2]) and math.isnan(table.values["RS"][3])
    assert table.values["LS"][0] == pytest.approx(0.7)  # session mean
    assert table.values["LS"][2] == pytest.approx(0.4)  # NaN turn skipped
    assert math.isnan(table.values["LS"][3])            # nothing measured
    final = sr.session_values(_metrics_frame(), "final_turn")["m"]
    assert final.values["LS"][0] == pytest.approx(0.8)
    assert final.values["SS"][0] == pytest.approx(0.7)


def test_sudden_and_gradual_streams_use_each_session_once():
    sudden = sr.ordered_streams("sudden", 48, 5, 1)
    gradual = sr.ordered_streams("gradual", 48, 5, 2)
    for stream in sudden:
        assert sorted(stream) == list(range(96))
        assert all(i < 48 for i in stream[:48]) and all(i >= 48 for i in stream[48:])
    for stream in gradual:
        assert sorted(stream) == list(range(96))
        counts = [sum(i >= 48 for i in stream[b * 16:(b + 1) * 16]) for b in range(6)]
        assert counts == list(sr.GRADUAL_DIALOGUE_COUNTS)
    assert sr.ordered_streams("sudden", 48, 5, 1) == sudden


def test_bootstrap_streams_keep_task_pairs():
    for stream in sr.ordered_streams("sudden", 48, 3, 9, bootstrap=True):
        fresh = sorted(stream[:48])
        dialogue = sorted(i - 48 for i in stream[48:])
        assert fresh == dialogue


def test_null_streams_are_disjoint_from_calibration():
    calibration = sr.null_streams(10, 50, 1)
    validation = sr.null_streams(10, 50, 2, exclude=calibration)
    assert len(set(calibration)) == 50 and not set(calibration) & set(validation)
    assert tuple(range(10)) not in calibration


def test_select_parameter_takes_first_value_passing_every_model():
    counts = {"a": {1.0: 30, 2.0: 1, 3.0: 0}, "b": {1.0: 0, 2.0: 9, 3.0: 0}}
    assert sr.select_parameter(counts, 2000, (1.0, 2.0, 3.0), 0.0033) == 3.0
    assert sr.select_parameter(counts, 2000, (1.0, 2.0), 0.0033) is None


@pytest.mark.parametrize("detector,value", [("adwin", 0.05), ("adwin", 0.002),
                                            ("cusum", 0.5), ("page_hinkley", 0.25),
                                            ("ddm", 2.5)])
def test_alarm_times_match_reference_class_with_resets(detector, value):
    rng = random.Random(7)
    for _ in range(25):
        values = [min(1.0, max(0.0, rng.gauss(0.8, 0.1))) for _ in range(50)]
        values += [min(1.0, max(0.0, rng.gauss(0.45, 0.15))) for _ in range(60)]
        signal = list(enumerate(values, start=1))
        reference = build_detector(detector, **{sr.GRIDS[detector].parameter: value})
        expected = []
        for t, v in signal:
            alarm = reference.update(v, t, "x")
            if alarm is not None:
                expected.append(alarm.t)
        assert sr.alarm_times(signal, detector, value) == expected
        if expected:
            stop = expected[0]
            assert sr.alarm_times(signal, detector, value, stop_at=stop) == expected[:1]


def test_window_signal_skips_missing_values():
    values = np.array([1.0, np.nan, 0.0, 1.0, 0.0])
    signal = sr.window_signal(values, [0, 1, 2, 3, 4], 2)
    assert signal == [(3, 0.5), (4, 0.5), (5, 0.5)]


def test_categories_and_first_post_change_alarm():
    gradual = sr.SCENARIOS["gradual"]
    assert sr.categorize(None, gradual) == "no_alarm"
    assert sr.categorize(10, gradual) == "false_alarm"
    assert sr.categorize(20, gradual) == "early_detection"
    assert sr.categorize(49, gradual) == "post_threshold"
    assert sr.first_at_or_after([5, 30, 60], 17) == 30


def test_lead_helpers():
    times = np.array([[1.0, 2.0, np.inf], [3.0, 1.0, 2.0], [1.0, 1.0, 5.0]])
    shares = sr.first_to_alarm_share(times)
    assert list(shares) == pytest.approx([1 / 3, 1 / 3, 0.0])
    assert sr.prediction_holds({"CS": 0.4, "LS": 0.3, "CQ": 0.1}, ("CS", "LS"))
    assert not sr.prediction_holds({"CS": 0.4, "LS": 0.1, "CQ": 0.1}, ("CS", "LS"))
    assert not sr.prediction_holds({"CS": 0.4, "CQ": 0.1}, ("CS", "LS"))
    perfect = np.array([[1, 2, 3]] * 4, dtype=float)
    assert sr.kendalls_w(perfect) == pytest.approx(1.0)
    assert sr.cliffs_delta([0.1, 0.2], [0.8, 0.9]) == -1.0


def test_strategies_come_from_assignment_layer():
    table = sr.strategy_detectors()
    assert table["proposal_mapping"]["LS"] == "page_hinkley"
    assert set(table["adwin_all"].values()) == {"adwin"}
    assert ("CQ", "cusum") in sr.calibration_cells()
