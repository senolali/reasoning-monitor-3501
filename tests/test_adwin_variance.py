"""ADWIN defect regression and the opt-in corrected detector (no network, no GPU)."""

import numpy as np

from reasoning_monitor.detectors.assignment import assign_detector
from reasoning_monitor.detectors.detectors import ADWIN, ADWINVariance, build_detector
from reasoning_monitor.metrics.dimensions import build_metric


def first_alarm(detector, values):
    for t, value in enumerate(values, start=1):
        alarm = detector.update(value, t, "m")
        if alarm is not None:
            return alarm.t
    return None


STEP = [0.8] * 50 + [0.2] * 50


def test_historical_adwin_defect_is_preserved():
    # Documented defect: the variance-free bound cannot be crossed on short streams.
    assert first_alarm(ADWIN(delta=0.05), STEP) is None


def test_corrected_adwin_detects_a_clear_step_after_the_change():
    # The published bound is conservative on short streams: a noiseless 0.6 drop
    # at t=51 is cut at t=77 with delta=0.05 and t=90 with delta=0.01.
    assert 50 < first_alarm(ADWINVariance(delta=0.05), STEP) <= 80
    assert 50 < first_alarm(ADWINVariance(delta=0.01), STEP) <= 95


def test_corrected_adwin_is_quiet_on_constant_and_rising_streams():
    assert first_alarm(ADWINVariance(delta=0.002), [0.8] * 100) is None
    assert first_alarm(ADWINVariance(delta=0.002), [0.2] * 50 + [0.8] * 50) is None


def test_corrected_adwin_false_alarm_rate_on_noisy_null():
    rng = np.random.default_rng(0)
    alarms = 0
    for _ in range(200):
        x = (rng.random(100) < 0.8).astype(float)
        window = np.convolve(x, np.ones(10) / 10, mode="valid")
        alarms += first_alarm(ADWINVariance(delta=0.002), window) is not None
    assert alarms / 200 <= 0.10


def test_corrected_strategies_are_opt_in():
    metric = build_metric("CS", window_size=10)
    assert assign_detector(metric, strategy="proposal_mapping").name == "adwin"
    assert assign_detector(metric, strategy="proposal_mapping_v").name == "adwin_v"
    assert assign_detector(build_metric("CQ", window_size=10),
                           strategy="adwin_all_v").name == "adwin_v"
    assert build_detector("adwin_v", delta=0.01).delta == 0.01
