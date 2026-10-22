import runpy
from pathlib import Path

import pandas as pd


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / 'run_six_metric_mini.py'))


def test_fixed_plan_no_full_campaign(monkeypatch):
    m = module(monkeypatch)
    config, jobs, protocol, fingerprint = m['mini_plan']()
    assert len(jobs) == 336
    assert {j['dialogue_id'] for j in jobs} == {'cdr-v14-budget-002'}
    assert len({j['model'] for j in jobs}) == 2
    assert protocol['maximum_output_tokens'] == 688128
    assert len(fingerprint) == 64


def test_first_error_uses_previous_turn_not_error_turn(monkeypatch):
    m = module(monkeypatch)
    rows = []
    for turn in (1, 2, 3):
        rows.append(dict(model='m', condition='history', turn=turn,
                         **{metric: (0.0 if turn == 3 else 1.0) for metric in m['METRICS']},
                         **{f'{metric}_status': 'measured' for metric in m['METRICS']}))
    report = m['assess'](pd.DataFrame(rows))
    assert report['events'][0]['first_error_turn'] == 3
    assert report['events'][0]['immediately_preceding_metrics']['CS'] == 1
    assert not report['early_warning_established']
    rows[-1]['CQ'] = 1
    report = m['assess'](pd.DataFrame(rows))
    assert report['events'][0]['outcome'] == 'no_error_observed'
    assert report['events'][0]['immediately_preceding_metrics']['CS'] is None
