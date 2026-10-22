from types import SimpleNamespace

import numpy as np
import pytest

from reasoning_monitor.dialogue_six_metrics import execute_calls, plan_calls, score_cell


class Scorers:
    def coherence(self, steps):
        return 0.75

    def similarity(self, left, right):
        return [0.8] * len(left)


def record(role='primary', index=0, answer='2'):
    return dict(role=role, call_index=index, answer=answer, gold_answer='2', ok=True,
                n_tokens=30, finish_reason='stop', raw_text='1. Add one.\n2. Result is two.')


def test_missing_samples_not_perfect_scores():
    values, status = score_cell([record()], Scorers(), 5, 2, [], [])
    assert values['CQ'] == 1
    for metric in ('CS', 'RS', 'SS'):
        assert np.isnan(values[metric])
        assert status[metric] == 'missing_samples'


def test_complete_metrics_and_c_subset():
    rows = [record(), record('repeat', 1), record('paraphrase', 0)]
    values, _ = score_cell(rows, Scorers(), 2, 1, [], [])
    assert values['CS'] == values['RS'] == 1
    assert values['SS'] == pytest.approx(0.8)
    rows[0]['answer'] = '3'
    values, status = score_cell(rows, Scorers(), 2, 1, [], [])
    assert np.isnan(values['RS']) and status['RS'] == 'C_subset_excluded'
    rows[0]['finish_reason'] = 'length'
    values, status = score_cell(rows, Scorers(), 2, 1, [], [])
    assert all(np.isnan(x) for x in values.values())


def test_efficiency_window_is_causal_and_bounded():
    tokens, correctness = [], []
    for i in range(6):
        row = record()
        row['n_tokens'] = 10 + i
        _, status = score_cell([row], Scorers(), 5, 2, tokens, correctness)
        assert status['ES'] == ('warmup' if i < 4 else 'measured')
    assert tokens == [11, 12, 13, 14, 15]


def test_same_context_branches_and_resume(tmp_path):
    config = dict(k_repeats=2, l_paraphrases=1, selected_dialogue_ids=['d'],
                  conditions=['user_only_history'], system_prompt='Solve',
                  models=[dict(name='m', max_tokens=100)])
    bank = dict(dialogues=[dict(dialogue_id='d', template='budget', turns=[
        dict(turn=i, qid=str(i), user_prompt=f'Update {i}', paraphrases=[f'Variant {i}'],
             gold_answer='2') for i in (1, 2)])])
    jobs = list(plan_calls(config, bank))
    assert len(jobs) == 6
    assert jobs[0]['messages'] == jobs[1]['messages']
    assert jobs[3]['messages'][0] == jobs[5]['messages'][0]
    assert 'Update 1' in jobs[3]['messages'][0]['content']
    assert 'Variant 1' not in jobs[3]['messages'][0]['content']

    class Client:
        calls = 0

        def generate_messages(self, messages, seed, gold):
            assert gold == ''
            self.calls += 1
            return SimpleNamespace(answer='2', raw_text='2', reasoning_steps=[], ok=True,
                                   n_tokens=1, latency_s=0, finish_reason='stop')

    client = Client()
    path = tmp_path / 'calls.jsonl'
    execute_calls(jobs[:2], {'m': client}, path, 'hash', lambda n: None)
    execute_calls(jobs, {'m': client}, path, 'hash', lambda n: None)
    assert client.calls == 6
    with pytest.raises(ValueError):
        execute_calls(jobs, {'m': client}, path, 'changed', lambda n: None)
