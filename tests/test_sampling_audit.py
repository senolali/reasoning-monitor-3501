from copy import deepcopy

import pytest

from reasoning_monitor.dialogue_experiment import messages_sha256
from reasoning_monitor.evaluation.sampling_audit import audit_probe


class Scorer:
    def coherence(self, steps):
        return 0.8

    def similarity(self, left, right):
        return [0.9] * len(left)


def fixture_rows():
    rows = []
    for turn in (2, 11):
        for i in range(5):
            messages = [dict(role='user', content=f'Turn {turn}')]
            rows.append(dict(model='m', dialogue_id='d', condition='user_only_history',
                             turn=turn, role='primary' if i == 0 else 'repeat', call_index=i,
                             config_hash='f', messages=messages, messages_sha256=messages_sha256(messages),
                             max_tokens=100, n_tokens=20, ok=True, finish_reason='stop',
                             gold_answer='2', answer='2', raw_text=(
                                 f'1. Add {i if turn == 11 else 0}.\n2. Compute.\nFINAL_ANSWER: 2')))
    return rows


def test_constant_cell_does_not_fail_model_and_sparse_metrics_missing():
    rows = fixture_rows()
    result = audit_probe(rows, rows, 'f', Scorer())
    assert result['models']['m']['posthoc_acquisition_readiness']
    assert result['models']['m']['diverse_cells'] == 1
    for cell in result['cells']:
        assert cell['scores']['CS'] == 1
        assert cell['unique_answers'] == 1
        assert cell['scores']['RS'] is None and cell['scores']['ES'] is None


def test_all_constant_and_format_failure_remain_distinct_from_variability():
    rows = fixture_rows()
    for r in rows:
        r['raw_text'] = '1. Add.\n2. Compute.\nFINAL_ANSWER: 2'
    result = audit_probe(rows, rows, 'f', Scorer())
    assert result['models']['m']['operational_pass']
    assert not result['models']['m']['posthoc_acquisition_readiness']
    rows[0]['raw_text'] = 'FINAL_ANSWER: 3'
    result = audit_probe(rows, rows, 'f', Scorer())
    assert not result['models']['m']['operational_pass']
    assert result['cells'][0]['scores']['CS'] is None


def test_provenance_and_duplicate_records_fail_closed():
    rows = fixture_rows()
    expected = deepcopy(rows)
    rows[0]['messages'][0]['content'] = 'Changed'
    with pytest.raises(ValueError, match='provenance'):
        audit_probe(rows, expected, 'f', Scorer())
    with pytest.raises(ValueError, match='duplicate'):
        audit_probe(expected[:-1] + [expected[0]], expected, 'f', Scorer())
