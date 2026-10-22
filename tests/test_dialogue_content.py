import pytest

from reasoning_monitor.evaluation.dialogue_content import (
    content_frame,
    equations,
    numeric_expression,
    prompt_delta,
)


def test_numeric_checks_and_unsupported_syntax():
    assert numeric_expression('(5 * 4) + 14 + 10') == 44
    assert equations('Score = 20 + 14 + 10 = 54')[0]['wrong']
    assert not equations('1. Total: 1,000 + 500 = 1,500.')[0]['wrong']
    assert equations('x + 2 = 3') == []
    assert equations('FINAL_ANSWER: 44') == []
    assert not equations('- 34 (North) + 39 (Central) + 55 (South) = 128')[0]['wrong']
    assert not equations('-3 + 2 = -1')[0]['wrong']
    with pytest.raises(ValueError):
        numeric_expression('__import__("os")')


def test_transition_uses_prompt_and_previous_model_answer_only():
    assert prompt_delta('software_incident', 'The response team has resolved 2 major incident(s).') == -4
    assert prompt_delta('study_cohort', 'The site has corrected the excluded count downward by 2.') == 2
    items = [dict(qid=str(t), dialogue_id='d', condition='user_only_history',
                  turn=t, template='research_budget', user_prompt=prompt,
                  answer=answer, raw_text='', gold_answer='999')
             for t, prompt, answer in [(1, 'initial', '100'),
                                       (2, 'A refund of 20 has been credited', '120'),
                                       (3, 'Record expenditure of 10', '115')]]
    frame, _ = content_frame(items)
    assert frame.transition_available.tolist() == [0, 1, 1]
    assert frame.transition_violation.tolist() == [0, 0, 1]
    items[2]['gold_answer'] = '115'
    assert content_frame(items)[0].equals(frame)
