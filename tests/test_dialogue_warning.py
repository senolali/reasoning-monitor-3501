from copy import deepcopy

import numpy as np

from reasoning_monitor.evaluation.dialogue_warning import (
    FEATURE_SETS,
    grouped_predictions,
    prediction_rows,
)


def records():
    return [
        dict(
            model="test", dialogue_id=f"d{d}", template=f"family{d}",
            split="calibration", qid=f"d{d}-{t}", turn=t,
            condition="user_only_history", role="primary", gold_answer="1",
            answer="0" if t >= 8 else "1", raw_text="FINAL_ANSWER: 1",
            context_chars=t * 100, n_tokens=t * 5, latency_s=0.1,
            ok=True, finish_reason="stop", intrinsic_updates=1,
        )
        for d in range(3) for t in range(1, 13)
    ]


def test_future_output_cannot_change_current_features():
    original = records()
    changed = deepcopy(original)
    changed[7]["n_tokens"] = 999
    first = prediction_rows(original)
    second = prediction_rows(changed)
    columns = sorted(set(sum(FEATURE_SETS.values(), [])))
    assert first.iloc[6][columns].equals(second.iloc[6][columns])
    assert first.iloc[6].next_error == 1
    assert first.iloc[6].before_first_error
    assert not first.iloc[7].before_first_error


def test_test_dialogue_labels_do_not_affect_its_predictions():
    frame = prediction_rows(records())
    original = grouped_predictions(frame, ["turn", "n_tokens"], "dialogue_id")
    changed = frame.copy()
    mask = changed.dialogue_id == "d0"
    changed.loc[mask, "next_error"] = 1 - changed.loc[mask, "next_error"]
    after = grouped_predictions(changed, ["turn", "n_tokens"], "dialogue_id")
    np.testing.assert_allclose(original[mask], after[mask])
