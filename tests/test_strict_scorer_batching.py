import math

from reasoning_monitor.dialogue_six_metrics import StrictScorers


class FakeNLI:
    def __call__(self, inputs, **_kwargs):
        return [[
            {"label": "contradiction", "score": 0.9 if "NOT" in text else 0.1},
            {"label": "entailment", "score": 0.1 if "NOT" in text else 0.9},
            {"label": "neutral", "score": 0.0},
        ] for text in inputs]


def test_coherence_many_matches_scalar_formula_and_reports_progress():
    scorer = StrictScorers.__new__(StrictScorers)
    scorer.nli = FakeNLI()
    traces = [
        ["A", "B", "C"],
        ["A", "NOT B", "C"],
        ["single"],
        ["short", "trace"],
    ]
    scalar = [scorer.coherence(trace) for trace in traces]
    progress = []
    batched = scorer.coherence_many(traces, chunk_size=2,
                                    callback=lambda done, total: progress.append((done, total)))
    for expected, observed in zip(scalar, batched, strict=True):
        assert (math.isnan(expected) and math.isnan(observed)) or expected == observed
    assert progress[-1] == (5, 5)
