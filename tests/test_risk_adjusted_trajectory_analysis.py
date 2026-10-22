from __future__ import annotations

import numpy as np
import pandas as pd

from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (
    build_risk_set,
    evaluate_locked_holdout,
    fit_hazard_comparison,
    nested_grouped_validation,
    risk_set_summary,
)


def _family_rows(model: str, family: str, cq: list[int]) -> list[dict]:
    return [
        {
            "model": model,
            "family_id": family,
            "trajectory_level": level,
            "trajectory_split": "calibration",
            "domain": "arithmetic_chain",
            "qid": f"{family}-L{level}",
            "CQ_raw": correct,
            "RS_raw": 0.9 - level / 100,
            "LS_raw": 0.8 - level / 100,
            "ES_raw": 0.7 - level / 100,
            "call_failed": False,
        }
        for level, correct in enumerate(cq, start=1)
    ]


def test_risk_set_stops_after_the_first_failure():
    frame = pd.DataFrame(
        _family_rows("model-a", "event", [1, 1, 0, 0, 0, 0])
        + _family_rows("model-a", "censored", [1, 1, 1, 1, 1, 1])
        + _family_rows("model-a", "entry-failure", [0, 0, 0, 0, 0, 0])
    )

    risk = build_risk_set(frame)

    event = risk[risk["family_id"] == "event"]
    assert event["trajectory_level"].tolist() == [1, 2]
    assert event["failure_next"].tolist() == [0, 1]
    assert len(risk[risk["family_id"] == "censored"]) == 5
    assert "entry-failure" not in set(risk["family_id"])
    assert risk_set_summary(risk) == {
        "rows": 7,
        "families_entered": 2,
        "events": 1,
        "censored_families": 1,
        "events_by_model": {"model-a": 1},
    }


def _synthetic_risk_data(seed: int = 20260926) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    models = ("gpt4o_mini", "llama4_scout", "mistral_7b")
    templates = ("arithmetic_chain", "shortest_path", "precedence_schedule")
    for family_index in range(120):
        family = f"family-{family_index:03d}"
        template = templates[family_index % len(templates)]
        split = "training" if family_index < 84 else "holdout"
        family_risk = rng.normal(0, 0.8)
        for model_index, model in enumerate(models):
            for level in range(1, 6):
                quality = rng.normal(-0.35 * family_risk, 0.55)
                rs = float(1 / (1 + np.exp(-(quality + rng.normal(0, 0.08)))))
                ls = float(1 / (1 + np.exp(-(quality + rng.normal(0, 0.08)))))
                es = float(1 / (1 + np.exp(-(quality + rng.normal(0, 0.08)))))
                logit = -1.2 + 0.28 * level + 0.18 * model_index - 4.0 * quality
                probability = 1 / (1 + np.exp(-logit))
                event = int(rng.random() < probability)
                rows.append(
                    {
                        "model": model,
                        "family_id": family,
                        "trajectory_split": split,
                        "task_template": template,
                        "trajectory_level": level,
                        "next_level": level + 1,
                        "qid_t": f"{family}-{model}-L{level}",
                        "qid_next": f"{family}-{model}-L{level + 1}",
                        "failure_next": event,
                        "RS_raw": rs,
                        "LS_raw": ls,
                        "ES_raw": es,
                    }
                )
                if event:
                    break
    return pd.DataFrame(rows)


def test_full_hazard_model_recovers_prospective_signal():
    risk = _synthetic_risk_data()
    training = risk[risk["trajectory_split"] == "training"].reset_index(drop=True)
    holdout = risk[risk["trajectory_split"] == "holdout"].reset_index(drop=True)

    validation = nested_grouped_validation(
        training,
        c_grid=(0.1, 1.0, 10.0),
        outer_splits=3,
        inner_splits=3,
    )
    fitted = fit_hazard_comparison(
        training,
        c_grid=(0.1, 1.0, 10.0),
        n_splits=3,
    )
    result = evaluate_locked_holdout(fitted, holdout, n_bootstrap=300)

    assert validation["mean_improvement"] > 0
    assert result["improvement"] > 0
    assert result["ci95"][0] > 0
    assert len(result["supported_models"]) >= 2
    assert result["claim_supported"] is True
