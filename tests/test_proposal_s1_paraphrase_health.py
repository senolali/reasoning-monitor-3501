from pathlib import Path

import pandas as pd
import yaml

from reasoning_monitor.evaluation.proposal_s1_paraphrase_health import (
    assess_model_records,
)
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    return yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_paraphrase_health_gpt_llama4_gate.yaml")
        .read_text(encoding="utf-8")
    )


def _fixture(rs_values: list[float]):
    records = []
    rows = []
    t = 0
    for dataset in ("gsm8k", "arc_challenge", "math"):
        for item in range(5):
            qid = f"{dataset}-{item}"
            t += 1
            marker = (
                "FINAL_CHOICE: B"
                if dataset == "arc_challenge"
                else "FINAL_ANSWER: 10"
            )
            answer = "B" if dataset == "arc_challenge" else "10"
            for role, count in (("run", 1), ("paraphrase", 3)):
                for call_index in range(count):
                    records.append(
                        {
                            "t": t,
                            "qid": qid,
                            "dataset": dataset,
                            "role": role,
                            "call_index": call_index,
                            "raw_text": f"1. Reason briefly.\n{marker}",
                            "answer": answer,
                            "gold_answer": answer,
                            "ok": True,
                            "n_tokens": 20,
                            "finish_reason": "stop",
                        }
                    )
            rows.append(
                {
                    "t": t,
                    "qid": qid,
                    "dataset": dataset,
                    "RS_raw": rs_values[(t - 1) % len(rs_values)],
                }
            )
    return records, pd.DataFrame(rows)


def test_paraphrase_health_config_has_frozen_120_call_budget():
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "proposal_s1_paraphrase_health_gpt_llama4_asu.yaml"
        ).read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 120
    assert raw["k_repeats"] == 1
    assert raw["l_paraphrases"] == 3
    assert [model["name"] for model in raw["models"]] == [
        "gpt4o_mini",
        "llama4_scout",
    ]


def test_paraphrase_health_gate_passes_observable_variable_rs():
    records, frame = _fixture([1.0, 2 / 3])

    result = assess_model_records(records, frame, _gate())

    assert result["rs_evaluable"] == 15
    assert result["rs_unique_values"] == 2
    assert result["overall_gate_pass"] is True


def test_paraphrase_health_gate_rejects_constant_rs():
    records, frame = _fixture([1.0])

    result = assess_model_records(records, frame, _gate())

    assert result["checks"]["rs_nondegeneracy"] is False
    assert result["overall_gate_pass"] is False
