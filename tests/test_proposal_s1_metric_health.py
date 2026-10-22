from pathlib import Path

import pandas as pd
import yaml

from reasoning_monitor.evaluation.proposal_s1_metric_health import (
    assess_model_records,
)
from reasoning_monitor.types import METRIC_ORDER
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _gate():
    return yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_metric_health_gpt_llama4_gate.yaml")
        .read_text(encoding="utf-8")
    )


def _fixture(diverse: bool = True):
    records = []
    rows = []
    t = 0
    for dataset in ("gsm8k", "arc_challenge", "math"):
        for item in range(5):
            t += 1
            qid = f"{dataset}-{item}"
            marker = "FINAL_CHOICE: B" if dataset == "arc_challenge" else "FINAL_ANSWER: 10"
            answer = "B" if dataset == "arc_challenge" else "10"
            for role, count in (("run", 5), ("paraphrase", 3)):
                for call_index in range(count):
                    suffix = call_index if diverse or role == "paraphrase" else 0
                    records.append(
                        {
                            "t": t,
                            "qid": qid,
                            "dataset": dataset,
                            "role": role,
                            "call_index": call_index,
                            "raw_text": f"1. Reason {suffix}.\n{marker}",
                            "answer": answer,
                            "gold_answer": answer,
                            "ok": True,
                            "n_tokens": 20,
                            "finish_reason": "stop",
                        }
                    )
            metric_values = {
                f"{metric}_raw": 0.2 + 0.1 * ((t + index) % 3)
                for index, metric in enumerate(METRIC_ORDER)
            }
            rows.append(
                {
                    "t": t,
                    "qid": qid,
                    "dataset": dataset,
                    "call_failed": False,
                    **metric_values,
                }
            )
    return records, pd.DataFrame(rows)


def test_metric_health_config_has_frozen_240_call_budget():
    raw = yaml.safe_load(
        (ROOT / "configs" / "proposal_s1_metric_health_gpt_llama4_asu.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(raw)["model_calls"] == 240
    assert raw["k_repeats"] == 5
    assert raw["l_paraphrases"] == 3
    assert raw["coherence_scorer"] == "nli"
    assert raw["stability_scorer"] == "bertscore"
    assert raw["publication_mode"] is True
    assert [model["name"] for model in raw["models"]] == [
        "gpt4o_mini",
        "llama4_scout",
    ]


def test_metric_health_gate_passes_complete_healthy_records():
    records, frame = _fixture()

    result = assess_model_records(records, frame, _gate(), True)

    assert result["n_records"] == 120
    assert result["k_repeat_diversity_rate"] == 1.0
    assert result["overall_gate_pass"] is True


def test_metric_health_gate_rejects_identical_k_repeats():
    records, frame = _fixture(diverse=False)

    result = assess_model_records(records, frame, _gate(), True)

    assert result["checks"]["repeat_diversity"] is False
    assert result["overall_gate_pass"] is False

