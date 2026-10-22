from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reasoning_monitor.data.loaders import (
    HF_SPECS,
    e2h_partition,
    e2h_refined_partition,
    tag_difficulty_e2h_extremes,
    tag_difficulty_e2h_quantiles,
)
from reasoning_monitor.evaluation.answer_parsing_audit import summarize_records
from reasoning_monitor.evaluation.e2h_amc_calibration import (
    SCENARIOS as AMC_SCENARIOS,
)
from reasoning_monitor.evaluation.e2h_amc_calibration import (
    _completion_health,
    assess_e2h_amc_calibration,
    evaluate_amc_model,
)
from reasoning_monitor.evaluation.e2h_benchmark_calibration import (
    ALPHA_PER_MODEL,
    MODELS,
    SCENARIOS,
    assess_e2h_benchmark_calibration,
    permutation_slope_test,
)
from reasoning_monitor.evaluation.e2h_benchmark_holdout import (
    TAG,
    assess_e2h_mistral_holdout,
    cq_holdout_gate,
)
from reasoning_monitor.evaluation.e2h_extreme_calibration import (
    ALPHA_PER_MODEL as EXTREME_ALPHA,
)
from reasoning_monitor.evaluation.e2h_extreme_calibration import (
    MODELS as EXTREME_MODELS,
)
from reasoning_monitor.evaluation.e2h_extreme_calibration import (
    SCENARIOS as EXTREME_SCENARIOS,
)
from reasoning_monitor.evaluation.e2h_extreme_calibration import (
    assess_e2h_extreme_calibration,
    evaluate_extreme_model,
)
from reasoning_monitor.evaluation.e2h_final_holdout import (
    EXPECTED_RECORDS as FINAL_EXPECTED_RECORDS,
)
from reasoning_monitor.evaluation.e2h_final_holdout import (
    FAMILY_ALPHA,
    FINAL_HOLDOUT_SPECS,
    assess_e2h_final_holdout,
    validate_final_holdout_config,
)
from reasoning_monitor.evaluation.e2h_gpt_detector_analysis import (
    analyze_e2h_gpt_detectors,
    validate_detector_analysis_config,
)
from reasoning_monitor.evaluation.trace_reparse_sensitivity import (
    _observations,
    validate_trace_reparse_config,
)
from reasoning_monitor.experiment import ExperimentConfig
from reasoning_monitor.metrics.dimensions import answers_equal, require_amc_symbolic_matcher
from reasoning_monitor.metrics.rqeval_adapters import CQAdapter
from reasoning_monitor.types import METRIC_ORDER, Query, QueryObservation, Sample
from scripts.estimate_campaign import estimate

ROOT = Path(__file__).resolve().parents[1]


def _queries(n: int = 300) -> list[Query]:
    return [
        Query(
            qid=f"e2h_gsm8k-{i:05d}",
            t=0,
            dataset="e2h_gsm8k",
            domain="math",
            prompt=f"Problem {i}",
            gold_answer=str(i),
            difficulty_rating=float(i),
            difficulty_rating_std=0.1,
            difficulty_quantile=(i + 0.5) / n,
            difficulty_model_avg_acc=1.0 - (i + 0.5) / n,
        )
        for i in range(n)
    ]


def test_official_quantiles_and_fixed_partitions_are_disjoint_and_complete():
    queries = _queries()
    all_tiers = tag_difficulty_e2h_quantiles(queries, seed=4)
    calibration = tag_difficulty_e2h_quantiles(
        queries, seed=4, partition="calibration"
    )
    holdout = tag_difficulty_e2h_quantiles(queries, seed=4, partition="holdout")

    all_ids = {q.qid for items in all_tiers.values() for q in items}
    calibration_ids = {q.qid for items in calibration.values() for q in items}
    holdout_ids = {q.qid for items in holdout.values() for q in items}
    assert calibration_ids.isdisjoint(holdout_ids)
    assert calibration_ids | holdout_ids == all_ids
    assert all(e2h_partition(qid) == "calibration" for qid in calibration_ids)
    assert all(q.difficulty_rating is not None for items in calibration.values() for q in items)
    assert max(q.difficulty_quantile for q in all_tiers["easy"]) <= 1 / 3
    assert min(q.difficulty_quantile for q in all_tiers["hard"]) > 2 / 3


def test_one_sided_permutation_test_accepts_a_strong_negative_trend():
    ratings = np.arange(90, dtype=float)
    cq = np.array([1.0] * 45 + [0.0] * 45)
    slope, p_value = permutation_slope_test(ratings, cq, permutations=999, seed=7)

    assert slope < 0
    assert p_value <= ALPHA_PER_MODEL


def test_official_e2h_calibration_config_is_frozen_at_270_calls():
    path = ROOT / "configs" / "e2h_benchmark_calibration_main_models_asu.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)

    assert cfg.stream_design == "e2h_benchmark_gsm8k"
    assert HF_SPECS["e2h_gsm8k"]["revision"] == (
        "55bc0d2fb10954151e669d2026b87fa896f2fa26"
    )
    assert cfg.datasets == ["e2h_gsm8k"]
    assert cfg.allow_synthetic_datasets is False
    assert cfg.difficulty_scheme == "e2h_rating_quantiles"
    assert cfg.difficulty_partition == "calibration"
    assert cfg.scenarios == ["ST3", "ST4", "ST5"]
    assert cfg.k_repeats == 1 and cfg.l_paraphrases == 0
    assert estimate(raw)["model_calls"] == 270


def test_call_free_assessor_marks_strong_models_eligible(tmp_path):
    config = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [44],
        "window_sizes": [5],
        "queries_per_regime": 30,
        "k_repeats": 1,
        "l_paraphrases": 0,
        "models": [{"name": model} for model in MODELS],
    }
    (tmp_path / "data").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"config": config}), encoding="utf-8"
    )
    qids: dict[str, list[str]] = {}
    for band in ("easy", "medium", "hard"):
        candidates = (f"{band}-{i}" for i in range(10_000))
        qids[band] = [qid for qid in candidates if e2h_partition(qid) == "calibration"][:30]

    for model in MODELS:
        for band, scenario in SCENARIOS.items():
            cq = [1.0] * 30 if band == "easy" else (
                [1.0] * 15 + [0.0] * 15 if band == "medium" else [0.0] * 30
            )
            offset = {"easy": 0.0, "medium": 1.0, "hard": 2.0}[band]
            frame = pd.DataFrame({
                "qid": qids[band],
                "difficulty": [band] * 30,
                "difficulty_rating": offset + np.arange(30) / 30,
                "CQ_raw": cq,
                "call_failed": [False] * 30,
            })
            tag = f"{scenario}_{model}_seed44_W5"
            frame.to_csv(tmp_path / "data" / f"per_query_{tag}.csv", index=False)
            records = [
                {"qid": qid, "role": "run", "ok": True}
                for qid in qids[band]
            ]
            (tmp_path / "data" / f"records_{tag}.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

    result = assess_e2h_benchmark_calibration(tmp_path)

    assert result["eligible_models"] == list(MODELS)
    assert result["any_model_eligible"] is True
    assert (tmp_path / "data" / "e2h_benchmark_calibration_gate.csv").exists()


def test_mistral_holdout_config_is_frozen_at_720_calls():
    path = ROOT / "configs" / "e2h_benchmark_mistral_holdout_kl_asu.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)

    assert cfg.seeds == [45]
    assert cfg.stream_design == "e2h_benchmark_gsm8k"
    assert cfg.difficulty_partition == "holdout"
    assert cfg.scenarios == ["ST2"]
    assert cfg.queries_per_regime == 30
    assert cfg.window_sizes == [10]
    assert cfg.k_repeats == 5 and cfg.l_paraphrases == 3
    assert [model["name"] for model in cfg.models] == ["mistral_7b"]
    assert cfg.publication_mode is True
    assert estimate(raw)["model_calls"] == 720


def test_holdout_cq_gate_uses_continuous_rating_and_easy_hard_gap():
    frame = pd.DataFrame({
        "t": np.arange(1, 91),
        "difficulty": ["easy"] * 30 + ["medium"] * 30 + ["hard"] * 30,
        "difficulty_rating": np.linspace(0.01, 0.99, 90),
        "CQ_raw": [1.0] * 30 + [1.0] * 15 + [0.0] * 15 + [0.0] * 30,
    })

    result = cq_holdout_gate(frame)

    assert result["cq_rating_slope"] < 0
    assert result["permutation_p_one_sided"] <= 0.05
    assert result["easy_hard_gap"] == 1.0
    assert result["cq_holdout_gate_pass"] is True


def test_call_free_holdout_assessor_accepts_complete_strong_run(tmp_path):
    config = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "holdout",
        "scenarios": ["ST2"],
        "seeds": [45],
        "window_sizes": [10],
        "queries_per_regime": 30,
        "k_repeats": 5,
        "l_paraphrases": 3,
        "use_rqeval": True,
        "coherence_scorer": "nli",
        "stability_scorer": "bertscore",
        "publication_mode": True,
        "detector_strategy": "proposal_mapping",
        "models": [{"name": "mistral_7b"}],
    }
    manifest = {
        "config": config,
        "preflight": {"problems": []},
        "degenerate_metrics": {},
        "failed_calls": {TAG: 0},
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    qids: dict[str, list[str]] = {}
    for band in ("easy", "medium", "hard"):
        candidates = (f"holdout-{band}-{i}" for i in range(10_000))
        qids[band] = [qid for qid in candidates if e2h_partition(qid) == "holdout"][:30]
    ordered_qids = qids["easy"] + qids["medium"] + qids["hard"]
    difficulties = ["easy"] * 30 + ["medium"] * 30 + ["hard"] * 30
    quantiles = np.concatenate([
        np.linspace(0.01, 0.32, 30),
        np.linspace(0.34, 0.65, 30),
        np.linspace(0.68, 0.99, 30),
    ])
    cq = [1.0] * 30 + [1.0] * 15 + [0.0] * 15 + [0.0] * 30
    varying = np.linspace(0.2, 0.9, 90)
    frame = pd.DataFrame({
        "t": np.arange(1, 91),
        "qid": ordered_qids,
        "difficulty": difficulties,
        "difficulty_rating": quantiles,
        "difficulty_quantile": quantiles,
        "call_failed": [False] * 90,
        "CQ_raw": cq,
        "CS_raw": varying,
        "RS_raw": varying[::-1],
        "LS_raw": varying,
        "ES_raw": varying[::-1],
        "SS_raw": varying,
    })
    frame.to_csv(data_dir / f"per_query_{TAG}.csv", index=False)

    records = []
    for qid in ordered_qids:
        records.extend(
            {
                "qid": qid,
                "role": "run",
                "call_index": index,
                "raw_text": f"{qid}-run-{index}",
                "ok": True,
            }
            for index in range(5)
        )
        records.extend(
            {
                "qid": qid,
                "role": "paraphrase",
                "call_index": index,
                "raw_text": f"{qid}-paraphrase-{index}",
                "ok": True,
            }
            for index in range(3)
        )
    (data_dir / f"records_{TAG}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = assess_e2h_mistral_holdout(tmp_path)

    assert result["overall_gate_pass"] is True
    assert (data_dir / "e2h_mistral_holdout_gate.csv").exists()
    assert (tmp_path / "E2H_MISTRAL_HOLDOUT_ASSESSMENT.md").exists()


def test_refined_e2h_partitions_and_extreme_bands_are_fixed_and_disjoint():
    queries = _queries(900)
    grouped = {
        name: {q.qid for q in queries if e2h_refined_partition(q.qid) == name}
        for name in ("calibration", "range_calibration", "final_holdout")
    }
    assert grouped["calibration"].isdisjoint(grouped["range_calibration"])
    assert grouped["calibration"].isdisjoint(grouped["final_holdout"])
    assert grouped["range_calibration"].isdisjoint(grouped["final_holdout"])
    assert set().union(*grouped.values()) == {q.qid for q in queries}

    tiers = tag_difficulty_e2h_extremes(
        queries, seed=46, partition="range_calibration"
    )
    assert max(q.difficulty_quantile for q in tiers["easy"]) <= 0.20
    assert all(0.40 <= q.difficulty_quantile <= 0.60 for q in tiers["medium"])
    assert min(q.difficulty_quantile for q in tiers["hard"]) >= 0.80
    assert all(
        e2h_refined_partition(q.qid) == "range_calibration"
        for items in tiers.values() for q in items
    )


def test_extreme_calibration_config_and_gate_are_frozen():
    path = ROOT / "configs" / "e2h_extreme_calibration_gpt_llama_asu.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)
    assert cfg.seeds == [46]
    assert cfg.difficulty_scheme == "e2h_extreme_quantiles"
    assert cfg.difficulty_partition == "range_calibration"
    assert cfg.scenarios == ["ST3", "ST4", "ST5"]
    assert [model["name"] for model in cfg.models] == ["gpt4o_mini", "llama3_8b"]
    assert cfg.k_repeats == 1 and cfg.l_paraphrases == 0
    assert estimate(raw)["model_calls"] == 180

    frame = pd.DataFrame({
        "difficulty": ["easy"] * 30 + ["medium"] * 30 + ["hard"] * 30,
        "difficulty_rating": np.linspace(0.01, 0.99, 90),
        "CQ_raw": [1.0] * 30 + [1.0] * 15 + [0.0] * 15 + [0.0] * 30,
    })
    gate = evaluate_extreme_model(frame)
    assert gate["easy_hard_gap"] == 1.0
    assert gate["permutation_p_one_sided"] <= EXTREME_ALPHA


def test_call_free_extreme_assessor_accepts_complete_strong_models(tmp_path):
    config = {
        "stream_design": "e2h_benchmark_gsm8k",
        "datasets": ["e2h_gsm8k"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_extreme_quantiles",
        "difficulty_partition": "range_calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [46],
        "window_sizes": [5],
        "queries_per_regime": 30,
        "k_repeats": 1,
        "l_paraphrases": 0,
        "models": [{"name": model} for model in EXTREME_MODELS],
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"config": config}), encoding="utf-8"
    )

    quantiles = {
        "easy": np.linspace(0.01, 0.20, 30),
        "medium": np.linspace(0.40, 0.60, 30),
        "hard": np.linspace(0.80, 0.99, 30),
    }
    qids: dict[str, list[str]] = {}
    for band in ("easy", "medium", "hard"):
        candidates = (f"range-{band}-{i}" for i in range(10_000))
        qids[band] = [
            qid for qid in candidates
            if e2h_refined_partition(qid) == "range_calibration"
        ][:30]

    for model in EXTREME_MODELS:
        for band, scenario in EXTREME_SCENARIOS.items():
            cq = [1.0] * 30 if band == "easy" else (
                [1.0] * 15 + [0.0] * 15 if band == "medium" else [0.0] * 30
            )
            frame = pd.DataFrame({
                "qid": qids[band],
                "difficulty": [band] * 30,
                "difficulty_rating": quantiles[band],
                "difficulty_quantile": quantiles[band],
                "CQ_raw": cq,
                "call_failed": [False] * 30,
            })
            tag = f"{scenario}_{model}_seed46_W5"
            frame.to_csv(data_dir / f"per_query_{tag}.csv", index=False)
            records = [
                {"qid": qid, "role": "run", "ok": True}
                for qid in qids[band]
            ]
            (data_dir / f"records_{tag}.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

    result = assess_e2h_extreme_calibration(tmp_path)

    assert result["eligible_models"] == list(EXTREME_MODELS)
    assert (data_dir / "e2h_extreme_calibration_gate.csv").exists()


def test_e2h_amc_config_and_symbolic_answer_matching_are_frozen():
    path = ROOT / "configs" / "e2h_amc_calibration_gpt4o_mini_asu.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**raw)

    assert HF_SPECS["e2h_amc"]["name"] == "E2H-AMC"
    assert HF_SPECS["e2h_amc"]["revision"] == (
        "55bc0d2fb10954151e669d2026b87fa896f2fa26"
    )
    assert cfg.stream_design == "e2h_benchmark_amc"
    assert cfg.datasets == ["e2h_amc"]
    assert cfg.allow_synthetic_datasets is False
    assert cfg.difficulty_partition == "calibration"
    assert cfg.scenarios == ["ST3", "ST4", "ST5"]
    assert [model["name"] for model in cfg.models] == ["gpt4o_mini"]
    assert cfg.k_repeats == 1 and cfg.l_paraphrases == 0
    assert estimate(raw)["model_calls"] == 90

    require_amc_symbolic_matcher()
    assert answers_equal(r"\boxed{\frac{1}{32}}", "1/32", "e2h_amc")
    assert answers_equal(r"4 \cdot 3^{7}", r"3^{7} \cdot 4", "e2h_amc")
    assert answers_equal(r"2^{2005}", "2^2005", "e2h_amc")
    assert answers_equal("} \\frac{3}{8}\n\\]", r"\frac38", "e2h_amc")
    assert answers_equal("} A=B\n\\]", "A=B", "e2h_amc")
    assert answers_equal(r"6\frac{1}{2}", "6.5", "e2h_amc")
    assert answers_equal("Final answer: Answer: 36", "36", "e2h_amc")
    assert answers_equal(
        r"\left(-\frac{5\sqrt{3}}{3},5\right)",
        r"\left(-\frac{5}{3}\sqrt{3},5\right)",
        "e2h_amc",
    )
    assert not answers_equal(r"2^{2011}-2012", "2012", "e2h_amc")

    query = Query(
        qid="e2h_amc-test", t=1, dataset="e2h_amc", domain="math_competition",
        prompt="Test", gold_answer=r"\frac{1}{32}",
    )
    sample = Sample(
        answer="1/32", reasoning_steps=(), latency_s=0.0, n_tokens=1
    )
    observation = QueryObservation(query=query, runs=(sample,), paraphrase_runs=())
    assert CQAdapter(window_size=5).per_query_value(observation) == 1.0


def test_call_free_amc_assessor_accepts_a_complete_strong_run(tmp_path):
    config = {
        "stream_design": "e2h_benchmark_amc",
        "datasets": ["e2h_amc"],
        "allow_synthetic_datasets": False,
        "difficulty_scheme": "e2h_rating_quantiles",
        "difficulty_partition": "calibration",
        "scenarios": ["ST3", "ST4", "ST5"],
        "seeds": [47],
        "window_sizes": [5],
        "queries_per_regime": 30,
        "k_repeats": 1,
        "l_paraphrases": 0,
        "models": [{"name": "gpt4o_mini", "max_tokens": 512}],
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"config": config}), encoding="utf-8"
    )

    quantiles = {
        "easy": np.linspace(0.01, 0.32, 30),
        "medium": np.linspace(0.34, 0.65, 30),
        "hard": np.linspace(0.68, 0.99, 30),
    }
    qids: dict[str, list[str]] = {}
    for band in ("easy", "medium", "hard"):
        candidates = (f"amc-{band}-{i}" for i in range(10_000))
        qids[band] = [
            qid for qid in candidates if e2h_partition(qid) == "calibration"
        ][:30]

    frames = []
    for band, scenario in AMC_SCENARIOS.items():
        cq = [1.0] * 30 if band == "easy" else (
            [1.0] * 15 + [0.0] * 15 if band == "medium" else [0.0] * 30
        )
        frame = pd.DataFrame({
            "qid": qids[band],
            "dataset": ["e2h_amc"] * 30,
            "difficulty": [band] * 30,
            "difficulty_rating": quantiles[band],
            "difficulty_quantile": quantiles[band],
            "CQ_raw": cq,
            "call_failed": [False] * 30,
        })
        frames.append(frame)
        tag = f"{scenario}_gpt4o_mini_seed47_W5"
        frame.to_csv(data_dir / f"per_query_{tag}.csv", index=False)
        records = [
            {
                "qid": qid,
                "role": "run",
                "ok": True,
                "answer": "1" if score else "0",
                "gold_answer": "1",
                "raw_text": f"Answer: {'1' if score else '0'}",
                "n_tokens": 10,
                "finish_reason": "stop",
            }
            for qid, score in zip(qids[band], cq, strict=True)
        ]
        (data_dir / f"records_{tag}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    gate = evaluate_amc_model(pd.concat(frames, ignore_index=True))
    assert gate["easy_hard_gap"] == 1.0
    result = assess_e2h_amc_calibration(tmp_path)

    assert result["eligible_for_amc_holdout"] is True
    assert (data_dir / "e2h_amc_calibration_gate.csv").exists()
    assert (tmp_path / "E2H_AMC_CALIBRATION_ASSESSMENT.md").exists()


def test_amc_completion_health_rejects_provider_and_legacy_truncation():
    records = [
        {"n_tokens": 120, "finish_reason": "stop", "raw_text": "Answer: 2"},
        {"n_tokens": 511, "finish_reason": "length", "raw_text": "unfinished"},
        {"n_tokens": 512, "raw_text": "legacy unfinished"},
    ]

    health = _completion_health(records, max_tokens=512)

    assert health["n_truncated"] == 2
    assert health["n_answer_markers"] == 1
    assert health["execution_valid"] is False


def test_amc_retry_changes_only_output_budget_and_destination():
    original = yaml.safe_load(
        (ROOT / "configs" / "e2h_amc_calibration_gpt4o_mini_asu.yaml")
        .read_text(encoding="utf-8")
    )
    retry = yaml.safe_load(
        (ROOT / "configs" / "e2h_amc_calibration_gpt4o_mini_retry_asu.yaml")
        .read_text(encoding="utf-8")
    )

    assert estimate(retry)["model_calls"] == 90
    assert retry["output_dir"] != original["output_dir"]
    assert retry["models"][0]["max_tokens"] == 2048
    retry_comparable = {
        **retry,
        "output_dir": original["output_dir"],
        "models": [{**retry["models"][0], "max_tokens": 512}],
    }
    assert retry_comparable == original


def test_section47_final_holdout_configs_are_frozen_and_budgeted():
    expected = {
        "llama3_8b": ("e2h_gsm8k", "e2h_extreme_quantiles", "final_holdout"),
        "gpt4o_mini": ("e2h_amc", "e2h_rating_quantiles", "holdout"),
        "mistral_7b": ("e2h_gsm8k", "e2h_rating_quantiles", "final_holdout"),
    }
    for track, spec in FINAL_HOLDOUT_SPECS.items():
        raw = yaml.safe_load(
            (ROOT / "configs" / spec.config_name).read_text(encoding="utf-8")
        )
        validate_final_holdout_config(raw, spec)
        dataset, scheme, partition = expected[track]
        assert raw["datasets"] == [dataset]
        assert raw["difficulty_scheme"] == scheme
        assert raw["difficulty_partition"] == partition
        assert raw["models"][0]["max_tokens"] == 2048
        assert estimate(raw)["model_calls"] == FINAL_EXPECTED_RECORDS


def test_section47_call_free_assessor_accepts_complete_gpt_track(tmp_path):
    spec = FINAL_HOLDOUT_SPECS["gpt4o_mini"]
    config = yaml.safe_load(
        (ROOT / "configs" / spec.config_name).read_text(encoding="utf-8")
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "config": config,
            "failed_calls": {spec.tag: 0},
            "preflight": {"problems": []},
            "degenerate_metrics": {},
        }),
        encoding="utf-8",
    )

    quantiles = {
        "easy": np.linspace(0.01, 0.32, 30),
        "medium": np.linspace(0.34, 0.65, 30),
        "hard": np.linspace(0.68, 0.99, 30),
    }
    qids: dict[str, list[str]] = {}
    for band in ("easy", "medium", "hard"):
        candidates = (f"section47-amc-{band}-{i}" for i in range(20_000))
        qids[band] = [
            qid for qid in candidates if e2h_partition(qid) == "holdout"
        ][:30]

    rows = []
    records = []
    t = 0
    for band in ("easy", "medium", "hard"):
        scores = [1.0] * 30 if band == "easy" else (
            [1.0] * 15 + [0.0] * 15 if band == "medium" else [0.0] * 30
        )
        for index, (qid, score) in enumerate(
            zip(qids[band], scores, strict=True)
        ):
            t += 1
            row = {
                "t": t,
                "qid": qid,
                "dataset": "e2h_amc",
                "difficulty": band,
                "difficulty_rating": float(quantiles[band][index]),
                "difficulty_quantile": float(quantiles[band][index]),
                "call_failed": False,
            }
            for metric_index, metric in enumerate(METRIC_ORDER):
                row[f"{metric}_raw"] = (
                    score
                    if metric == "CQ"
                    else 0.15 + 0.1 * ((t + metric_index) % 7)
                )
            rows.append(row)
            answer = "1" if score else "0"
            for role, count in (("run", 5), ("paraphrase", 3)):
                for call_index in range(count):
                    records.append({
                        "qid": qid,
                        "role": role,
                        "call_index": call_index,
                        "ok": True,
                        "answer": answer,
                        "gold_answer": "1",
                        "raw_text": (
                            f"Reasoning {role} variant {call_index}. Answer: {answer}"
                        ),
                        "n_tokens": 100,
                        "finish_reason": "stop",
                    })

    pd.DataFrame(rows).to_csv(
        data_dir / f"per_query_{spec.tag}.csv", index=False
    )
    (data_dir / f"records_{spec.tag}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = assess_e2h_final_holdout(tmp_path, "gpt4o_mini")
    gate = pd.read_csv(data_dir / "e2h_final_holdout_gate.csv").iloc[0]

    assert result["overall_gate_pass"] is True
    assert result["criteria"]["family_alpha"] == FAMILY_ALPHA
    assert gate["n_records"] == FINAL_EXPECTED_RECORDS
    assert gate["n_truncated"] == 0
    assert gate["easy_hard_gap"] == 1.0
    assert gate["metric_health_pass"]
    assert gate["execution_valid"]
    assert (data_dir / "e2h_final_holdout_cq_rescore.csv").exists()
    assert (tmp_path / "E2H_FINAL_HOLDOUT_ASSESSMENT.md").exists()


def test_section49_detector_config_is_frozen():
    config = yaml.safe_load(
        (ROOT / "configs" / "e2h_gpt_detector_analysis.yaml").read_text(
            encoding="utf-8"
        )
    )

    validate_detector_analysis_config(config)


def test_section49_call_free_analysis_requires_two_transition_lead(tmp_path):
    config_path = tmp_path / "section49.yaml"
    config_path.write_text(
        (ROOT / "configs" / "e2h_gpt_detector_analysis.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "gpt4o_mini",
                "dataset": "e2h_amc",
                "execution_valid": "true",
                "metric_health_pass": True,
                "cq_holdout_gate_pass": True,
                "overall_gate_pass": True,
            }
        ]
    ).to_csv(data_dir / "e2h_final_holdout_gate.csv", index=False)
    (tmp_path / "e2h_final_holdout_manifest.json").write_text(
        json.dumps({"source": "synthetic-test"}), encoding="utf-8"
    )

    rows = []
    phase_values = {"easy": 0.9, "medium": 0.6, "hard": 0.3}
    for t in range(1, 91):
        difficulty = "easy" if t <= 30 else "medium" if t <= 60 else "hard"
        row = {"t": t, "difficulty": difficulty}
        for metric_index, metric in enumerate(METRIC_ORDER):
            row[f"{metric}_raw"] = phase_values[difficulty] - metric_index * 0.01
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        data_dir / "per_query_ST2_gpt4o_mini_seed48_W10.csv", index=False
    )

    alarms = [
        {"t": 35, "metric": "CS", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 40, "metric": "CQ", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 65, "metric": "CS", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 70, "metric": "CQ", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 20, "metric": "LS", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 34, "metric": "LS", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 64, "metric": "LS", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 39, "metric": "ES", "detector": "cusum", "statistic": 1.0, "level": "drift"},
        {"t": 69, "metric": "ES", "detector": "cusum", "statistic": 1.0, "level": "warning"},
    ]
    pd.DataFrame(alarms).to_csv(
        data_dir / "alarms_ST2_gpt4o_mini_seed48_W10.csv", index=False
    )

    result = analyze_e2h_gpt_detectors(tmp_path, config_path)
    timing = pd.read_csv(data_dir / "section49_alarm_timing.csv")
    gate = pd.read_csv(data_dir / "section49_early_warning_gate.csv")

    assert result["no_model_calls"] is True
    assert result["overall_early_warning_gate_pass"] is True
    assert result["passing_metrics"] == ["CS"]
    cs_first = timing.loc[
        (timing["transition"] == "easy_to_medium") & (timing["metric"] == "CS")
    ].iloc[0]
    assert cs_first["delay_queries"] == 5
    assert np.isclose(cs_first["normalized_delay"], 5 / 30)
    assert cs_first["lead_over_cq_queries"] == 5
    assert not gate.loc[gate["metric"] == "LS", "early_warning_gate_pass"].iloc[0]
    assert not gate.loc[gate["metric"] == "ES", "early_warning_gate_pass"].iloc[0]
    assert (tmp_path / "SECTION49_GPT_DETECTOR_ANALYSIS.md").exists()


def test_answer_parsing_audit_separates_answer_and_trace_failures():
    records = []
    shared_prefix = ". ".join(f"Step {index}" for index in range(12))
    for call_index, answer in enumerate(("7", "8", "7", "8", "7")):
        raw = f"{shared_prefix}. Variant {call_index}. Answer: {answer}"
        records.append(
            {
                "qid": "q1",
                "dataset": "e2h_gsm8k",
                "difficulty": "hard",
                "role": "run",
                "call_index": call_index,
                "gold_answer": "7",
                "raw_text": raw,
                "answer": answer,
                "reasoning_steps": [f"Step {index}" for index in range(12)],
            }
        )

    summary, record_frame, query_frame = summarize_records(
        records, "test_model", "e2h_gsm8k"
    )

    assert summary["cq_changed_by_raw_rescore"] == 0
    assert summary["cs_changed_queries"] == 0
    assert summary["capped_trace_records"] == 5
    assert summary["trace_collapsed_queries"] == 1
    assert not record_frame["stored_trace_has_answer_marker"].any()
    assert query_frame.iloc[0]["answer_equivalence_classes"] == 2


def test_section51_trace_reparse_config_is_frozen():
    config = yaml.safe_load(
        (ROOT / "configs" / "e2h_gpt_trace_reparse_sensitivity.yaml").read_text(
            encoding="utf-8"
        )
    )

    validate_trace_reparse_config(config)


def test_section51_replay_accepts_source_rows_without_solution_steps():
    per_query = pd.DataFrame(
        [
            {
                "t": 1,
                "qid": "q1",
                "dataset": "e2h_amc",
                "difficulty": "easy",
                "difficulty_rating": 0.1,
                "difficulty_rating_std": 0.01,
                "difficulty_quantile": 0.1,
                "difficulty_model_avg_acc": np.nan,
            }
        ]
    )
    records = []
    for role, count in (("run", 5), ("paraphrase", 3)):
        for index in range(count):
            records.append(
                {
                    "qid": "q1",
                    "role": role,
                    "call_index": index,
                    "domain": "math_competition",
                    "prompt": "Question",
                    "gold_answer": "A",
                    "raw_text": "Reasoning.\n\nAnswer: A",
                    "answer": "A",
                    "n_tokens": 5,
                    "ok": True,
                }
            )

    observations = _observations(per_query, records)

    assert len(observations) == 1
    assert observations[0].query.solution_steps is None
    assert observations[0].primary.reasoning_steps[-1] == "Answer: A"
