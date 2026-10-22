import hashlib
from collections import Counter, defaultdict

from reasoning_monitor.data.controlled_trajectories import (
    LEVEL_SIZES,
    OUTPUT_CONTRACT,
    bank_queries,
    bank_sha256,
    generate_controlled_bank,
    item_gold_answer,
    write_controlled_bank,
)
from reasoning_monitor.experiment import (
    ExperimentConfig,
    _configured_stream,
    _validate_scenarios,
)
from reasoning_monitor.monitor import ReasoningMonitor
from reasoning_monitor.types import QueryObservation, Sample
from scripts.estimate_campaign import estimate


def test_rat_v1_bank_is_deterministic_and_solver_backed():
    first = generate_controlled_bank()
    second = generate_controlled_bank()

    assert bank_sha256(first) == bank_sha256(second)
    assert len(first["items"]) == 360
    assert len({item["family_id"] for item in first["items"]}) == 60
    assert all(item_gold_answer(item) == item["gold_answer"] for item in first["items"])


def test_rat_v1_bank_file_uses_the_canonical_cross_platform_hash(tmp_path):
    bank = generate_controlled_bank()
    path = tmp_path / "bank.json"

    write_controlled_bank(path, bank)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == bank_sha256(bank)


def test_rat_v1_splits_whole_families_without_leakage():
    bank = generate_controlled_bank()
    family_splits = defaultdict(set)
    for item in bank["items"]:
        family_splits[item["family_id"]].add(item["split"])

    assert all(len(splits) == 1 for splits in family_splits.values())
    assert Counter(next(iter(splits)) for splits in family_splits.values()) == {
        "calibration": 24,
        "development": 18,
        "holdout": 18,
    }
    assert {
        split: len(bank_queries(bank, split))
        for split in ("calibration", "development", "holdout")
    } == {
        "calibration": 144,
        "development": 108,
        "holdout": 108,
    }


def test_rat_v1_prompts_have_genuine_variants_and_a_strict_contract():
    bank = generate_controlled_bank()

    for item in bank["items"]:
        assert item["prompt"].endswith(OUTPUT_CONTRACT)
        assert len(item["paraphrases"]) == 3
        assert len(set(item["paraphrases"])) == 3
        assert item["prompt"] not in item["paraphrases"]
        assert all(text.endswith(OUTPUT_CONTRACT) for text in item["paraphrases"])


def test_rat_v1_structural_complexity_increases_with_level():
    bank = generate_controlled_bank()
    grouped = defaultdict(list)
    for item in bank["items"]:
        grouped[item["family_id"]].append(item)

    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item["level"])
        entities = [item["structure"]["n_entities"] for item in ordered]
        relations = [item["structure"]["n_relations"] for item in ordered]
        assert entities == sorted(entities)
        assert relations == sorted(relations)
        assert relations[0] < relations[-1]
        template = ordered[0]["template"]
        observed_sizes = (
            relations
            if template == "arithmetic_chain"
            else [item["structure"]["n_entities"] for item in ordered]
        )
        assert observed_sizes == list(LEVEL_SIZES[template])

        if template == "shortest_path":
            for item in ordered:
                backbone = {
                    (edge["left"], edge["right"]): edge["weight"]
                    for edge in item["structure"]["edges"]
                }
                expected = sum(
                    backbone[(node, node + 1)]
                    for node in range(item["structure"]["n_nodes"] - 1)
                )
                assert int(item["gold_answer"]) == expected


def test_monitor_queries_preserve_family_metadata():
    queries = bank_queries(generate_controlled_bank(), "holdout")

    assert len(queries) == 108
    assert queries[0].family_id is not None
    assert queries[0].trajectory_level == 1
    assert queries[0].trajectory_split == "holdout"
    assert queries[0].difficulty == "level_1"

    sample = Sample(
        answer="1",
        reasoning_steps=("1. Test",),
        latency_s=0.1,
        n_tokens=4,
        raw_text="1. Test\nFINAL_ANSWER: 1",
    )
    records = ReasoningMonitor._records_for(
        QueryObservation(query=queries[0], runs=(sample,), paraphrase_runs=())
    )
    assert records[0]["family_id"] == queries[0].family_id
    assert records[0]["trajectory_level"] == 1
    assert records[0]["trajectory_split"] == "holdout"


def test_experiment_builds_the_frozen_calibration_stream(tmp_path):
    bank_path = tmp_path / "bank.json"
    write_controlled_bank(bank_path, generate_controlled_bank())
    config = ExperimentConfig(
        stream_design="controlled_trajectory",
        datasets=[],
        scenarios=["RAT"],
        controlled_trajectory_bank=str(bank_path),
        controlled_trajectory_split="calibration",
    )

    _validate_scenarios(config.scenarios, config.stream_design)
    spec, stream, plan = _configured_stream(config, "RAT", {}, config.seed)

    assert spec.n_queries == len(stream) == len(plan) == 144
    assert len({query.family_id for query in stream}) == 24
    assert {query.trajectory_level for query in stream} == {1, 2, 3, 4, 5, 6}
    assert all(row["trajectory_split"] == "calibration" for row in plan)

    budget = estimate(
        {
            "seed": 20260926,
            "seeds": [20260926],
            "stream_design": "controlled_trajectory",
            "scenarios": ["RAT"],
            "controlled_trajectory_bank": "bank.json",
            "controlled_trajectory_split": "calibration",
            "window_sizes": [6],
            "k_repeats": 1,
            "l_paraphrases": 3,
            "models": [
                {"max_tokens": 768},
                {"max_tokens": 768},
                {"max_tokens": 768},
            ],
        },
        root=tmp_path,
    )
    assert budget["queries_per_model_seed_window"] == 144
    assert budget["model_calls"] == 1728
    assert budget["max_output_tokens"] == 1_327_104
