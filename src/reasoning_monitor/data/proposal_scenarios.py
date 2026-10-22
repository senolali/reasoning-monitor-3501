"""Accepted-proposal multi-dataset stream scenarios.

This module is intentionally separate from ``data.scenarios``.  The latter
contains the frozen E2H-GSM8K design and its ST1-ST6 identifiers; keeping the
two catalogues apart prevents a proposal-faithful run from silently changing
or reinterpreting an archived E2H campaign.

Only S1 is implemented for the first validation pilot.  It moves gradually
from GSM8K through ARC-Challenge to MATH in six composition phases while using
each dataset equally often overall.
"""

from __future__ import annotations

import random
from dataclasses import replace

from reasoning_monitor.data.loaders import LoadedDataset
from reasoning_monitor.types import DriftType, Query, ScenarioSpec

_DEFAULT_QUERIES_PER_DATASET = 50
_DATASET_ORDER = ("gsm8k", "arc_challenge", "math")
_DIFFICULTY_BY_DATASET = {
    "gsm8k": "easy",
    "arc_challenge": "medium",
    "math": "hard",
}


def proposal_s1_spec(queries_per_dataset: int = _DEFAULT_QUERIES_PER_DATASET) -> ScenarioSpec:
    """Return the gradual S1 specification at the requested pilot size.

    Six equal phases are used.  Consequently, ``queries_per_dataset`` must be
    positive and even: each phase then contains half that many questions.
    The first and last phases are stationary endpoints; the four intervening
    phases form the gradual transition.
    """
    if queries_per_dataset < 2 or queries_per_dataset % 2:
        raise ValueError("proposal S1 queries_per_dataset must be a positive even integer")
    phase_size = queries_per_dataset // 2
    return ScenarioSpec(
        name="S1",
        drift_type=DriftType.GRADUAL,
        change_point=phase_size,
        n_queries=3 * queries_per_dataset,
        transition_width=4 * phase_size,
        description=(
            "Gradual dataset-composition shift: "
            "GSM8K -> ARC-Challenge -> MATH"
        ),
    )


PROPOSAL_SCENARIOS: dict[str, ScenarioSpec] = {"S1": proposal_s1_spec()}


def proposal_answer_order_calibration_spec(
    queries_per_dataset: int = 10,
) -> ScenarioSpec:
    if queries_per_dataset < 1:
        raise ValueError("answer-order calibration requires at least one item per dataset")
    return ScenarioSpec(
        name="S1",
        drift_type=DriftType.SUDDEN,
        change_point=queries_per_dataset,
        n_queries=2 * queries_per_dataset,
        description="Prompt-order calibration: frozen GSM8K then MATH items",
        regimes=(
            ("easy", queries_per_dataset),
            ("hard", queries_per_dataset),
        ),
    )


def build_proposal_answer_order_calibration_stream(
    datasets: dict[str, LoadedDataset],
    sequence: list[dict[str, str]],
    queries_per_dataset: int = 10,
) -> tuple[ScenarioSpec, list[Query], list[dict]]:
    """Materialise the frozen Section 57 QID sequence without resampling."""
    spec = proposal_answer_order_calibration_spec(queries_per_dataset)
    expected_counts = {"gsm8k": queries_per_dataset, "math": queries_per_dataset}
    observed_counts = {
        name: sum(item.get("dataset") == name for item in sequence)
        for name in expected_counts
    }
    if len(sequence) != spec.n_queries or observed_counts != expected_counts:
        raise ValueError(
            "answer-order calibration selection must contain exactly "
            f"{expected_counts}; found {observed_counts}"
        )
    if len({item.get("qid") for item in sequence}) != len(sequence):
        raise ValueError("answer-order calibration QIDs must be unique")

    lookup = {
        query.qid: query
        for dataset in datasets.values()
        for query in dataset.queries
    }
    missing = [item["qid"] for item in sequence if item["qid"] not in lookup]
    if missing:
        raise ValueError(f"frozen calibration QIDs missing from loaded data: {missing[:3]}")

    stream: list[Query] = []
    plan: list[dict] = []
    for t, item in enumerate(sequence, start=1):
        source = lookup[item["qid"]]
        if source.dataset != item["dataset"]:
            raise ValueError(f"dataset mismatch for frozen QID {source.qid}")
        difficulty = _DIFFICULTY_BY_DATASET[source.dataset]
        stream.append(replace(source, t=t, difficulty=difficulty))
        plan.append(
            {
                "t": t,
                "phase": 1 if source.dataset == "gsm8k" else 2,
                "dataset": source.dataset,
                "difficulty_role": difficulty,
                "qid": source.qid,
            }
        )
    return spec, stream, plan


def proposal_frozen_selection_spec(n_queries: int) -> ScenarioSpec:
    """Return a stationary specification for a frozen development selection."""
    if n_queries < 1:
        raise ValueError("frozen proposal selection requires at least one item")
    return ScenarioSpec(
        name="S1",
        drift_type=DriftType.NONE,
        change_point=0,
        n_queries=n_queries,
        description="Frozen proposal development selection",
        regimes=(("stress", n_queries),),
    )


def build_proposal_frozen_selection_stream(
    datasets: dict[str, LoadedDataset],
    sequence: list[dict[str, str]],
    n_queries: int,
) -> tuple[ScenarioSpec, list[Query], list[dict]]:
    """Materialise an auditable QID sequence without sampling or reordering."""
    spec = proposal_frozen_selection_spec(n_queries)
    if len(sequence) != n_queries:
        raise ValueError(
            f"frozen proposal selection needs {n_queries} items; "
            f"found {len(sequence)}"
        )
    if len({item.get("qid") for item in sequence}) != len(sequence):
        raise ValueError("frozen proposal selection QIDs must be unique")

    lookup = {
        query.qid: query
        for dataset in datasets.values()
        for query in dataset.queries
    }
    missing = [item["qid"] for item in sequence if item["qid"] not in lookup]
    if missing:
        raise ValueError(f"frozen proposal QIDs missing from loaded data: {missing[:3]}")

    stream: list[Query] = []
    plan: list[dict] = []
    for t, item in enumerate(sequence, start=1):
        source = lookup[item["qid"]]
        if source.dataset != item["dataset"]:
            raise ValueError(f"dataset mismatch for frozen QID {source.qid}")
        difficulty = _DIFFICULTY_BY_DATASET[source.dataset]
        paraphrases = source.paraphrases
        if "paraphrases" in item:
            delimiter = "\nOutput format:\n"
            if delimiter not in source.prompt:
                raise ValueError(
                    f"frozen paraphrases require an output contract for {source.qid}"
                )
            source_body, contract = source.prompt.split(delimiter, maxsplit=1)
            expected_source = str(item.get("source_prompt_body") or "").strip()
            if expected_source != source_body.strip():
                raise ValueError(
                    f"source prompt body changed for frozen QID {source.qid}"
                )
            bodies = tuple(str(text).strip() for text in item["paraphrases"])
            if not bodies or any(not text for text in bodies):
                raise ValueError(f"empty frozen paraphrase for {source.qid}")
            if len(set(bodies)) != len(bodies):
                raise ValueError(f"duplicate frozen paraphrase for {source.qid}")
            if source_body.strip() in bodies:
                raise ValueError(f"original prompt reused as paraphrase for {source.qid}")
            paraphrases = tuple(
                f"{body}{delimiter}{contract}" for body in bodies
            )
        stream.append(
            replace(source, t=t, difficulty=difficulty, paraphrases=paraphrases)
        )
        plan.append(
            {
                "t": t,
                "phase": 1,
                "dataset": source.dataset,
                "difficulty_role": difficulty,
                "selection_role": item.get("selection_role", "selected"),
                "qid": source.qid,
            }
        )
    return spec, stream, plan


def _phase_counts(queries_per_dataset: int) -> tuple[dict[str, int], ...]:
    """Six gradual phases with exact equal totals across the three datasets."""
    phase_size = queries_per_dataset // 2
    low = int(0.30 * phase_size)
    high = phase_size - low
    return (
        {"gsm8k": phase_size},
        {"gsm8k": high, "arc_challenge": low},
        {"gsm8k": low, "arc_challenge": high},
        {"arc_challenge": high, "math": low},
        {"arc_challenge": low, "math": high},
        {"math": phase_size},
    )


def build_proposal_s1_stream(
    datasets: dict[str, LoadedDataset],
    seed: int = 42,
    queries_per_dataset: int = _DEFAULT_QUERIES_PER_DATASET,
) -> tuple[ScenarioSpec, list[Query], list[dict]]:
    """Materialise S1 and return its spec, stream and auditable stream plan."""
    spec = proposal_s1_spec(queries_per_dataset)
    missing = [name for name in _DATASET_ORDER if name not in datasets]
    if missing:
        raise ValueError(f"proposal S1 requires datasets: {', '.join(missing)}")

    pools: dict[str, list[Query]] = {}
    positions: dict[str, int] = {}
    for offset, name in enumerate(_DATASET_ORDER, start=1):
        pool = list(datasets[name].queries)
        if len(pool) < queries_per_dataset:
            raise ValueError(
                f"proposal S1 needs {queries_per_dataset} unique {name} items; "
                f"only {len(pool)} loaded"
            )
        random.Random(seed + offset).shuffle(pool)
        pools[name] = pool
        positions[name] = 0

    stream: list[Query] = []
    plan: list[dict] = []
    t = 0
    for phase, counts in enumerate(_phase_counts(queries_per_dataset), start=1):
        labels = [name for name, count in counts.items() for _ in range(count)]
        random.Random(seed + 100 + phase).shuffle(labels)
        for name in labels:
            source = pools[name][positions[name]]
            positions[name] += 1
            t += 1
            difficulty = _DIFFICULTY_BY_DATASET[name]
            stream.append(
                Query(
                    qid=source.qid,
                    t=t,
                    dataset=source.dataset,
                    domain=source.domain,
                    prompt=source.prompt,
                    gold_answer=source.gold_answer,
                    difficulty=difficulty,
                    paraphrases=source.paraphrases,
                    solution_steps=source.solution_steps,
                    difficulty_rating=source.difficulty_rating,
                    difficulty_rating_std=source.difficulty_rating_std,
                    difficulty_quantile=source.difficulty_quantile,
                    difficulty_model_avg_acc=source.difficulty_model_avg_acc,
                )
            )
            plan.append(
                {
                    "t": t,
                    "phase": phase,
                    "dataset": name,
                    "difficulty_role": difficulty,
                    "qid": source.qid,
                }
            )

    assert len(stream) == spec.n_queries
    assert positions == {name: queries_per_dataset for name in _DATASET_ORDER}
    return spec, stream, plan
