"""Controlled-composition stream scenarios (E2H-GSM8K design, v0.2).

Design rationale
----------------
The revised methodology answers the reviewer's central criticism: in the old
S0–S6 design, a quality drop could not be cleanly attributed to *reasoning
degradation* vs. *questions getting harder*, because drift was created by
mixing datasets of different difficulty.

The v0.2 design isolates a single, well-controlled phenomenon — difficulty
composition shift within one dataset (GSM8K, split into Easy/Medium/Hard
terciles) — and surrounds it with stationary single-difficulty controls so the
temporal transition in ST2 can be interpreted against known per-regime
baselines.

Six streams
-----------
  ST1  mixed E/M/H, fully random          → stationary baseline (all difficulties)
  ST2  Easy → Medium → Hard               → controlled composition shift (MAIN)
  ST3  Easy only, random                  → stationary Easy control
  ST4  Medium only, random                → stationary Medium control
  ST5  Hard only, random                  → stationary Hard control
  ST6  Hard → Medium → Easy               → reverse/recovery test

Within every regime, queries are drawn by RANDOM PERMUTATION of that
difficulty pool — never sorted — so within-regime order carries no difficulty
signal. Only the regime *boundaries* (ST2, ST6) are change points.

Multi-seed
----------
Each stream is regenerated under R independent seeds:
    ST_j^(1), ST_j^(2), ..., ST_j^(R)
so results never rest on one lucky permutation. The key analysis is whether
the cross-seed *alarm order* (e.g. SS → CS → CQ) is reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from reasoning_monitor.data.loaders import LoadedDataset, tag_difficulty
from reasoning_monitor.types import DriftType, Query, ScenarioSpec

# Default regime length (queries per difficulty segment).
_REGIME = 50


SCENARIOS: dict[str, ScenarioSpec] = {
    "ST1": ScenarioSpec(
        "ST1", DriftType.NONE, change_point=-1, n_queries=3 * _REGIME,
        description="Mixed E/M/H fully random — stationary baseline",
        regimes=(("mixed", 3 * _REGIME),),
    ),
    "ST2": ScenarioSpec(
        "ST2", DriftType.COMPOSITION, change_point=_REGIME, n_queries=3 * _REGIME,
        description="Easy→Medium→Hard — controlled composition shift (MAIN)",
        regimes=(("easy", _REGIME), ("medium", _REGIME), ("hard", _REGIME)),
    ),
    "ST3": ScenarioSpec(
        "ST3", DriftType.NONE, change_point=-1, n_queries=3 * _REGIME,
        description="Easy only, random — stationary Easy control",
        regimes=(("easy", 3 * _REGIME),),
    ),
    "ST4": ScenarioSpec(
        "ST4", DriftType.NONE, change_point=-1, n_queries=3 * _REGIME,
        description="Medium only, random — stationary Medium control",
        regimes=(("medium", 3 * _REGIME),),
    ),
    "ST5": ScenarioSpec(
        "ST5", DriftType.NONE, change_point=-1, n_queries=3 * _REGIME,
        description="Hard only, random — stationary Hard control",
        regimes=(("hard", 3 * _REGIME),),
    ),
    "ST6": ScenarioSpec(
        "ST6", DriftType.REVERSE, change_point=_REGIME, n_queries=3 * _REGIME,
        description="Hard→Medium→Easy — reverse / recovery test",
        regimes=(("hard", _REGIME), ("medium", _REGIME), ("easy", _REGIME)),
    ),
}


def effective_change_point(spec: ScenarioSpec) -> int:
    """Primary t* for single-change-point evaluation and figures.

    Stationary streams (ST1, ST3–ST5): -1 (no change point).
    Staged streams (ST2, ST6): the FIRST regime boundary. Both boundaries are
    available via spec.change_points for per-transition analysis.
    """
    if (
        not spec.regimes
        and spec.drift_type == DriftType.GRADUAL
        and spec.change_point > 0
        and spec.transition_width > 0
    ):
        return spec.change_point + spec.transition_width // 2
    cps = spec.change_points
    return cps[0] if cps else -1


def resize_regimes(spec: ScenarioSpec, queries_per_regime: int | None) -> ScenarioSpec:
    """Return a pilot-sized scenario while preserving its regime topology."""
    if queries_per_regime is None:
        return spec
    if queries_per_regime < 1:
        raise ValueError("queries_per_regime must be positive or null")
    if not spec.regimes:
        raise ValueError(f"scenario {spec.name} has no regimes to resize")
    regimes = tuple((difficulty, queries_per_regime) for difficulty, _ in spec.regimes)
    change_point = queries_per_regime if len(regimes) > 1 else -1
    return ScenarioSpec(
        name=spec.name,
        drift_type=spec.drift_type,
        change_point=change_point,
        n_queries=sum(count for _difficulty, count in regimes),
        transition_width=spec.transition_width,
        description=f"{spec.description} [pilot: {queries_per_regime}/regime]",
        regimes=regimes,
    )


class _Sampler:
    """Draws without replacement from a difficulty pool, reshuffling on exhaust."""

    def __init__(self, pool: list[Query], rng: random.Random) -> None:
        self._pool = list(pool)
        self._rng = rng
        self._i = 0
        self._rng.shuffle(self._pool)

    def next(self) -> Query:
        if self._i >= len(self._pool):
            self._rng.shuffle(self._pool)
            self._i = 0
        q = self._pool[self._i]
        self._i += 1
        return q


def build_stream(
    spec: ScenarioSpec,
    datasets: dict[str, LoadedDataset],
    seed: int = 42,
    base_dataset: str = "gsm8k",
    difficulty_scheme: str = "terciles",
    difficulty_partition: str = "all",
) -> list[Query]:
    """Materialise one scenario into an ordered query list with t = 1..N.

    The base dataset is split into declared Easy/Medium/Hard bands (seeded), then
    each regime is filled by random permutation of the matching pool. The
    'mixed' regime (ST1) draws uniformly across all three bands.
    """
    # Each difficulty pool gets its own generator (seed+1..seed+4) so that
    # adding or removing a regime cannot shift another regime's draw order.
    pool = list(datasets[base_dataset].queries)
    tiers = tag_difficulty(
        pool,
        seed=seed,
        scheme=difficulty_scheme,
        partition=difficulty_partition,
    )

    samplers = {
        "easy":   _Sampler(tiers["easy"], random.Random(seed + 1)),
        "medium": _Sampler(tiers["medium"], random.Random(seed + 2)),
        "hard":   _Sampler(tiers["hard"], random.Random(seed + 3)),
    }
    mixed_pool = tiers["easy"] + tiers["medium"] + tiers["hard"]
    samplers["mixed"] = _Sampler(mixed_pool, random.Random(seed + 4))

    stream: list[Query] = []
    t = 0
    for difficulty, count in spec.regimes:
        sampler = samplers[difficulty]
        for _ in range(count):
            t += 1
            q = sampler.next()
            stream.append(
                Query(qid=q.qid, t=t, dataset=q.dataset, domain=q.domain,
                      prompt=q.prompt, gold_answer=q.gold_answer,
                      difficulty=q.difficulty, paraphrases=q.paraphrases,
                      solution_steps=q.solution_steps,
                      difficulty_rating=q.difficulty_rating,
                      difficulty_rating_std=q.difficulty_rating_std,
                      difficulty_quantile=q.difficulty_quantile,
                      difficulty_model_avg_acc=q.difficulty_model_avg_acc)
            )
    return stream


def build_multiseed_streams(
    spec: ScenarioSpec,
    datasets: dict[str, LoadedDataset],
    seeds: Sequence[int],
    base_dataset: str = "gsm8k",
    difficulty_scheme: str = "terciles",
    difficulty_partition: str = "all",
) -> list[list[Query]]:
    """Generate R independent realisations of a scenario, one per seed.

    Returns [stream_seed0, stream_seed1, ...]. The cross-seed analysis
    (reproducibility of the alarm order) consumes exactly this list.
    """
    return [build_stream(
        spec,
        datasets,
        seed=s,
        base_dataset=base_dataset,
        difficulty_scheme=difficulty_scheme,
        difficulty_partition=difficulty_partition,
    )
            for s in seeds]
