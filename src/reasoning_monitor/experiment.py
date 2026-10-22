"""End-to-end experiment orchestration.

v0.1.3 changes
--------------
* ExperimentConfig:
  - datasets now includes MATH by default (matching proposal + default.yaml).
  - scenarios now includes S0 by default (null scenario for false-alarm baseline).
  - detector_strategy field added; propagated to MonitorConfig.
  - run_strategy_comparison flag: when True, H2 ablation runs all three
    strategies (distribution_based, proposal_mapping, adwin_all) automatically.

* effective_change_point used throughout:
  - test_change and evaluate_detection both receive effective_cp (midpoint
    for gradual scenarios, change_point for sudden/recurring).
  - Results store both transition_start and effective_t_star for the
    sensitivity analysis described in DEVIATIONS §16.

* Raw CQ/DDM separated from windowed CQ/CUSUM:
  - evaluate_detection is called once per level ("drift", "drift_raw").
  - Rows carry a "level" column so the two channels can be analysed
    independently and combined CQ performance cannot be artificially
    inflated by merging the two alarm streams.

* __version__ from package metadata (importlib.metadata).
"""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("reasoning-monitor")
except Exception:
    __version__ = "0.2.0"

from reasoning_monitor.data.controlled_trajectories import bank_queries
from reasoning_monitor.data.loaders import difficulty_proxy_summary, load_all, write_manifest
from reasoning_monitor.data.proposal_scenarios import (
    PROPOSAL_SCENARIOS,
    build_proposal_answer_order_calibration_stream,
    build_proposal_frozen_selection_stream,
    build_proposal_s1_stream,
    proposal_answer_order_calibration_spec,
    proposal_frozen_selection_spec,
    proposal_s1_spec,
)
from reasoning_monitor.data.scenarios import (
    SCENARIOS,
    build_stream,
    effective_change_point,
    resize_regimes,
)
from reasoning_monitor.detectors.assignment import DETECTOR_STRATEGIES, assignment_table
from reasoning_monitor.evaluation.metrics_eval import (
    compute_f1,
    cross_seed_alarm_order,
    evaluate_detection,
    holm_bonferroni,
    test_change,
)
from reasoning_monitor.models.clients import build_client
from reasoning_monitor.monitor import MonitorConfig, MonitorResult, ReasoningMonitor
from reasoning_monitor.reporting import figures, tables
from reasoning_monitor.types import METRIC_ORDER, DriftType, Query, ScenarioSpec

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    seed: int = 42
    seeds: list[int] = field(default_factory=list)
    """Independent stream/model seeds. Empty preserves the legacy single seed."""
    output_dir: str = "results"
    stream_design: str = "e2h_gsm8k"
    """Scenario catalogue: frozen E2H-GSM8K or accepted proposal streams."""
    datasets: list[str] = field(default_factory=lambda: [
        "gsm8k",   # E2H-GSM8K: primary dataset, split into Easy/Medium/Hard terciles
    ])
    allow_synthetic_datasets: bool = True
    """Live scientific runs set this false so download failure cannot buy calls on fixtures."""
    replication_dataset: str | None = "arc_challenge"
    """Second dataset for external replication only — same protocol, no changes.
    Set to None to skip replication."""
    run_replication: bool = False
    replication_scenarios: list[str] = field(
        default_factory=lambda: ["ST2", "ST3", "ST5"]
    )
    replication_seeds: list[int] = field(default_factory=list)
    replication_items: int = 150
    items_per_dataset: int = 300
    difficulty_scheme: str = "terciles"
    """Difficulty assignment, including official E2H tercile/extreme bands."""
    difficulty_partition: str = "all"
    """Fixed E2H broad or refined calibration/holdout partition."""
    dataset_selection_manifest: str | None = None
    """Frozen QID/UUID selection for calibration datasets such as SuperGPQA."""
    dataset_selection_split: str | None = None
    """Named split inside dataset_selection_manifest (for example pilot)."""
    controlled_trajectory_bank: str | None = None
    """Frozen exact-solver item bank for the independent RAT-v1 protocol."""
    controlled_trajectory_split: str | None = None
    """Family-isolated RAT-v1 split: calibration, development or holdout."""
    dataset_prompt_mode: str | None = None
    """Dataset-specific frozen prompt mode, used by SuperGPQA retries."""
    max_queries_per_scenario: int | None = None
    """Operational pilot cap. None runs the complete preregistered stream."""
    queries_per_regime: int | None = None
    """Pilot-only regime size; preserves staged transitions at a smaller scale."""
    proposal_queries_per_dataset: int | None = None
    """Pilot size for proposal S1; total stream length is three times this value."""
    frozen_selection_size: int | None = None
    """Item count for an explicitly ordered proposal development selection."""
    scenarios: list[str] = field(default_factory=lambda: [
        "ST1", "ST2", "ST3", "ST4", "ST5", "ST6",   # E2H-GSM8K controlled-composition streams
    ])
    window_sizes: list[int] = field(default_factory=lambda: [20])
    ablation_window_sizes: list[int] = field(default_factory=lambda: [10, 20, 30])
    run_ablation: bool = True
    run_strategy_comparison: bool = True  # H2: compare all three detector strategies
    alpha: float = 0.05
    guard: int = 5
    coherence_scorer: str = "lexical"
    stability_scorer: str = "auto"
    nli_model: str | None = None
    bertscore_model: str | None = None
    k_repeats: int = 5
    l_paraphrases: int = 3
    semantic_answer_policy: str | None = None
    """Frozen semantic answer-equivalence policy for CS/RS; None preserves history."""
    detector_strategy: str = "distribution_based"  # v0.1.3: propagated to MonitorConfig
    detector_overrides: dict[str, dict] = field(default_factory=dict)
    """Explicit per-metric detector parameters; recorded in the manifest."""
    proposal_s1_max_cq_delay: int | None = None
    """Prospective S1 CQ delay gate, measured from the effective midpoint."""
    use_rqeval: bool = True  # delegate metric scoring to published rqeval package
    publication_mode: bool = False
    fail_on_degenerate: bool = True
    """Fail after generation if a metric stayed constant; pilots may disable this."""
    resume: bool = False
    """Reuse complete query records from records_<tag>.jsonl checkpoints."""
    """Turns the MonitorConfig guards on and makes preflight failures fatal.
    Before v0.2.1 the guards existed only on MonitorConfig and nothing here set
    them, so no YAML or CLI run could ever reach them."""
    models: list[dict] = field(
        default_factory=lambda: [
            {"backend": "simulated", "name": "sim-strong",
             "competence": {"math:base": 0.88, "math:easy": 0.93, "math:hard": 0.66,
                            "science:base": 0.74, "science:easy": 0.80, "science:hard": 0.55,
                            "commonsense:base": 0.77, "commonsense:easy": 0.83, "commonsense:hard": 0.58,
                            "math_hard:base": 0.55, "math_hard:easy": 0.62, "math_hard:hard": 0.38}},
            {"backend": "simulated", "name": "sim-mid",
             "competence": {"math:base": 0.78, "math:easy": 0.86, "math:hard": 0.48,
                            "science:base": 0.60, "science:easy": 0.68, "science:hard": 0.38,
                            "commonsense:base": 0.65, "commonsense:easy": 0.72, "commonsense:hard": 0.44,
                            "math_hard:base": 0.40, "math_hard:easy": 0.48, "math_hard:hard": 0.24}},
            {"backend": "simulated", "name": "sim-small",
             "competence": {"math:base": 0.62, "math:easy": 0.72, "math:hard": 0.30,
                            "science:base": 0.48, "science:easy": 0.56, "science:hard": 0.25,
                            "commonsense:base": 0.53, "commonsense:easy": 0.60, "commonsense:hard": 0.31,
                            "math_hard:base": 0.28, "math_hard:easy": 0.34, "math_hard:hard": 0.15}},
        ]
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _make_monitor(
    model_spec: dict, W: int, cfg: ExperimentConfig, run_seed: int | None = None
) -> ReasoningMonitor:
    """Build a ReasoningMonitor with all config fields propagated."""
    client = build_client(model_spec, seed=cfg.seed if run_seed is None else run_seed)
    return ReasoningMonitor(
        client,
        MonitorConfig(
            window_size=W,
            k_repeats=cfg.k_repeats,                    # v0.2.1: propagated
            l_paraphrases=cfg.l_paraphrases,            # v0.2.1: propagated
            semantic_answer_policy=cfg.semantic_answer_policy,
            coherence_scorer=cfg.coherence_scorer,
            stability_scorer=cfg.stability_scorer,      # v0.2.1: propagated
            nli_model=cfg.nli_model,                    # v0.2.1: propagated
            bertscore_model=cfg.bertscore_model,        # v0.2.1: propagated
            detector_strategy=cfg.detector_strategy,   # v0.1.3: propagated
            detector_overrides=cfg.detector_overrides,
            use_rqeval=cfg.use_rqeval,                  # v0.1.3: propagated
            publication_mode=cfg.publication_mode,      # v0.2.1: propagated
        ),
    )


def _evaluate_scenario(
    result,
    spec,
    model_name: str,
    W: int,
    scen_name: str,
    cfg: ExperimentConfig,
    monitor: ReasoningMonitor,
) -> tuple[list[dict], list[dict]]:
    change_rows: list[dict] = []
    detection_rows: list[dict] = []
    cps = (
        list(spec.change_points) or [-1]
        if spec.regimes
        else [effective_change_point(spec)]
    )
    for transition_index, eff_cp in enumerate(cps, start=1):
        ts = spec.change_point if spec.transition_width > 0 else eff_cp
        evaluation_start = 0 if transition_index == 1 else cps[transition_index - 2]
        evaluation_end = cps[transition_index] if transition_index < len(cps) else spec.n_queries + 1
        transition = "stationary"
        if spec.regimes and eff_cp > 0:
            transition = (
                f"{spec.regimes[transition_index - 1][0]}_to_"
                f"{spec.regimes[transition_index][0]}"
            )
        elif spec.drift_type.value == "gradual":
            transition = "gradual"

        tests = [
            test_change(
                result.snapshots, eff_cp, metric, guard=cfg.guard,
                evaluation_start=evaluation_start, evaluation_end=evaluation_end,
            )
            for metric in METRIC_ORDER
        ] if eff_cp > 0 else []
        tests = [test for test in tests if test is not None]
        flags = holm_bonferroni([test.p_value for test in tests], cfg.alpha)
        for test, significant in zip(tests, flags, strict=True):
            row = asdict(test)
            row.update(
                scenario=scen_name, model=model_name, window_size=W,
                significant=bool(significant), transition=transition,
                transition_index=transition_index,
                transition_start=ts, effective_t_star=eff_cp,
            )
            change_rows.append(row)

        n_pre = sum(
            1 for snapshot in result.snapshots
            if evaluation_start <= snapshot.t < (eff_cp if eff_cp > 0 else evaluation_end)
        )
        channels = [(metric, monitor.detectors[metric], "drift") for metric in METRIC_ORDER]
        channels.extend(
            (metric, detector, "drift_raw")
            for metric, detector in monitor.raw_detectors.items()
        )
        for metric, detector, level in channels:
            detection = evaluate_detection(
                result.alarms, eff_cp, metric, detector.name, n_pre,
                level=level, transition_start=ts,
                evaluation_start=evaluation_start, evaluation_end=evaluation_end,
            )
            row = asdict(detection)
            row.update(
                scenario=scen_name, model=model_name, window_size=W,
                strategy=cfg.detector_strategy, transition=transition,
                transition_index=transition_index,
                transition_start=ts, effective_t_star=eff_cp,
            )
            detection_rows.append(row)
    return change_rows, detection_rows

    # Legacy single-transition path retained below for source-history context.
    """Return (change_rows, detection_rows) for one scenario × model × W run."""
    eff_cp = effective_change_point(spec)     # v0.1.3: midpoint for gradual scenarios
    ts     = spec.change_point                # transition start (always stored for sensitivity)

    change_rows: list[dict] = []
    detection_rows: list[dict] = []

    # ── change tests (Mann-Whitney U, Holm-Bonferroni corrected) ──────────
    tests = [
        test_change(result.snapshots, eff_cp, m, guard=cfg.guard)
        for m in METRIC_ORDER
    ]
    tests = [t for t in tests if t is not None]
    if tests and eff_cp > 0:
        flags = holm_bonferroni([t.p_value for t in tests], cfg.alpha)
        for t, sig in zip(tests, flags, strict=True):
            row = asdict(t)
            row.update(scenario=scen_name, model=model_name, window_size=W,
                       significant=bool(sig),
                       transition_start=ts, effective_t_star=eff_cp)
            change_rows.append(row)

    # ── detection evaluation ───────────────────────────────────────────────
    n_pre = sum(1 for s in result.snapshots if eff_cp < 0 or s.t < eff_cp)

    for m in METRIC_ORDER:
        det = monitor.detectors[m]

        # Primary windowed detector
        d = evaluate_detection(
            result.alarms, eff_cp, m, det.name, n_pre,
            level="drift", transition_start=ts,
        )
        row = asdict(d)
        row.update(scenario=scen_name, model=model_name, window_size=W,
                   strategy=cfg.detector_strategy,
                   transition_start=ts, effective_t_star=eff_cp)
        detection_rows.append(row)

    # Secondary raw detector (CQ only, level="drift_raw") — kept separate so
    # CQ performance cannot be inflated by merging windowed + raw alarms.
    for m, raw_det in monitor.raw_detectors.items():
        d = evaluate_detection(
            result.alarms, eff_cp, m, raw_det.name, n_pre,
            level="drift_raw", transition_start=ts,
        )
        row = asdict(d)
        row.update(scenario=scen_name, model=model_name, window_size=W,
                   strategy=cfg.detector_strategy,
                   transition_start=ts, effective_t_star=eff_cp)
        detection_rows.append(row)

    return change_rows, detection_rows


def _proposal_s1_cq_delay_gate(
    cfg: ExperimentConfig,
    detections: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the prospective calibrated-CQ delay gate for proposal S1."""
    columns = [
        "model", "seed", "window_size", "detector", "detected", "delay",
        "n_false_alarms", "max_delay", "zero_false_alarms", "delay_pass",
        "gate_pass",
    ]
    if cfg.proposal_s1_max_cq_delay is None or detections.empty:
        return pd.DataFrame(columns=columns)
    if cfg.proposal_s1_max_cq_delay < 0:
        raise ValueError("proposal_s1_max_cq_delay must be non-negative or null")

    selected = detections[
        (detections["scenario"] == "S1")
        & (detections["metric"] == "CQ")
        & (detections["level"] == "drift")
    ]
    rows = []
    for row in selected.itertuples(index=False):
        detected = bool(row.detected)
        delay = None if pd.isna(row.delay) else float(row.delay)
        zero_false_alarms = int(row.n_false_alarms) == 0
        delay_pass = detected and delay is not None and delay <= cfg.proposal_s1_max_cq_delay
        rows.append({
            "model": row.model,
            "seed": int(row.seed),
            "window_size": int(row.window_size),
            "detector": row.detector,
            "detected": detected,
            "delay": delay,
            "n_false_alarms": int(row.n_false_alarms),
            "max_delay": cfg.proposal_s1_max_cq_delay,
            "zero_false_alarms": zero_false_alarms,
            "delay_pass": delay_pass,
            "gate_pass": zero_false_alarms and delay_pass,
        })
    return pd.DataFrame(rows, columns=columns)


def _validate_scenarios(
    names: list[str], stream_design: str = "e2h_gsm8k"
) -> None:
    """Fail with a readable message instead of a bare KeyError deep in the loop.

    The v0.2.0 rename S0–S6 → ST1–ST6 left the old names in the CLI, the pilot
    script and the live config; each one died with `KeyError: 'S1'` only after
    the datasets had been loaded.
    """
    catalogues = {
        "e2h_gsm8k": SCENARIOS,
        "e2h_benchmark_gsm8k": SCENARIOS,
        "e2h_benchmark_amc": SCENARIOS,
        "supergpqa_mcq": SCENARIOS,
        "proposal_multidataset": PROPOSAL_SCENARIOS,
        "proposal_answer_order_calibration": PROPOSAL_SCENARIOS,
        "proposal_frozen_selection": PROPOSAL_SCENARIOS,
        "controlled_trajectory": {"RAT": None},
    }
    if stream_design not in catalogues:
        raise ValueError(
            f"unknown stream_design {stream_design!r}; expected one of {list(catalogues)}"
        )
    catalogue = catalogues[stream_design]
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        if stream_design in {
            "e2h_gsm8k", "e2h_benchmark_gsm8k", "e2h_benchmark_amc",
            "supergpqa_mcq",
        }:
            hint = (
                "The proposal S1 name is available only with "
                "stream_design='proposal_multidataset'."
            )
        else:
            hint = "Only proposal S1 is implemented in the present pilot gate."
        raise ValueError(
            f"unknown scenario(s) {unknown}; expected one of {list(catalogue)}. {hint}"
        )


def _configured_spec(cfg: ExperimentConfig, scenario: str) -> ScenarioSpec:
    if cfg.stream_design == "controlled_trajectory":
        queries = _controlled_trajectory_queries(cfg, scenario)
        return ScenarioSpec(
            name="RAT",
            drift_type=DriftType.NONE,
            change_point=-1,
            n_queries=len(queries),
            description=(
                "Nested exact-solver trajectories for prospective next-level "
                "CQ-failure forecasting"
            ),
        )
    if cfg.stream_design in {
        "proposal_multidataset",
        "proposal_answer_order_calibration",
        "proposal_frozen_selection",
    }:
        size = cfg.proposal_queries_per_dataset or 50
        if scenario != "S1":
            raise ValueError("only proposal S1 is implemented")
        if cfg.stream_design == "proposal_frozen_selection":
            return proposal_frozen_selection_spec(cfg.frozen_selection_size or 1)
        if cfg.stream_design == "proposal_answer_order_calibration":
            return proposal_answer_order_calibration_spec(size)
        return proposal_s1_spec(size)
    return resize_regimes(SCENARIOS[scenario], cfg.queries_per_regime)


def _controlled_trajectory_queries(
    cfg: ExperimentConfig,
    scenario: str,
) -> list[Query]:
    if scenario != "RAT":
        raise ValueError("controlled trajectories require scenario 'RAT'")
    if not cfg.controlled_trajectory_bank:
        raise ValueError("controlled trajectories require controlled_trajectory_bank")
    split = cfg.controlled_trajectory_split
    if split not in {"calibration", "development", "holdout"}:
        raise ValueError(
            "controlled_trajectory_split must be calibration, development or holdout"
        )
    bank_path = Path(cfg.controlled_trajectory_bank)
    if not bank_path.is_absolute():
        bank_path = Path.cwd() / bank_path
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("protocol_id") != "RAT-v1":
        raise ValueError("controlled trajectory bank is not a RAT-v1 bank")
    queries = bank_queries(bank, split)
    if not queries:
        raise ValueError(f"controlled trajectory split {split!r} is empty")
    return queries


def _configured_stream(
    cfg: ExperimentConfig,
    scenario: str,
    datasets: dict,
    seed: int,
) -> tuple[ScenarioSpec, list[Query], list[dict]]:
    """Build a stream without coupling proposal S1 to the frozen E2H module."""
    if cfg.stream_design == "controlled_trajectory":
        stream = _controlled_trajectory_queries(cfg, scenario)
        spec = ScenarioSpec(
            name="RAT",
            drift_type=DriftType.NONE,
            change_point=-1,
            n_queries=len(stream),
            description=(
                "Nested exact-solver trajectories for prospective next-level "
                "CQ-failure forecasting"
            ),
        )
        plan = [
            {
                "t": query.t,
                "dataset": query.dataset,
                "task_template": query.domain,
                "qid": query.qid,
                "family_id": query.family_id,
                "trajectory_level": query.trajectory_level,
                "trajectory_split": query.trajectory_split,
                "difficulty_role": query.difficulty,
            }
            for query in stream
        ]
        return spec, stream, plan
    if cfg.stream_design == "proposal_multidataset":
        size = cfg.proposal_queries_per_dataset or 50
        if scenario != "S1":
            raise ValueError("only proposal S1 is implemented")
        return build_proposal_s1_stream(datasets, seed=seed, queries_per_dataset=size)
    if cfg.stream_design == "proposal_answer_order_calibration":
        size = cfg.proposal_queries_per_dataset or 10
        if scenario != "S1":
            raise ValueError("only proposal S1 calibration is implemented")
        if not cfg.dataset_selection_manifest:
            raise ValueError("answer-order calibration requires dataset_selection_manifest")
        selection_path = Path(cfg.dataset_selection_manifest)
        if not selection_path.is_absolute():
            selection_path = Path.cwd() / selection_path
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        return build_proposal_answer_order_calibration_stream(
            datasets,
            sequence=list(selection["sequence"]),
            queries_per_dataset=size,
        )
    if cfg.stream_design == "proposal_frozen_selection":
        size = cfg.frozen_selection_size or 1
        if scenario != "S1":
            raise ValueError("only proposal S1 frozen selections are implemented")
        if not cfg.dataset_selection_manifest:
            raise ValueError("frozen selection requires dataset_selection_manifest")
        selection_path = Path(cfg.dataset_selection_manifest)
        if not selection_path.is_absolute():
            selection_path = Path.cwd() / selection_path
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        return build_proposal_frozen_selection_stream(
            datasets,
            sequence=list(selection["sequence"]),
            n_queries=size,
        )
    spec = resize_regimes(SCENARIOS[scenario], cfg.queries_per_regime)
    stream = build_stream(
        spec,
        datasets,
        seed=seed,
        base_dataset=cfg.datasets[0],
        difficulty_scheme=cfg.difficulty_scheme,
        difficulty_partition=cfg.difficulty_partition,
    )
    plan = [
        {
            "t": query.t,
            "phase": None,
            "dataset": query.dataset,
            "difficulty_role": query.difficulty,
            "qid": query.qid,
            "difficulty_rating": query.difficulty_rating,
            "difficulty_rating_std": query.difficulty_rating_std,
            "difficulty_quantile": query.difficulty_quantile,
            "difficulty_model_avg_acc": query.difficulty_model_avg_acc,
        }
        for query in stream
    ]
    return spec, stream, plan


def run_experiment(cfg: ExperimentConfig) -> dict:
    t_start = time.time()
    set_seeds(cfg.seed)
    _validate_scenarios(cfg.scenarios, cfg.stream_design)
    if "e2h_amc" in cfg.datasets:
        from reasoning_monitor.metrics.dimensions import require_amc_symbolic_matcher

        require_amc_symbolic_matcher()

    out = Path(cfg.output_dir)
    for sub in ("figures", "tables", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    log.info("loading datasets: %s", ", ".join(cfg.datasets))
    datasets = load_all(
        cfg.datasets,
        limit_per_dataset=cfg.items_per_dataset,
        seed=cfg.seed,
        allow_synthetic=cfg.allow_synthetic_datasets,
        selection_manifest=cfg.dataset_selection_manifest,
        selection_split=cfg.dataset_selection_split,
        prompt_mode=cfg.dataset_prompt_mode,
    )
    write_manifest(datasets, out / "data" / "dataset_manifest.json")
    difficulty_dataset = (
        cfg.datasets[0]
        if cfg.stream_design in {
            "e2h_benchmark_gsm8k", "e2h_benchmark_amc", "supergpqa_mcq"
        }
        else "gsm8k"
    )
    if (
        cfg.stream_design in {
            "e2h_gsm8k", "e2h_benchmark_gsm8k", "e2h_benchmark_amc",
            "supergpqa_mcq",
        }
        and difficulty_dataset in datasets
    ):
        proxy_summary = difficulty_proxy_summary(
            datasets[difficulty_dataset].queries,
            seed=cfg.seed,
            scheme=cfg.difficulty_scheme,
            partition=cfg.difficulty_partition,
        )
        (out / "data" / "difficulty_proxy_validation.json").write_text(
            json.dumps(proxy_summary, indent=2), encoding="utf-8"
        )
    for name, d in datasets.items():
        log.info("  %s: %d items (%s)", name, len(d.queries), d.source)

    all_changes: list[dict] = []
    all_detections: list[dict] = []
    scenario_rows: list[dict] = []
    traces: dict[str, pd.DataFrame] = {}
    preflight_report: dict = {}
    degenerate: dict[str, list[str]] = {}
    failed_calls: dict[str, int] = {}
    snapshot_query_ratio: dict[str, float] = {}
    base_results: dict[tuple[str, str, int, int], MonitorResult] = {}

    run_seeds = cfg.seeds or [cfg.seed]
    for scen_name, run_seed in (
        (name, seed) for name in cfg.scenarios for seed in run_seeds
    ):
        set_seeds(run_seed)
        spec, stream, stream_plan = _configured_stream(
            cfg, scen_name, datasets, run_seed
        )
        if cfg.max_queries_per_scenario is not None:
            if cfg.max_queries_per_scenario < 1:
                raise ValueError("max_queries_per_scenario must be positive or null")
            stream = stream[:cfg.max_queries_per_scenario]
            stream_plan = stream_plan[:cfg.max_queries_per_scenario]
        pd.DataFrame(stream_plan).to_csv(
            out / "data" / f"stream_{scen_name}_seed{run_seed}.csv", index=False
        )
        eff_cp = effective_change_point(spec)

        for model_spec in cfg.models:
            for W in cfg.window_sizes:
                tag = f"{scen_name}_{model_spec['name']}_seed{run_seed}_W{W}"
                log.info("running %s (%d queries)", tag, len(stream))

                monitor = _make_monitor(model_spec, W, cfg, run_seed=run_seed)
                if not preflight_report:
                    # Same client class and scorer config for every cell of the
                    # grid, so one check before the first call is enough — and it
                    # happens before any money is spent.
                    preflight_report = monitor.preflight(strict=cfg.publication_mode)
                records_path = out / "data" / f"records_{tag}.jsonl"
                result = monitor.run(
                    stream, checkpoint_path=records_path, resume=cfg.resume
                )
                base_results[(scen_name, model_spec["name"], run_seed, W)] = result
                failed_calls[tag] = sum(not bool(record.get("ok")) for record in result.records)
                snapshot_query_ratio[tag] = len(result.snapshots) / max(len(stream), 1)

                trace = result.to_frame()
                trace.to_csv(out / "data" / f"trace_{tag}.csv", index=False)
                result.alarms_frame().to_csv(out / "data" / f"alarms_{tag}.csv", index=False)
                # Raw generations first, interpretation second: these two files
                # are what makes the run re-scorable offline.
                _write_records(result.records, records_path)
                result.per_query.to_csv(out / "data" / f"per_query_{tag}.csv", index=False)
                traces[tag] = trace

                flat = result.degenerate_metrics()
                if flat:
                    degenerate[tag] = flat
                    log.warning(
                        "%s: metric(s) %s never varied — check the scorer, not the model",
                        tag, ", ".join(flat),
                    )
                    if cfg.publication_mode and cfg.fail_on_degenerate:
                        raise ValueError(
                            f"{tag}: {', '.join(flat)} constant across the whole run; "
                            "a degenerate metric cannot be reported as evidence of "
                            "stability (publication_mode=True)"
                        )

                c_rows, d_rows = _evaluate_scenario(
                    result, spec, model_spec["name"], W, scen_name, cfg, monitor
                )
                for row in c_rows + d_rows:
                    row["seed"] = run_seed
                all_changes.extend(c_rows)
                all_detections.extend(d_rows)

                comp = trace["composite"]
                if eff_cp > 0:
                    pre  = comp[trace["t"] < eff_cp].mean()
                    post = comp[trace["t"] > eff_cp].mean()
                else:
                    mid  = len(comp) // 2
                    pre, post = comp[:mid].mean(), comp[mid:].mean()
                scenario_rows.append({
                    "scenario": scen_name, "drift_type": spec.drift_type.value,
                    "transition_start": spec.change_point, "effective_t_star": eff_cp,
                    "n_queries": len(stream), "model": model_spec["name"],
                    "seed": run_seed,
                    "window_size": W, "q_pre": float(pre), "q_post": float(post),
                    "q_delta": float(post - pre),
                })

    changes_df    = pd.DataFrame(all_changes)
    detections_df = pd.DataFrame(all_detections)
    scenarios_df  = pd.DataFrame(scenario_rows)

    difficulty_rows = []
    if cfg.stream_design == "e2h_gsm8k":
        for model_name in [model["name"] for model in cfg.models]:
            for run_seed in run_seeds:
                for window in cfg.window_sizes:
                    keys = {
                        scenario: (scenario, model_name, run_seed, window)
                        for scenario in ("ST3", "ST4", "ST5")
                    }
                    if not all(key in base_results for key in keys.values()):
                        continue
                    means = {
                        scenario: float(
                            base_results[key].per_query["CQ_raw"].dropna().mean()
                        )
                        for scenario, key in keys.items()
                    }
                    easy_medium_gap = means["ST3"] - means["ST4"]
                    medium_hard_gap = means["ST4"] - means["ST5"]
                    difficulty_rows.append({
                        "model": model_name, "seed": run_seed, "window_size": window,
                        "cq_easy": means["ST3"], "cq_medium": means["ST4"],
                        "cq_hard": means["ST5"],
                        "easy_medium_gap": easy_medium_gap,
                        "medium_hard_gap": medium_hard_gap,
                        "monotonic": means["ST3"] > means["ST4"] > means["ST5"],
                        "passes_min_gap_0_10": (
                            easy_medium_gap >= 0.10 and medium_hard_gap >= 0.10
                        ),
                    })
    pd.DataFrame(difficulty_rows).to_csv(
        out / "data" / "difficulty_cq_validation.csv", index=False
    )

    proposal_cq_rows = []
    if cfg.stream_design == "proposal_multidataset":
        for (scenario, model_name, run_seed, window), result in base_results.items():
            frame = result.per_query[result.per_query["call_failed"] == False]  # noqa: E712
            for dataset, group in frame.groupby("dataset", sort=False):
                cq = group["CQ_raw"].dropna()
                proposal_cq_rows.append({
                    "scenario": scenario,
                    "model": model_name,
                    "seed": run_seed,
                    "window_size": window,
                    "dataset": dataset,
                    "n": int(len(cq)),
                    "n_correct": int(cq.sum()),
                    "cq": float(cq.mean()) if len(cq) else None,
                    "first_t": int(group["t"].min()),
                    "last_t": int(group["t"].max()),
                })
    pd.DataFrame(proposal_cq_rows).to_csv(
        out / "data" / "proposal_s1_cq_by_dataset.csv", index=False
    )

    changes_df.to_csv(out / "data" / "change_tests.csv", index=False)
    detections_df.to_csv(out / "data" / "detection_results.csv", index=False)
    scenarios_df.to_csv(out / "data" / "scenario_summary.csv", index=False)
    proposal_delay_gate_df = _proposal_s1_cq_delay_gate(cfg, detections_df)
    proposal_delay_gate_df.to_csv(
        out / "data" / "proposal_s1_cq_delay_gate.csv", index=False
    )

    order_summary_df, order_rows_df = cross_seed_alarm_order(
        detections_df, random_seed=cfg.seed
    )
    order_summary_df.to_csv(out / "data" / "cross_seed_alarm_order.csv", index=False)
    order_rows_df.to_csv(out / "data" / "cross_seed_alarm_ranks.csv", index=False)

    # H1: per-metric change test summary already in changes_df.
    # H2: F1 per (metric, detector, strategy) on the windowed channel.
    f1_rows = compute_f1(all_detections, strategy=cfg.detector_strategy, level="drift")
    f1_df   = pd.DataFrame(f1_rows)
    f1_df.to_csv(out / "data" / "h2_f1_main.csv", index=False)
    log.info("H2 F1 (main strategy=%s): %d rows written", cfg.detector_strategy, len(f1_rows))

    # ── H2 strategy comparison (v0.1.3) ──────────────────────────────────
    strategy_df = (
        _run_strategy_comparison(cfg, datasets, out, base_results)
        if cfg.run_strategy_comparison else pd.DataFrame()
    )

    # ── window-size ablation ──────────────────────────────────────────────
    ablation_df = _run_ablation(cfg, datasets, out) if cfg.run_ablation else pd.DataFrame()

    _generate_outputs(cfg, out, traces, changes_df, detections_df,
                      scenarios_df, ablation_df, strategy_df)

    # rqeval provenance — distribution version and internal __version__ may
    # differ (known rqeval metadata inconsistency: pip 2.0.2 vs __version__ 1.2.0).
    # Record both plus the import path for full reproducibility.
    rqeval_info = {}
    try:
        import importlib.metadata as _im
        rqeval_info["distribution_version"] = _im.version("rqeval")
    except Exception:
        rqeval_info["distribution_version"] = "unknown"
    try:
        import rqeval as _rq
        rqeval_info["internal_version"] = getattr(_rq, "__version__", "unknown")
        rqeval_info["path"] = _rq.__file__
    except Exception:
        rqeval_info["internal_version"] = "not_installed"
        rqeval_info["path"] = None
    try:
        import metrics as _m
        rqeval_info["metrics_path"] = _m.__file__
    except Exception:
        rqeval_info["metrics_path"] = None

    manifest = {
        "package_version": __version__,
        "rqeval": rqeval_info,
        "seed": cfg.seed,
        "seeds": run_seeds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "coherence_scorer": cfg.coherence_scorer,
        "stability_scorer": cfg.stability_scorer,
        "detector_strategy": cfg.detector_strategy,
        "use_rqeval": cfg.use_rqeval,
        "publication_mode": cfg.publication_mode,
        "preflight": preflight_report,
        "degenerate_metrics": degenerate,
        "failed_calls": failed_calls,
        "snapshot_query_ratio": snapshot_query_ratio,
        "proposal_s1_cq_delay_gate": {
            "max_delay": cfg.proposal_s1_max_cq_delay,
            "all_pass": (
                None if proposal_delay_gate_df.empty
                else bool(proposal_delay_gate_df["gate_pass"].all())
            ),
            "n_cells": int(len(proposal_delay_gate_df)),
        },
        "datasets": {n: {"source": d.source, "hash": d.content_hash, "n": len(d.queries)}
                     for n, d in datasets.items()},
        "models": [m["name"] for m in cfg.models],
        "backends": sorted({m.get("backend", "simulated") for m in cfg.models}),
        "stream_design": cfg.stream_design,
        "scenarios": cfg.scenarios,
        "window_sizes": cfg.window_sizes,
        "runtime_seconds": round(time.time() - t_start, 1),
        "config": asdict(cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if cfg.run_replication and cfg.replication_dataset:
        replication_values = asdict(cfg)
        replication_values.update({
            "output_dir": str(out / f"replication_{cfg.replication_dataset}"),
            "stream_design": "e2h_gsm8k",
            "datasets": [cfg.replication_dataset],
            "replication_dataset": None,
            "run_replication": False,
            "scenarios": cfg.replication_scenarios,
            "seeds": cfg.replication_seeds or run_seeds[:3],
            "items_per_dataset": cfg.replication_items,
            "run_ablation": False,
            "run_strategy_comparison": False,
            "max_queries_per_scenario": None,
            "queries_per_regime": None,
            "proposal_queries_per_dataset": None,
            "difficulty_scheme": "terciles",
            "difficulty_partition": "all",
        })
        replication_manifest = run_experiment(ExperimentConfig(**replication_values))
        manifest["replication"] = {
            "output_dir": replication_values["output_dir"],
            "dataset": cfg.replication_dataset,
            "dataset_source": replication_manifest["datasets"][cfg.replication_dataset]["source"],
            "seeds": replication_manifest["seeds"],
            "scenarios": replication_manifest["scenarios"],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("done in %.1fs → %s", manifest["runtime_seconds"], out.resolve())
    return manifest


def _write_records(records: list[dict], path: Path) -> None:
    """One JSON object per model call, in stream order."""
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _replay_detector_bank(result: MonitorResult, monitor: ReasoningMonitor) -> MonitorResult:
    """Run a fresh detector bank over saved metric values without model calls."""
    alarms = []
    for snapshot in result.snapshots:
        for metric, value in snapshot.values.items():
            alarm = monitor.detectors[metric].update(value, snapshot.t, metric)
            if alarm is not None:
                alarms.append(alarm)
    for metric, detector in monitor.raw_detectors.items():
        column = f"{metric}_raw"
        if column not in result.per_query:
            continue
        for row in result.per_query[["t", column]].itertuples(index=False):
            value = float(getattr(row, column))
            if np.isnan(value):
                continue
            alarm = detector.update(value, int(row.t), metric)
            if alarm is not None:
                alarms.append(type(alarm)(
                    t=alarm.t, metric=alarm.metric, detector=alarm.detector,
                    statistic=alarm.statistic, level="drift_raw",
                ))
    return MonitorResult(result.snapshots, alarms, result.per_query, result.records)


def _run_strategy_comparison(
    cfg: ExperimentConfig,
    datasets,
    out: Path,
    base_results: dict[tuple[str, str, int, int], MonitorResult] | None = None,
) -> pd.DataFrame:
    """H2 ablation: compare all three detector strategies across ALL scenarios × models.

    Proposal H2: "metriğe özel dedektörler, ADWIN baseline'a kıyasla daha düşük
    gecikme ve daha yüksek F1 skoru üretir."

    Design matrix: 3 strategies × (len(scenarios)) × (len(models)) × 6 metrics.
    S0 (null) contributes false-alarm data; S1–S6 contribute detection data.
    F1 is computed per (metric, detector, strategy) aggregated across scenarios.
    """
    if cfg.detector_overrides:
        raise ValueError(
            "run_strategy_comparison cannot be combined with detector_overrides; "
            "an explicit override would mask differences between strategy arms"
        )
    log.info("H2 strategy comparison: %s × %d scenarios × %d models",
             list(DETECTOR_STRATEGIES), len(cfg.scenarios), len(cfg.models))
    W = cfg.window_sizes[0]

    if base_results is not None:
        rows: list[dict] = []
        for strategy in DETECTOR_STRATEGIES:
            strategy_cfg = ExperimentConfig(**{
                **asdict(cfg), "detector_strategy": strategy,
                "run_ablation": False, "run_strategy_comparison": False,
            })
            for (scen_name, model_name, run_seed, window), result in base_results.items():
                if window != W:
                    continue
                model_spec = next(model for model in cfg.models if model["name"] == model_name)
                monitor = _make_monitor(model_spec, W, strategy_cfg, run_seed=run_seed)
                replayed = _replay_detector_bank(result, monitor)
                _, detection_rows = _evaluate_scenario(
                    replayed,
                    _configured_spec(cfg, scen_name),
                    model_name, W,
                    scen_name, strategy_cfg, monitor,
                )
                for row in detection_rows:
                    if row["level"] == "drift":
                        row["seed"] = run_seed
                        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out / "data" / "h2_strategy_comparison.csv", index=False)
        f1_rows = []
        for strategy in DETECTOR_STRATEGIES:
            f1_rows.extend(compute_f1(rows, strategy=strategy, level="drift"))
        pd.DataFrame(f1_rows).to_csv(
            out / "data" / "h2_f1_by_strategy.csv", index=False
        )
        return df

    rows: list[dict] = []
    for strategy in DETECTOR_STRATEGIES:
        # Every field that changes what is *measured* has to be carried over;
        # only the detector strategy may differ between the arms.  `use_rqeval`
        # was missing here, so a run configured with use_rqeval=False silently
        # compared three arms scored by a different implementation.
        strategy_cfg = ExperimentConfig(
            seed=cfg.seed, datasets=cfg.datasets,
            items_per_dataset=cfg.items_per_dataset,
            difficulty_scheme=cfg.difficulty_scheme,
            difficulty_partition=cfg.difficulty_partition,
            coherence_scorer=cfg.coherence_scorer,
            stability_scorer=cfg.stability_scorer,
            nli_model=cfg.nli_model,
            bertscore_model=cfg.bertscore_model,
            k_repeats=cfg.k_repeats,
            l_paraphrases=cfg.l_paraphrases,
            semantic_answer_policy=cfg.semantic_answer_policy,
            use_rqeval=cfg.use_rqeval,
            publication_mode=cfg.publication_mode,
            detector_strategy=strategy,
            run_ablation=False, run_strategy_comparison=False,
        )
        for scen_name in cfg.scenarios:
            spec, stream, _plan = _configured_stream(
                cfg, scen_name, datasets, cfg.seed
            )
            eff_cp  = effective_change_point(spec)
            ts      = spec.change_point

            for model_spec in cfg.models:
                monitor = _make_monitor(model_spec, W, strategy_cfg)
                result  = monitor.run(stream)
                _write_records(
                    result.records,
                    out / "data" / f"records_strategy-{strategy}_{scen_name}_{model_spec['name']}.jsonl",
                )
                n_pre   = sum(1 for s in result.snapshots if eff_cp < 0 or s.t < eff_cp)

                for m in METRIC_ORDER:
                    det = monitor.detectors[m]
                    d   = evaluate_detection(
                        result.alarms, eff_cp, m, det.name, n_pre,
                        level="drift", transition_start=ts,
                    )
                    rows.append({
                        "strategy": strategy, "scenario": scen_name,
                        "model": model_spec["name"], "metric": m,
                        "detector": det.name, "level": "drift",
                        "detected": d.detected,
                        "early_detected": d.early_detected,
                        "delay": d.delay,
                        "delay_from_transition_start": d.delay_from_transition_start,
                        "first_alarm_category": d.first_alarm_category,
                        "n_false_alarms": d.n_false_alarms,
                        "false_alarm_rate": d.false_alarm_rate,
                        "effective_t_star": eff_cp,
                        "transition_start": ts,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out / "data" / "h2_strategy_comparison.csv", index=False)

    # F1 per strategy — proposal H2 success criterion
    f1_rows = []
    for strategy in DETECTOR_STRATEGIES:
        f1_rows.extend(compute_f1(
            [r for r in rows if r["strategy"] == strategy],
            strategy=strategy, level="drift",
        ))
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv(out / "data" / "h2_f1_by_strategy.csv", index=False)

    log.info("H2: %d strategy×scenario×model×metric rows, F1 per strategy written", len(rows))
    return df


def _run_ablation(cfg: ExperimentConfig, datasets, out: Path) -> pd.DataFrame:
    """Window-size sensitivity on S2 (sudden domain drift)."""
    if cfg.stream_design != "e2h_gsm8k":
        raise ValueError(
            "window ablation is not yet defined for proposal streams; "
            "set run_ablation=false"
        )
    log.info("window-size ablation: W in %s", cfg.ablation_window_sizes)
    spec = resize_regimes(SCENARIOS["ST2"], cfg.queries_per_regime)
    stream = build_stream(
        spec,
        datasets,
        seed=cfg.seed,
        base_dataset=cfg.datasets[0],
        difficulty_scheme=cfg.difficulty_scheme,
    )
    model_spec = cfg.models[1] if len(cfg.models) > 1 else cfg.models[0]
    eff_cp     = effective_change_point(spec)

    rows: list[dict] = []
    for W in cfg.ablation_window_sizes:
        monitor = _make_monitor(model_spec, W, cfg)
        result  = monitor.run(stream)
        _write_records(
            result.records,
            out / "data" / f"records_ablation-W{W}_{model_spec['name']}.jsonl",
        )
        n_pre   = sum(1 for s in result.snapshots if s.t < eff_cp)
        for m in METRIC_ORDER:
            det = monitor.detectors[m]
            d   = evaluate_detection(
                result.alarms, eff_cp, m, det.name, n_pre,
                level="drift", transition_start=spec.change_point,
            )
            rows.append({
                "window_size": W, "metric": m, "detector": det.name,
                "delay": d.delay if d.delay is not None else np.nan,
                "detected": d.detected, "false_alarm_rate": d.false_alarm_rate,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "data" / "ablation_window.csv", index=False)
    return df


def _generate_outputs(cfg, out: Path, traces, changes_df, detections_df,
                      scenarios_df, ablation_df, strategy_df) -> None:
    fig_dir, tab_dir = out / "figures", out / "tables"

    # ST2 is the main composition-shift stream; prefer it, then any ST2 run,
    # then whatever exists.  The old selector looked for "S2_", a name that has
    # not existed since v0.2.0, so the headline figures were silently drawn from
    # ST1 — the stationary baseline.
    key = next((k for k in traces if k.startswith("ST2_") and "mid" in k), None) \
        or next((k for k in traces if k.startswith("ST2_")), None) \
        or next(iter(traces), None)
    if key:
        spec = _configured_spec(cfg, key.split("_")[0])
        eff_cp = effective_change_point(spec)
        figures.plot_metric_traces(
            traces[key], eff_cp, fig_dir / "fig1_metric_traces",
            title=f"Metric traces — {spec.description}",
        )
        alarms_path = out / "data" / f"alarms_{key}.csv"
        try:
            alarms = pd.read_csv(alarms_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            alarms = pd.DataFrame()
        figures.plot_composite_with_alarms(
            traces[key], alarms, eff_cp, fig_dir / "fig2_composite_alarms"
        )

    # Report only windowed detections (level="drift") for main figures/tables.
    # Raw CQ channel (level="drift_raw") is written to CSV for supplementary analysis.
    if not detections_df.empty:
        windowed = detections_df[detections_df["level"] == "drift"]
        raw_cq   = detections_df[detections_df["level"] == "drift_raw"]
        raw_cq.to_csv(out / "data" / "detection_results_raw_cq.csv", index=False)

        ranking = _ranking_from_frame(windowed)
        ranking.to_csv(out / "data" / "early_warning_ranking.csv", index=False)
        figures.plot_detection_delay(ranking, fig_dir / "fig3_detection_delay")
        tables.table_early_warning(ranking, tab_dir / "table_early_warning.tex")

        delay_matrix = windowed.pivot_table(
            index="metric", columns="scenario", values="delay", aggfunc="mean"
        ).reindex(list(METRIC_ORDER))
        figures.plot_scenario_heatmap(
            delay_matrix, fig_dir / "fig4_delay_heatmap",
            "Detection delay by metric and scenario (windowed, effective t*)", "Delay (queries)"
        )
        tables.table_detection(
            windowed[windowed["scenario"] == windowed["scenario"].iloc[0]],
            tab_dir / "table_detection.tex"
        )

    if not changes_df.empty:
        first = changes_df["scenario"].iloc[0]
        tables.table_change_tests(
            changes_df[changes_df["scenario"] == first], tab_dir / "table_change_tests.tex"
        )
        effect_matrix = changes_df.pivot_table(
            index="metric", columns="scenario", values="rank_biserial", aggfunc="mean"
        ).reindex(list(METRIC_ORDER))
        figures.plot_scenario_heatmap(
            effect_matrix, fig_dir / "fig5_effect_heatmap",
            "Effect size (rank-biserial) by metric and scenario", r"$r_{rb}$"
        )

    if not scenarios_df.empty:
        figures.plot_model_comparison(scenarios_df, fig_dir / "fig6_model_comparison")
        tables.table_scenario_summary(
            scenarios_df.groupby("scenario", as_index=False).agg(
                drift_type=("drift_type", "first"),
                transition_start=("transition_start", "first"),
                effective_t_star=("effective_t_star", "first"),
                n_queries=("n_queries", "first"),
                q_pre=("q_pre", "mean"), q_post=("q_post", "mean"),
                q_delta=("q_delta", "mean"),
            ),
            tab_dir / "table_scenarios.tex",
        )

    if not ablation_df.empty:
        agg = ablation_df.groupby(["window_size", "metric"], as_index=False)["delay"].mean()
        figures.plot_window_ablation(agg, fig_dir / "fig7_window_ablation")

    # H2 strategy comparison table
    if not strategy_df.empty:
        strategy_df.to_csv(out / "data" / "h2_strategy_comparison.csv", index=False)
        delay_by_strategy = strategy_df.pivot_table(
            index="metric", columns="strategy", values="delay", aggfunc="mean"
        )
        delay_by_strategy.to_csv(out / "data" / "h2_delay_by_strategy.csv")

    tables.table_assignment(assignment_table(cfg.detector_strategy), tab_dir / "table_assignment.tex")


def _ranking_from_frame(detections_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, grp in detections_df.groupby("metric"):
        delays = grp.loc[grp["detected"], "delay"].dropna()
        rows.append({
            "metric": metric,
            "detection_rate": float(grp["detected"].mean()),
            "mean_delay": float(delays.mean()) if len(delays) else np.nan,
            "median_delay": float(delays.median()) if len(delays) else np.nan,
            "mean_false_alarm_rate": float(grp["false_alarm_rate"].mean()),
            "n_scenarios": int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values("mean_delay", na_position="last").reset_index(drop=True)
