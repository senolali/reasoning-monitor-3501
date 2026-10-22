"""The online monitoring loop.

v0.1.2 changes
--------------
* K/L propagation: k_repeats and l_paraphrases from MonitorConfig are now
  forwarded to ConsistencyScore and RobustnessScore constructors.  Previously
  they were silently ignored (module-level constants shadowed the config).

* Two-level detection for CQ (raw_detection flag):
  CQ is Bernoulli per query (0 or 1).  Feeding the windowed mean to DDM
  violates DDM's Binomial assumptions.  When a metric carries raw_detection=True,
  the monitor maintains a *secondary* DDM detector that sees the raw per-query
  value alongside the primary windowed detector.  Alarms from the raw detector
  are tagged level="drift_raw" to distinguish them in the results frame.

* publication_mode guard: if True, coherence_scorer="lexical" raises ValueError
  at construction time.  This prevents lexical-fallback results from silently
  entering experiments intended for publication.

* NaN handling: RS returns NaN for C-subset exclusions; the updated
  WindowMeanMetric already skips NaN in its buffer.  The monitor additionally
  skips NaN when routing per-query values to raw detectors.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from reasoning_monitor.detectors.assignment import assign_detector
from reasoning_monitor.metrics.dimensions import build_metric
from reasoning_monitor.models.clients import SimulatedClient, make_paraphrases
from reasoning_monitor.types import (
    METRIC_ORDER,
    Alarm,
    MetricSnapshot,
    Query,
    QueryObservation,
    Sample,
)

log = logging.getLogger(__name__)

# Balanced weights (1/6 each) are the primary evaluation scenario.
# Domain-specific profiles below are ILLUSTRATIVE and NOT empirically validated
# for the named domains.  In publications, refer to them as
# "illustrative_safety_profile", "illustrative_coherence_profile" etc. rather
# than implying they have been validated for medical or legal use.
_W = 1.0 / 6.0
WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "balanced":                  {"CQ": _W,   "CS": _W,   "RS": _W,   "LS": _W,   "ES": _W,   "SS": _W},
    "illustrative_safety":       {"CQ": 0.30, "CS": 0.20, "RS": 0.30, "LS": 0.10, "ES": 0.05, "SS": 0.05},
    "illustrative_accuracy":     {"CQ": 0.40, "CS": 0.25, "RS": 0.15, "LS": 0.10, "ES": 0.05, "SS": 0.05},
    "illustrative_efficiency":   {"CQ": 0.20, "CS": 0.15, "RS": 0.15, "LS": 0.10, "ES": 0.30, "SS": 0.10},
    "illustrative_coherence":    {"CQ": 0.15, "CS": 0.25, "RS": 0.20, "LS": 0.35, "ES": 0.03, "SS": 0.02},
    "illustrative_reliability":  {"CQ": 0.30, "CS": 0.05, "RS": 0.30, "LS": 0.20, "ES": 0.03, "SS": 0.02},
    "illustrative_edge_device":  {"CQ": 0.30, "CS": 0.03, "RS": 0.10, "LS": 0.05, "ES": 0.50, "SS": 0.02},
    # Aliases kept for backward compat with v0.1.2 configs; prefer illustrative_ names
    "safety_priority":    {"CQ": 0.30, "CS": 0.20, "RS": 0.30, "LS": 0.10, "ES": 0.05, "SS": 0.05},
    "accuracy_priority":  {"CQ": 0.40, "CS": 0.25, "RS": 0.15, "LS": 0.10, "ES": 0.05, "SS": 0.05},
    "efficiency_priority":{"CQ": 0.20, "CS": 0.15, "RS": 0.15, "LS": 0.10, "ES": 0.30, "SS": 0.10},
    "medical":            {"CQ": 0.40, "CS": 0.05, "RS": 0.30, "LS": 0.20, "ES": 0.03, "SS": 0.02},
    "legal":              {"CQ": 0.15, "CS": 0.25, "RS": 0.20, "LS": 0.35, "ES": 0.03, "SS": 0.02},
    "edge_device":        {"CQ": 0.30, "CS": 0.03, "RS": 0.10, "LS": 0.05, "ES": 0.50, "SS": 0.02},
}

DEFAULT_WEIGHTS = WEIGHT_SCHEMES["balanced"]


@dataclass
class MonitorConfig:
    window_size: int = 20
    k_repeats: int = 5               # v0.1.3: K=5 per proposal experimental protocol
    l_paraphrases: int = 3
    metrics: Sequence[str] = tuple(METRIC_ORDER)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    detector_overrides: dict[str, dict] = field(default_factory=dict)
    detector_strategy: str = "distribution_based"
    """One of: 'distribution_based', 'proposal_mapping', 'adwin_all'.
    Recorded in the manifest so any run can be reproduced exactly.
    All three strategies are compared in the H2 ablation."""
    coherence_scorer: str = "lexical"
    stability_scorer: str = "auto"
    nli_model: str | None = None
    """NLI checkpoint for LS on the rqeval path.  None = rqeval's default."""
    bertscore_model: str | None = None
    """BERTScore checkpoint for SS on the rqeval path.  None = rqeval's default."""
    use_rqeval: bool = True
    """When True (default), metric scoring delegates to the published rqeval
    package (pip install rqeval).  The six scoring functions are exactly those
    used in arXiv:2605.24661 — no reimplementation.  Set False to use the
    internal implementations (for environments where rqeval is unavailable)."""
    publication_mode: bool = False
    semantic_answer_policy: str | None = None
    """Optional frozen policy path for semantic CS/RS answer equivalence."""

    def __post_init__(self) -> None:
        total = sum(self.weights[m] for m in self.metrics)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"composite weights must sum to 1.0, got {total:.4f}"
            )
        # v0.1.3: stricter publication guard — only "nli" is accepted,
        # not "auto" (which may silently fall back to lexical without a GPU).
        if self.publication_mode and self.coherence_scorer != "nli":
            raise ValueError(
                "publication_mode=True requires coherence_scorer='nli'. "
                "'auto' and 'lexical' are not accepted."
            )
        if self.publication_mode and self.stability_scorer != "bertscore":
            raise ValueError(
                "publication_mode=True requires stability_scorer='bertscore'. "
                "'auto' may silently fall back to Jaccard if bert-score is not installed."
            )
        if self.publication_mode and not self.use_rqeval:
            raise ValueError(
                "publication_mode=True requires use_rqeval=True. "
                "Internal metric implementations are not the published rqeval "
                "scoring functions; publishing with use_rqeval=False would report "
                "results from a different metric implementation than arXiv:2605.24661."
            )
        if self.semantic_answer_policy:
            from reasoning_monitor.metrics.semantic_answers import load_semantic_policy

            load_semantic_policy(self.semantic_answer_policy)
        from reasoning_monitor.detectors.assignment import DETECTOR_STRATEGIES
        if self.detector_strategy not in DETECTOR_STRATEGIES:
            raise ValueError(
                f"unknown detector_strategy {self.detector_strategy!r}; "
                f"expected one of {list(DETECTOR_STRATEGIES)}"
            )


@dataclass
class MonitorResult:
    snapshots: list[MetricSnapshot]
    alarms: list[Alarm]
    per_query: pd.DataFrame
    records: list[dict] = field(default_factory=list)
    """One entry per model call — prompt, verbatim completion, extracted answer,
    tokens, latency, success flag.  This is the only artefact from which a
    campaign can be re-scored offline; everything else in this object is an
    interpretation of it.  `run_experiment` writes it to `records_<tag>.jsonl`."""

    def records_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def degenerate_metrics(self, tol: float = 1e-12) -> list[str]:
        """Metrics whose windowed value never varied over the whole run.

        A constant metric is almost never a finding about the model: it is a
        scorer that silently fell back (LS with no NLI pipeline, SS with no
        BERTScore) or a condition that cannot vary (CS at temperature 0).  It
        must be caught before the numbers are read as evidence of stability.
        """
        if not self.snapshots:
            return []
        names = sorted({k for s in self.snapshots for k in s.values})
        flat: list[str] = []
        for name in names:
            vals = [s.values[name] for s in self.snapshots if name in s.values]
            if len(vals) >= 2 and (max(vals) - min(vals)) <= tol:
                flat.append(name)
        return flat

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.snapshots:
            row = {"t": s.t, "composite": s.composite}
            row.update(s.values)
            rows.append(row)
        return pd.DataFrame(rows)

    def alarms_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "t": a.t, "metric": a.metric, "detector": a.detector,
                    "statistic": a.statistic, "level": a.level,
                }
                for a in self.alarms
            ]
        )


class ReasoningMonitor:
    """Runs one model over one stream, producing metric traces and alarms."""

    def __init__(self, client, config: MonitorConfig | None = None):
        self.client = client
        self.config = config or MonitorConfig()
        self._build()

    def _build(self) -> None:
        cfg = self.config
        from reasoning_monitor.detectors.assignment import assign_raw_detector

        self.metrics = {}
        for name in cfg.metrics:
            if cfg.use_rqeval:
                self.metrics[name] = self._build_rqeval_metric(name, cfg)
            else:
                self.metrics[name] = self._build_internal_metric(name, cfg)

        # Primary (windowed) detectors — strategy-aware.
        self.detectors = {
            name: assign_detector(m, cfg.detector_overrides, cfg.detector_strategy)
            for name, m in self.metrics.items()
        }

        # Secondary raw detectors for Bernoulli metrics (CQ → DDM).
        self.raw_detectors = {
            name: assign_raw_detector(m)
            for name, m in self.metrics.items()
            if m.spec.raw_detection
        }

    def _build_rqeval_metric(self, name: str, cfg: MonitorConfig):
        """Build a metric using rqeval's published scoring implementation."""
        from reasoning_monitor.metrics.rqeval_adapters import build_rqeval_metric
        kwargs: dict = {}
        if name == "CS":
            kwargs["k_runs"] = cfg.k_repeats
        if name == "RS":
            kwargs["l_paraphrases"] = cfg.l_paraphrases
        if name in {"CS", "RS"}:
            kwargs["semantic_answer_policy"] = cfg.semantic_answer_policy
        if name == "SS":
            kwargs["k_runs"] = cfg.k_repeats
        if name == "LS":
            # Previously this line read
            #     nli_model = None if cfg.coherence_scorer == "lexical" else None
            # i.e. None in both branches, so `coherence_scorer` had no effect at
            # all on the rqeval path — the same shadowing bug that K/L had in
            # v0.1.1.  `nli_model=None` means "rqeval's default NLI checkpoint";
            # an explicit id overrides it.  Whether the pipeline actually loaded
            # is decided by preflight(), not by this argument.
            kwargs["nli_model"] = cfg.nli_model
        if name == "SS":
            kwargs["bertscore_model"] = cfg.bertscore_model
        return build_rqeval_metric(name, cfg.window_size, **kwargs)

    def _build_internal_metric(self, name: str, cfg: MonitorConfig):
        """Build a metric using the internal implementations (fallback)."""
        kwargs: dict = {}
        if name == "LS":
            from reasoning_monitor.metrics.coherence import build_scorer
            kwargs["scorer"] = build_scorer(cfg.coherence_scorer)
        if name == "CS":
            kwargs["k_runs"] = cfg.k_repeats
        if name == "RS":
            kwargs["l_paraphrases"] = cfg.l_paraphrases
        if name in {"CS", "RS"}:
            kwargs["semantic_answer_policy"] = cfg.semantic_answer_policy
        if name == "SS":
            from reasoning_monitor.metrics.coherence import build_trace_scorer
            kwargs["scorer"] = build_trace_scorer(cfg.stability_scorer)
            kwargs["k_runs"] = cfg.k_repeats
        return build_metric(name, cfg.window_size, **kwargs)

    # ── preflight ──────────────────────────────────────────────────────────

    def _scorer_status(self, name: str) -> dict:
        """Readiness of the scorer behind LS / SS, on either metric path."""
        metric = self.metrics.get(name)
        if metric is None:
            return {"ready": False, "scorer": "absent", "model": "none"}
        status_fn = getattr(metric, "scorer_status", None)
        if status_fn is not None:               # rqeval adapters
            return status_fn()
        scorer = getattr(metric, "scorer", None)  # internal implementations
        sname = getattr(scorer, "name", "unknown")
        heavy = {"LS": "deberta-v3-small-nli", "SS": "bertscore"}[name]
        ready = sname == heavy
        if ready:
            # The heavy scorers load lazily, so the class name alone proves
            # nothing: force the load and let a missing torch / failed download
            # report itself here rather than at query 1 of the campaign.
            try:
                scorer._ensure_loaded()
            except Exception as exc:
                return {"ready": False, "scorer": f"{sname} (load failed: {exc})",
                        "model": getattr(scorer, "model_id", sname)}
        return {"ready": ready, "scorer": sname,
                "model": getattr(scorer, "model_id", sname)}

    def preflight(self, strict: bool = False) -> dict:
        """Check, before any model call, that the run can measure what it claims.

        Three failure modes are silent at run time and fatal at analysis time:

        1. LS's NLI pipeline did not load — rqeval swallows the exception and
           scores every pair `neutral`, so LS is a constant 1.0.
        2. SS's BERTScore is absent — rqeval falls back to token Jaccard, which
           on templated traces is near-constant.
        3. The K repeats are issued at temperature 0, so they are identical by
           construction and CS/SS are 1.0 whatever the model does.

        Returns a report (recorded in the manifest).  With `strict=True` — which
        `run_experiment` passes in publication_mode — any problem raises instead.
        """
        cfg = self.config
        report: dict = {
            "scorers": {"LS": self._scorer_status("LS"), "SS": self._scorer_status("SS")},
            "problems": [],
            "warnings": [],
        }

        ls, ss = report["scorers"]["LS"], report["scorers"]["SS"]
        if "LS" in cfg.metrics and not ls["ready"]:
            msg = (f"LS scorer is {ls['scorer']!r}, not a real NLI model: LS will be "
                   f"constant and carry no signal (coherence_scorer={cfg.coherence_scorer!r})")
            (report["problems"] if cfg.coherence_scorer == "nli"
             else report["warnings"]).append(msg)
        if "SS" in cfg.metrics and not ss["ready"]:
            msg = (f"SS scorer is {ss['scorer']!r}, not BERTScore: SS will be near-constant "
                   f"(stability_scorer={cfg.stability_scorer!r})")
            (report["problems"] if cfg.stability_scorer == "bertscore"
             else report["warnings"]).append(msg)

        temperature = getattr(self.client, "temperature", None)
        if cfg.k_repeats > 1 and temperature is not None and float(temperature) == 0.0:
            report["problems"].append(
                f"k_repeats={cfg.k_repeats} at temperature=0: the repeats are identical "
                "by construction, so CS and SS are 1.0 by construction and cost "
                f"{cfg.k_repeats - 1} extra calls per query for no information. "
                "Raise the temperature for the repeats or set k_repeats=1."
            )

        for w in report["warnings"]:
            log.warning("preflight: %s", w)
        for pr in report["problems"]:
            log.error("preflight: %s", pr)
        if strict and report["problems"]:
            raise ValueError(
                "preflight failed:\n  - " + "\n  - ".join(report["problems"])
            )
        return report

    def _observe(self, q: Query) -> QueryObservation:
        cfg = self.config
        if isinstance(self.client, SimulatedClient):
            self.client.set_context(q.domain, q.difficulty)

        runs: list[Sample] = []
        for k in range(cfg.k_repeats):
            runs.append(self._generate(q.prompt, seed=k, gold=q.gold_answer))

        paraphrase_runs: list[Sample] = []
        for i, text in enumerate(make_paraphrases(q, cfg.l_paraphrases)):
            paraphrase_runs.append(
                self._generate(text, seed=100 + i, gold=q.gold_answer)
            )
        return QueryObservation(query=q, runs=runs, paraphrase_runs=paraphrase_runs)

    def _generate(self, prompt: str, seed: int, gold: str) -> Sample:
        try:
            return self.client.generate(prompt, seed=seed, gold=gold)
        except TypeError:
            return self.client.generate(prompt, seed=seed)

    @staticmethod
    def _records_for(obs: QueryObservation) -> list[dict]:
        """Flatten one observation into one record per model call.

        Written verbatim before anything interprets it, so that a parsing or
        metric fix found later can be applied offline instead of re-running the
        campaign.
        """
        q = obs.query
        out: list[dict] = []
        groups = (
            ("run", obs.runs, [q.prompt] * len(obs.runs)),
            (
                "paraphrase",
                obs.paraphrase_runs,
                make_paraphrases(q, len(obs.paraphrase_runs)),
            ),
        )
        for role, samples, prompts in groups:
            for idx, smp in enumerate(samples):
                out.append({
                    "t": q.t, "qid": q.qid, "dataset": q.dataset,
                    "domain": q.domain, "difficulty": q.difficulty,
                    "solution_steps": q.solution_steps,
                    "difficulty_rating": q.difficulty_rating,
                    "difficulty_rating_std": q.difficulty_rating_std,
                    "difficulty_quantile": q.difficulty_quantile,
                    "difficulty_model_avg_acc": q.difficulty_model_avg_acc,
                    "family_id": q.family_id,
                    "trajectory_level": q.trajectory_level,
                    "trajectory_split": q.trajectory_split,
                    "role": role, "call_index": idx,
                    "prompt": prompts[idx],
                    "gold_answer": q.gold_answer,
                    "raw_text": smp.raw_text,
                    "answer": smp.answer,
                    "reasoning_steps": list(smp.reasoning_steps),
                    "n_tokens": smp.n_tokens,
                    "latency_s": smp.latency_s,
                    "ok": smp.ok,
                    "finish_reason": smp.finish_reason,
                })
        return out

    @staticmethod
    def _write_checkpoint_record(handle, record: dict) -> None:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    def _observe_resumable(
        self,
        q: Query,
        saved_records: list[dict],
        checkpoint_handle,
    ) -> tuple[QueryObservation, list[dict]]:
        """Complete one observation, checkpointing every missing model call."""
        if isinstance(self.client, SimulatedClient):
            self.client.set_context(q.domain, q.difficulty)

        saved = {
            (record.get("role"), int(record.get("call_index", -1))): record
            for record in saved_records
            if record.get("qid") == q.qid
        }
        records: list[dict] = []
        plan = [
            ("run", index, q.prompt, index)
            for index in range(self.config.k_repeats)
        ]
        plan.extend(
            ("paraphrase", index, prompt, 100 + index)
            for index, prompt in enumerate(
                make_paraphrases(q, self.config.l_paraphrases)
            )
        )
        for role, index, prompt, seed in plan:
            record = saved.get((role, index))
            if record is None:
                sample = self._generate(prompt, seed=seed, gold=q.gold_answer)
                partial = QueryObservation(
                    q,
                    [sample] if role == "run" else [],
                    [sample] if role == "paraphrase" else [],
                )
                record = self._records_for(partial)[0]
                record["role"] = role
                record["call_index"] = index
                record["prompt"] = prompt
                self._write_checkpoint_record(checkpoint_handle, record)
            records.append(record)
        return self._observation_from_records(q, records), records

    @staticmethod
    def _observation_from_records(q: Query, records: list[dict]) -> QueryObservation:
        def sample(record: dict) -> Sample:
            return Sample(
                answer=record.get("answer", ""),
                reasoning_steps=tuple(record.get("reasoning_steps", ())),
                latency_s=float(record.get("latency_s", 0.0)),
                n_tokens=int(record.get("n_tokens", 0)),
                raw_text=record.get("raw_text", ""),
                ok=bool(record.get("ok", False)),
                finish_reason=record.get("finish_reason"),
            )

        ordered = sorted(records, key=lambda record: (record["role"], record["call_index"]))
        runs = [sample(record) for record in ordered if record["role"] == "run"]
        paraphrases = [sample(record) for record in ordered if record["role"] == "paraphrase"]
        return QueryObservation(q, runs, paraphrases)

    def _load_checkpoint(self, path: Path) -> dict[int, list[dict]]:
        grouped: dict[int, list[dict]] = {}
        if not path.exists():
            return grouped
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("ignoring truncated checkpoint line in %s", path)
                    continue
                grouped.setdefault(int(record["t"]), []).append(record)
        return grouped

    def _append_observation(
        self,
        q: Query,
        obs: QueryObservation,
        query_records: list[dict],
        snapshots: list[MetricSnapshot],
        alarms: list[Alarm],
        rows: list[dict],
        records: list[dict],
    ) -> None:
        """Score one observation and append it to an in-progress result."""
        records.extend(query_records)

        all_samples = tuple(obs.runs) + tuple(obs.paraphrase_runs)
        if not all_samples or not all(sample.ok for sample in all_samples):
            rows.append({
                "t": q.t, "qid": q.qid, "dataset": q.dataset,
                "domain": q.domain, "difficulty": q.difficulty,
                "difficulty_rating": q.difficulty_rating,
                "difficulty_rating_std": q.difficulty_rating_std,
                "difficulty_quantile": q.difficulty_quantile,
                "difficulty_model_avg_acc": q.difficulty_model_avg_acc,
                "family_id": q.family_id,
                "trajectory_level": q.trajectory_level,
                "trajectory_split": q.trajectory_split,
                **{f"{name}_raw": float("nan") for name in self.metrics},
                "latency_s": sum(sample.latency_s for sample in all_samples),
                "n_tokens": sum(sample.n_tokens for sample in all_samples),
                "call_failed": True,
            })
            return

        values: dict[str, float] = {}
        per_query: dict[str, float] = {}
        for name, metric in self.metrics.items():
            per_query_value = metric.per_query_value(obs)
            per_query[name] = per_query_value
            value = metric.update(obs)
            if value is not None and not math.isnan(value):
                values[name] = value

        rows.append(
            {
                "t": q.t, "qid": q.qid, "dataset": q.dataset,
                "domain": q.domain, "difficulty": q.difficulty,
                "difficulty_rating": q.difficulty_rating,
                "difficulty_rating_std": q.difficulty_rating_std,
                "difficulty_quantile": q.difficulty_quantile,
                "difficulty_model_avg_acc": q.difficulty_model_avg_acc,
                "family_id": q.family_id,
                "trajectory_level": q.trajectory_level,
                "trajectory_split": q.trajectory_split,
                **{f"{name}_raw": value for name, value in per_query.items()},
                "latency_s": obs.primary.latency_s,
                "n_tokens": obs.primary.n_tokens,
                "call_failed": False,
            }
        )

        if values:
            active_weight = sum(self.config.weights[name] for name in values)
            composite = sum(
                self.config.weights[name] * values[name] for name in values
            ) / active_weight
            snapshots.append(
                MetricSnapshot(t=q.t, values=dict(values), composite=composite)
            )
            for name, value in values.items():
                alarm = self.detectors[name].update(value, q.t, name)
                if alarm is not None:
                    alarms.append(alarm)

        for name, raw_detector in self.raw_detectors.items():
            per_query_value = per_query.get(name, float("nan"))
            if not math.isnan(per_query_value):
                alarm = raw_detector.update(per_query_value, q.t, name)
                if alarm is not None:
                    alarms.append(
                        Alarm(
                            t=alarm.t, metric=alarm.metric,
                            detector=alarm.detector,
                            statistic=alarm.statistic,
                            level="drift_raw",
                        )
                    )

    def run_from_records(
        self,
        stream: Sequence[Query],
        saved_records: Sequence[dict],
        progress: bool = False,
    ) -> MonitorResult:
        """Re-score saved calls in ``stream`` order without model generation.

        The stream may use all saved queries or a subset. Each selected query
        must have exactly the K original runs and L paraphrase runs declared by
        this monitor. Returned records use the new timestamp and retain the old
        one as ``source_t``; the supplied records are never mutated.
        """
        by_qid: dict[str, list[dict]] = {}
        for record in saved_records:
            by_qid.setdefault(str(record["qid"]), []).append(record)

        qids = [query.qid for query in stream]
        if len(qids) != len(set(qids)):
            raise ValueError("replay stream contains duplicate qids")

        expected = {
            *(('run', index) for index in range(self.config.k_repeats)),
            *(('paraphrase', index) for index in range(self.config.l_paraphrases)),
        }
        snapshots: list[MetricSnapshot] = []
        alarms: list[Alarm] = []
        rows: list[dict] = []
        records: list[dict] = []

        for i, query in enumerate(stream):
            source = by_qid.get(query.qid)
            if source is None:
                raise ValueError(f"no saved records for qid {query.qid!r}")
            actual = [
                (str(record.get("role")), int(record.get("call_index", -1)))
                for record in source
            ]
            if len(actual) != len(expected) or set(actual) != expected:
                raise ValueError(
                    f"saved records for {query.qid!r} do not match "
                    f"K={self.config.k_repeats}, L={self.config.l_paraphrases}: "
                    f"found {sorted(actual)}"
                )
            if any(str(record.get("dataset")) != query.dataset for record in source):
                raise ValueError(f"dataset mismatch in saved records for {query.qid!r}")

            remapped = []
            for record in source:
                item = dict(record)
                item["source_t"] = item.get("source_t", item.get("t"))
                item["t"] = query.t
                remapped.append(item)
            observation = self._observation_from_records(query, remapped)
            self._append_observation(
                query, observation, remapped, snapshots, alarms, rows, records
            )

            if progress and (i + 1) % 25 == 0:
                log.info("replayed %d/%d queries", i + 1, len(stream))

        return MonitorResult(
            snapshots=snapshots, alarms=alarms, per_query=pd.DataFrame(rows),
            records=records,
        )

    def run(
        self,
        stream: Sequence[Query],
        progress: bool = False,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> MonitorResult:
        snapshots: list[MetricSnapshot] = []
        alarms: list[Alarm] = []
        rows: list[dict] = []
        records: list[dict] = []

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        completed = self._load_checkpoint(checkpoint) if checkpoint and resume else {}
        checkpoint_handle = None
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            # Rewrite valid records once before appending. This removes a
            # truncated final JSON line left by a hard process interruption.
            checkpoint_handle = checkpoint.open("w", encoding="utf-8")
            if resume:
                for saved_rows in completed.values():
                    for record in saved_rows:
                        self._write_checkpoint_record(checkpoint_handle, record)

        for i, q in enumerate(stream):
            saved = completed.get(q.t, [])
            if checkpoint_handle is not None:
                obs, query_records = self._observe_resumable(
                    q, saved, checkpoint_handle
                )
            else:
                obs = self._observe(q)
                query_records = self._records_for(obs)
            self._append_observation(
                q, obs, query_records, snapshots, alarms, rows, records
            )

            if progress and (i + 1) % 25 == 0:
                log.info("processed %d/%d queries", i + 1, len(stream))

        if checkpoint_handle is not None:
            checkpoint_handle.close()
        return MonitorResult(
            snapshots=snapshots, alarms=alarms, per_query=pd.DataFrame(rows),
            records=records,
        )
