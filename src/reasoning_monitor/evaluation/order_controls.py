"""Call-free temporal-order controls for the accepted-proposal S1 pilot."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reasoning_monitor.monitor import MonitorConfig, ReasoningMonitor
from reasoning_monitor.types import Query

CONTROL_ARMS = (
    "global_shuffled",
    "stationary_gsm8k",
    "stationary_arc_challenge",
    "stationary_math",
)


class _ReplayOnlyClient:
    """Sentinel client: any accidental generation makes the analysis fail."""

    def __init__(self, name: str, temperature: float | None = None) -> None:
        self.name = name
        self.temperature = temperature

    def generate(self, *_args, **_kwargs):
        raise AssertionError("order-control replay must not call a model")


def _retime(queries: list[Query]) -> list[Query]:
    return [replace(query, t=index) for index, query in enumerate(queries, start=1)]


def build_s1_order_controls(
    ordered_stream: list[Query],
    seed: int = 10042,
) -> dict[str, list[Query]]:
    """Build one matched shuffle and three stationary controls.

    Every arm reuses only queries from ``ordered_stream``. The global shuffle
    has exactly the same items and dataset composition as S1. Each stationary
    arm contains the same saved items from one dataset in a deterministic
    random order, removing the original S1 phase position.
    """
    if not ordered_stream:
        raise ValueError("ordered S1 stream is empty")
    qids = [query.qid for query in ordered_stream]
    if len(qids) != len(set(qids)):
        raise ValueError("ordered S1 stream contains duplicate qids")

    datasets = Counter(query.dataset for query in ordered_stream)
    expected_datasets = {"gsm8k", "arc_challenge", "math"}
    if set(datasets) != expected_datasets:
        raise ValueError(
            f"S1 controls require datasets {sorted(expected_datasets)}, "
            f"found {sorted(datasets)}"
        )
    if len(set(datasets.values())) != 1:
        raise ValueError(f"S1 dataset totals are not balanced: {dict(datasets)}")

    arms: dict[str, list[Query]] = {"ordered_s1": _retime(list(ordered_stream))}
    shuffled = list(ordered_stream)
    random.Random(seed).shuffle(shuffled)
    arms["global_shuffled"] = _retime(shuffled)

    for offset, dataset in enumerate(sorted(expected_datasets), start=1):
        stationary = [query for query in ordered_stream if query.dataset == dataset]
        random.Random(seed + offset).shuffle(stationary)
        arms[f"stationary_{dataset}"] = _retime(stationary)
    return arms


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _queries_from_records(records: list[dict], stream_plan: pd.DataFrame) -> list[Query]:
    metadata: dict[str, dict] = {}
    for record in records:
        qid = str(record["qid"])
        if qid not in metadata or record.get("role") == "run":
            metadata[qid] = record

    plan_qids = stream_plan.sort_values("t")["qid"].astype(str).tolist()
    missing = [qid for qid in plan_qids if qid not in metadata]
    if missing:
        raise ValueError(f"stream plan qids missing from records: {missing[:5]}")

    queries = []
    for t, qid in enumerate(plan_qids, start=1):
        record = metadata[qid]
        queries.append(
            Query(
                qid=qid,
                t=t,
                dataset=str(record["dataset"]),
                domain=str(record["domain"]),
                prompt=str(record["prompt"]),
                gold_answer=str(record["gold_answer"]),
                difficulty=str(record.get("difficulty", "base")),
                solution_steps=record.get("solution_steps"),
                difficulty_rating=record.get("difficulty_rating"),
                difficulty_rating_std=record.get("difficulty_rating_std"),
                difficulty_quantile=record.get("difficulty_quantile"),
                difficulty_model_avg_acc=record.get("difficulty_model_avg_acc"),
            )
        )
    return queries


def _cq_monitor(model: dict[str, Any], config: dict[str, Any], window: int):
    client = _ReplayOnlyClient(model["name"], model.get("temperature"))
    monitor_config = MonitorConfig(
        window_size=window,
        k_repeats=int(config["k_repeats"]),
        l_paraphrases=int(config["l_paraphrases"]),
        metrics=("CQ",),
        weights={"CQ": 1.0},
        detector_overrides=dict(config.get("detector_overrides") or {}),
        detector_strategy=str(config.get("detector_strategy", "distribution_based")),
        use_rqeval=bool(config.get("use_rqeval", True)),
        publication_mode=False,
    )
    return ReasoningMonitor(client, monitor_config)


def _first_alarm(alarms: pd.DataFrame, detector: str, level: str) -> int | None:
    if alarms.empty:
        return None
    selected = alarms[
        (alarms["metric"] == "CQ")
        & (alarms["detector"] == detector)
        & (alarms["level"] == level)
    ]
    return None if selected.empty else int(selected["t"].min())


def _alarm_count(alarms: pd.DataFrame, detector: str, level: str) -> int:
    if alarms.empty:
        return 0
    return int((
        (alarms["metric"] == "CQ")
        & (alarms["detector"] == detector)
        & (alarms["level"] == level)
    ).sum())


def _write_assessment(
    path: Path,
    summary: pd.DataFrame,
    gate: pd.DataFrame,
    overall_pass: bool,
    seed: int,
) -> None:
    lines = [
        "# S1 Temporal-Order Control Assessment",
        "",
        "This analysis reuses the exact saved model responses from the S1 pilot. ",
        "It makes zero model or network calls and evaluates CQ only, which was the ",
        "declared decision metric for the K=1/L=0 pilot.",
        "",
        f"Control seed: `{seed}`. Window size and detector strategy are inherited ",
        "from the source manifest. Fresh metric and detector state is used for every arm.",
        "",
        "## Results",
        "",
        "| Model | Arm | N | CQ | First CQ-CUSUM | CUSUM alarms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        first = "-" if pd.isna(row.first_cusum_alarm) else str(int(row.first_cusum_alarm))
        lines.append(
            f"| {row.model} | {row.arm} | {row.n_queries} | "
            f"{row.cq_mean:.3f} | {first} | {row.n_cusum_alarms} |"
        )

    lines.extend([
        "",
        "## Frozen Gate",
        "",
        "For each model, ordered S1 must reproduce its existing CQ-CUSUM alarm and ",
        "the matched global shuffle plus all three stationary arms must produce no ",
        "windowed CQ-CUSUM alarm. Raw CQ-DDM is reported descriptively and is not ",
        "part of this gate.",
        "",
        "| Model | Ordered alarm reproduced | Control false-alarm cells | Gate |",
        "| --- | --- | ---: | --- |",
    ])
    for row in gate.itertuples(index=False):
        lines.append(
            f"| {row.model} | {'yes' if row.ordered_alarm_reproduced else 'no'} | "
            f"{row.control_false_alarm_cells} | {'PASS' if row.gate_pass else 'FAIL'} |"
        )

    if overall_pass:
        decision = (
            "The strict control gate passes. This supports proceeding to a separately "
            "specified paid K/L validation; it does not make the pilot confirmatory."
        )
        interpretation = (
            "The ordered alarm is specific to the declared S1 order under this control set."
        )
    else:
        decision = (
            "The strict control gate fails. Do not proceed to the paid K/L validation "
            "until the false-alarm behavior is reassessed under a newly declared rule."
        )
        interpretation = (
            "The stationary arms were clean, but at least one matched shuffled arm alarmed. "
            "The ordered S1 alarm therefore cannot yet be attributed specifically to the "
            "designed composition trajectory rather than incidental ordering in a "
            "heterogeneous item mixture."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        decision,
        "",
        interpretation,
        "",
        "All control alarms are false alarms by construction: the stationary arms have ",
        "no composition change, while the global arm destroys temporal composition order ",
        "without changing its questions, answers, or overall dataset counts.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_s1_order_controls(
    result_dir: str | Path,
    control_seed: int = 10042,
) -> dict[str, Any]:
    """Run the preregistered CQ order controls over an existing S1 result."""
    result_dir = Path(result_dir)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    if config.get("stream_design") != "proposal_multidataset":
        raise ValueError("source run is not proposal_multidataset")
    if config.get("scenarios") != ["S1"]:
        raise ValueError("source run must contain only proposal S1")

    seeds = config.get("seeds") or [config.get("seed")]
    windows = config.get("window_sizes") or []
    if len(seeds) != 1 or len(windows) != 1:
        raise ValueError("order-control pilot requires one seed and one window size")
    seed, window = int(seeds[0]), int(windows[0])
    data_dir = result_dir / "data"
    output_dir = result_dir / "order_controls"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = config.get("models") or []
    if not models:
        raise ValueError("source manifest contains no models")
    record_paths = {
        model["name"]: data_dir / f"records_S1_{model['name']}_seed{seed}_W{window}.jsonl"
        for model in models
    }
    for path in record_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    stream_plan_path = data_dir / f"stream_S1_seed{seed}.csv"
    stream_plan = pd.read_csv(stream_plan_path)
    reference_records = _read_jsonl(record_paths[models[0]["name"]])
    ordered_stream = _queries_from_records(reference_records, stream_plan)
    arms = build_s1_order_controls(ordered_stream, seed=control_seed)
    source_t = {query.qid: query.t for query in ordered_stream}

    stream_rows = []
    for arm, stream in arms.items():
        for query in stream:
            stream_rows.append({
                "arm": arm,
                "t": query.t,
                "source_t": source_t[query.qid],
                "qid": query.qid,
                "dataset": query.dataset,
            })
    pd.DataFrame(stream_rows).to_csv(output_dir / "control_streams.csv", index=False)

    summaries = []
    ordered_parity: dict[str, bool] = {}
    record_hashes: dict[str, str] = {}
    for model in models:
        model_name = model["name"]
        records_path = record_paths[model_name]
        records = _read_jsonl(records_path)
        record_hashes[model_name] = _sha256(records_path)
        if {str(record["qid"]) for record in records} != set(source_t):
            raise ValueError(f"record qids for {model_name} do not match the S1 stream")

        original_alarm_path = data_dir / f"alarms_S1_{model_name}_seed{seed}_W{window}.csv"
        original_alarms = pd.read_csv(original_alarm_path)
        original_times = original_alarms.loc[
            (original_alarms["metric"] == "CQ")
            & (original_alarms["detector"] == "cusum")
            & (original_alarms["level"] == "drift"),
            "t",
        ].astype(int).tolist()

        for arm, stream in arms.items():
            monitor = _cq_monitor(model, config, window)
            result = monitor.run_from_records(stream, records)
            if result.per_query["call_failed"].any():
                raise ValueError(f"saved call failure encountered for {model_name}/{arm}")

            per_query = result.per_query.copy()
            per_query.insert(0, "arm", arm)
            per_query.to_csv(output_dir / f"per_query_{arm}_{model_name}.csv", index=False)
            trace = result.to_frame()
            trace.insert(0, "arm", arm)
            trace.to_csv(output_dir / f"trace_{arm}_{model_name}.csv", index=False)
            alarms = result.alarms_frame()
            if alarms.empty:
                alarms = pd.DataFrame(
                    columns=["t", "metric", "detector", "statistic", "level"]
                )
            alarms.insert(0, "arm", arm)
            alarms.to_csv(output_dir / f"alarms_{arm}_{model_name}.csv", index=False)

            cusum_times = alarms.loc[
                (alarms["metric"] == "CQ")
                & (alarms["detector"] == "cusum")
                & (alarms["level"] == "drift"),
                "t",
            ].astype(int).tolist()
            if arm == "ordered_s1":
                ordered_parity[model_name] = cusum_times == original_times

            summaries.append({
                "model": model_name,
                "arm": arm,
                "is_control": arm in CONTROL_ARMS,
                "n_queries": len(stream),
                "n_snapshots": len(result.snapshots),
                "cq_mean": float(per_query["CQ_raw"].mean()),
                "first_cusum_alarm": _first_alarm(alarms, "cusum", "drift"),
                "n_cusum_alarms": _alarm_count(alarms, "cusum", "drift"),
                "first_raw_ddm_alarm": _first_alarm(alarms, "ddm", "drift_raw"),
                "n_raw_ddm_alarms": _alarm_count(alarms, "ddm", "drift_raw"),
                "dataset_counts": json.dumps(
                    dict(sorted(Counter(query.dataset for query in stream).items()))
                ),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "control_summary.csv", index=False)
    gate_rows = []
    for model in models:
        model_name = model["name"]
        selected = summary[summary["model"] == model_name]
        false_alarm_cells = int(
            ((selected["is_control"]) & (selected["n_cusum_alarms"] > 0)).sum()
        )
        ordered_alarm = bool(
            (selected.loc[selected["arm"] == "ordered_s1", "n_cusum_alarms"] > 0).all()
        )
        parity = bool(ordered_parity.get(model_name, False))
        gate_rows.append({
            "model": model_name,
            "ordered_alarm_reproduced": ordered_alarm and parity,
            "ordered_alarm_parity": parity,
            "control_false_alarm_cells": false_alarm_cells,
            "gate_pass": ordered_alarm and parity and false_alarm_cells == 0,
        })
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(output_dir / "control_gate.csv", index=False)
    overall_pass = bool(gate["gate_pass"].all())

    control_manifest = {
        "analysis": "proposal_s1_cq_temporal_order_controls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result_dir": str(result_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_stream_sha256": _sha256(stream_plan_path),
        "source_record_sha256": record_hashes,
        "control_seed": control_seed,
        "source_seed": seed,
        "window_size": window,
        "detector_strategy": config.get("detector_strategy", "distribution_based"),
        "k_repeats": config["k_repeats"],
        "l_paraphrases": config["l_paraphrases"],
        "metric": "CQ",
        "no_model_calls": True,
        "arms": list(arms),
        "gate_rule": (
            "Each model must reproduce its ordered S1 CQ-CUSUM alarm and have "
            "zero CQ-CUSUM alarms in the matched shuffle and all stationary arms."
        ),
        "overall_gate_pass": overall_pass,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(control_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_assessment(
        output_dir / "CONTROL_ASSESSMENT.md",
        summary,
        gate,
        overall_pass,
        control_seed,
    )
    return control_manifest
