"""Call-free semantic-contract rescore for the completed LIC-RQ-v2 integration."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reasoning_monitor.lost_conversation_runner import METRICS

PLOT_METRICS = (*METRICS, "RQ6_mean")
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
NEED_MORE = re.compile(r"\bNEED_MORE_INFORMATION\b", re.IGNORECASE)
FINAL_MARKER = "FINAL_ANSWER:"
STATUS_MARKER = "STATUS:"


def semantic_parse(text: str) -> dict:
    """Accept narrow marker variants without inferring answers from prose.

    Only the last non-empty line is considered. Numbered prefixes, units,
    trailing explanation and the observed nested status marker are tolerated.
    Marker-free prose and placeholders remain invalid.
    """
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    final = lines[-1]
    upper = final.upper()
    marker_positions = [position for marker in (FINAL_MARKER, STATUS_MARKER)
                        if (position := upper.rfind(marker)) >= 0]
    if not marker_positions:
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    payload = final[max(marker_positions):]
    payload = payload.split(":", 1)[1].strip() if ":" in payload else ""
    if NEED_MORE.search(payload):
        return {"kind": "need_more_information", "answer": "", "contract_valid": True}
    if FINAL_MARKER not in upper:
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    numbers = NUMBER.findall(payload)
    if not numbers:
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    # Explicit equations report their result last; ordinary answer-plus-unit
    # surfaces report the answer first. This avoids treating explanatory inputs
    # such as "$64 ... (128 kg * $0.50/kg)" as the answer.
    raw = numbers[-1] if "=" in payload else numbers[0]
    raw = raw.replace(",", "")
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    if not math.isfinite(value):
        return {"kind": "invalid", "answer": "", "contract_valid": False}
    answer = str(int(value)) if value.is_integer() else f"{value:.12g}"
    return {"kind": "answer_attempt", "answer": answer, "contract_valid": True}


def semantic_records(records: Sequence[Mapping]) -> list[dict]:
    rescored = []
    for source in records:
        row = dict(source)
        parsed = semantic_parse(row["raw_text"])
        row["strict_contract_valid"] = bool(row["contract_valid"])
        row["strict_response_kind"] = row["response_kind"]
        row["strict_answer"] = row["answer"]
        row["contract_valid"] = parsed["contract_valid"]
        row["response_kind"] = parsed["kind"]
        row["answer"] = parsed["answer"]
        rescored.append(row)
    return rescored


def contract_summary(records: Sequence[Mapping]) -> dict:
    models = {}
    for model in sorted({row["model"] for row in records}):
        rows = [row for row in records if row["model"] == model]
        strict = sum(bool(row["strict_contract_valid"]) for row in rows)
        semantic = sum(bool(row["contract_valid"]) for row in rows)
        models[model] = {
            "records": len(rows),
            "strict_valid": strict,
            "strict_valid_rate": strict / len(rows),
            "semantic_valid": semantic,
            "semantic_valid_rate": semantic / len(rows),
            "recovered": sum(not row["strict_contract_valid"] and row["contract_valid"]
                             for row in rows),
            "semantic_response_kinds": dict(sorted(Counter(
                row["response_kind"] for row in rows).items())),
        }
    return models


def final_endpoint(frame: pd.DataFrame) -> dict:
    finals = frame[frame.turn == frame.total_turns]
    result = {}
    for (model, condition), group in finals.groupby(["model", "condition"]):
        measured = group[group.CQ_status == "measured"]
        result.setdefault(model, {})[condition] = {
            "tasks": len(group),
            "cq_observed": len(measured),
            "correct": int(measured.CQ.sum()),
            "final_success_rate_all_tasks": float(measured.CQ.sum() / len(group)),
            "invalid_final_answers": int((group.CQ_status == "invalid_final_answer").sum()),
        }
    return result


def metric_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for (model, condition), group in frame.groupby(["model", "condition"]):
        row = {}
        for metric in PLOT_METRICS:
            measured = group[metric].dropna()
            row[metric] = {"n": len(measured),
                           "mean": float(measured.mean()) if len(measured) else None}
        result.setdefault(model, {})[condition] = row
    return result


def readiness_gap(records: Sequence[Mapping]) -> tuple[pd.DataFrame, dict]:
    """Compare primary answer readiness under matched revealed information.

    +1 means the fresh-prefix arm attempts an answer while dialogue history is
    still explicitly unresolved. This is an exploratory integration diagnostic,
    not an RQEval metric or a retrospectively established warning rule.
    """
    primaries = [row for row in records if row["role"] == "primary"]
    by_key = {(row["model"], row["task_id"], int(row["turn"]), row["condition"]): row
              for row in primaries}
    rows = []
    for model, task_id, turn, _condition in sorted(by_key):
        if _condition != "dialogue_history":
            continue
        history = by_key[(model, task_id, turn, "dialogue_history")]
        fresh = by_key[(model, task_id, turn, "fresh_prefix")]
        values = {}
        for name, row in (("dialogue_history", history), ("fresh_prefix", fresh)):
            kind = row["response_kind"]
            values[name] = 1.0 if kind == "answer_attempt" else (
                0.0 if kind == "need_more_information" else math.nan)
        gap = (values["fresh_prefix"] - values["dialogue_history"]
               if all(math.isfinite(value) for value in values.values()) else math.nan)
        rows.append({
            "model": model, "task_id": task_id, "turn": turn,
            "total_turns": int(history["total_turns"]),
            "information_fraction": turn / int(history["total_turns"]),
            "dialogue_readiness": values["dialogue_history"],
            "fresh_readiness": values["fresh_prefix"],
            "readiness_gap": gap,
        })
    frame = pd.DataFrame(rows).drop_duplicates(["model", "task_id", "turn"])
    summary = {}
    for model, group in frame.groupby("model"):
        task_rows = []
        for task_id, task in group.groupby("task_id"):
            positive = task[(task.readiness_gap > 0) & (task.turn < task.total_turns)]
            first = int(positive.turn.min()) if len(positive) else None
            total = int(task.total_turns.iloc[0])
            final = task[task.turn == task.total_turns].iloc[0]
            task_rows.append({
                "task_id": task_id,
                "first_pre_final_positive_gap_turn": first,
                "lead_turns": total - first if first is not None else None,
                "final_gap": (float(final.readiness_gap)
                              if math.isfinite(final.readiness_gap) else None),
            })
        leads = [row["lead_turns"] for row in task_rows if row["lead_turns"] is not None]
        summary[model] = {
            "tasks": len(task_rows),
            "tasks_with_pre_final_positive_gap": len(leads),
            "median_lead_turns": float(pd.Series(leads).median()) if leads else None,
            "tasks_with_positive_final_gap": sum(row["final_gap"] == 1.0 for row in task_rows),
            "task_details": task_rows,
        }
    return frame, summary


def plot_readiness_gap(frame: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    models = list(frame.model.unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3.5 * len(models)),
                             sharex=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models, strict=True):
        data = frame[frame.model == model]
        for task_id, task in data.groupby("task_id"):
            axis.plot(task.information_fraction, task.readiness_gap, "o-", alpha=0.7,
                      label=task_id.split("/")[-1])
        axis.axhline(0, color="#444444", linewidth=1)
        axis.fill_between([0, 1], 0, 1, color="#D55E00", alpha=0.08)
        axis.set(title=model, ylabel="Fresh - dialogue readiness", ylim=(-1.1, 1.1),
                 xlim=(0, 1.03))
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Revealed information fraction")
    fig.suptitle("Exploratory resolution-readiness gap (+1 = dialogue history lags)")
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"resolution_readiness_gap.{extension}"
        fig.savefig(path, dpi=150)
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_metrics(frame: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    colors = {"dialogue_history": "#0072B2", "fresh_prefix": "#D55E00"}
    labels = {"dialogue_history": "Dialogue history", "fresh_prefix": "Fresh prefix"}
    paths = []
    for model, data in frame.groupby("model"):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True,
                                 constrained_layout=True)
        for metric, axis in zip(PLOT_METRICS, axes.flat, strict=False):
            axis.set_title(metric + (" (complete cases)" if metric == "RQ6_mean" else ""))
            notes = []
            for condition in ("dialogue_history", "fresh_prefix"):
                subset = data[data.condition == condition]
                grouped = subset.groupby("information_fraction")[metric]
                mean, count = grouped.mean(), grouped.count()
                if mean.notna().any():
                    axis.plot(mean.index, mean, "o-", color=colors[condition],
                              label=labels[condition])
                notes.append(f"{labels[condition]} n={int(count.sum())}")
            if not data[metric].notna().any():
                axis.text(0.5, 0.5, "NOT OBSERVABLE\nunder metric contract",
                          transform=axis.transAxes, ha="center", va="center", color="#666666")
            axis.text(0.02, 0.02, " | ".join(notes), transform=axis.transAxes, fontsize=7)
            axis.set(xlim=(0, 1.03), ylim=(-0.03, 1.08), xlabel="Revealed information fraction",
                     ylabel="Score")
            axis.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=2)
        fig.suptitle(f"LIC-RQ-v2 semantic-contract rescore | {model}")
        for extension in ("png", "pdf"):
            path = output / f"seven_metrics_semantic_{model}.{extension}"
            fig.savefig(path, dpi=150)
            paths.append(path.name)
        plt.close(fig)
    return paths


def write_json(path: Path, value: Mapping) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
