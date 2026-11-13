"""LHE-RQ-v1 pilot analysis (call-free): select turn complexity k per model.

Frozen rule (LHE_RQ_EARLY_WARNING_PROTOCOL.md §7): the smallest k with median
onset tau in turns 15-45, >= 70% healthy starts and <= 50% non-declining
conversations. Writes results/lhe_rq_v1_pilot/k_selection.json and step-accuracy
trajectories (PNG and PDF) with per-turn support.

  python scripts\\analyze_lhe_pilot.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reasoning_monitor import lhe_acquisition as acq  # noqa: E402
from reasoning_monitor import lhe_early_warning as lhe  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_lhe_early_warning import fingerprint  # noqa: E402

CONFIG = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "configs/lhe_rq_v1_pilot_asu.yaml")
RULE = {"tau_median_range": (15, 45), "min_healthy": 0.70, "max_non_declining": 0.50}


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if "selection_rule" in config:        # pilot-3 onward: rule frozen in the config
        RULE.update({k: tuple(v) if isinstance(v, list) else v
                     for k, v in config["selection_rule"].items()})
    dictionary, samples = lhe.load_dataset(ROOT / config["data_dir"])
    units = acq.build_units(config["models"], samples, config["conversation_ids"],
                            config["horizon"])
    output = ROOT / config["output_dir"]
    saved, failures = acq.load_checkpoint(output / "data/records.jsonl",
                                          fingerprint(config, dictionary))
    rows_by_unit = defaultdict(list)
    for row in saved.values():
        rows_by_unit[(row["model"], row["dialogue_id"])].append(row)

    per_model: dict[str, dict] = {}
    curves: dict[str, list[list[float]]] = defaultdict(list)
    for unit in units:
        rows = rows_by_unit[(unit["model"], unit["dialogue_id"])]
        scored = lhe.score_conversation(rows, unit["conversation"], 1, 0)
        # Censoring: an interrupted conversation is analysed only up to its last
        # observed turn; unobserved turns are never counted as errors.
        observed = max((r["turn"] for r in rows if r["role"] == "primary"), default=0)
        step = [r["step_correct"] for r in scored][:observed]
        curves[unit["model"]].append(step + [math.nan] * (config["horizon"] - observed))
        entry = per_model.setdefault(unit["model"], {
            "k": unit["k"], "mode": unit.get("mode", "cot"),
            "provider_model": unit["provider_model"], "conversations": 0,
            "complete": 0, "healthy": 0, "non_declining": 0, "censored": 0, "taus": [],
            "calls": 0, "parsed": 0, "strict": 0, "truncated": 0})
        incomplete = observed < config["horizon"]
        entry["conversations"] += 1
        entry["complete"] += len(rows) == config["horizon"]
        entry["calls"] += len(rows)
        entry["parsed"] += sum(r["answer"] is not None for r in rows)
        entry["strict"] += sum(bool(r["strict"]) for r in rows)
        entry["truncated"] += sum(bool(r["truncated"]) for r in rows)
        if lhe.healthy_start(step):
            entry["healthy"] += 1
            tau = lhe.decline_onset(step)
            if tau is not None:
                entry["taus"].append(tau)
            elif incomplete:
                entry["censored"] += 1      # no decline yet, but not observed to the end
            else:
                entry["non_declining"] += 1

    for entry in per_model.values():
        n, healthy = entry["conversations"], entry["healthy"]
        entry["healthy_rate"] = healthy / n
        entry["non_declining_rate"] = entry["non_declining"] / healthy if healthy else math.nan
        entry["tau_median"] = statistics.median(entry["taus"]) if entry["taus"] else None
        entry["parse_rate"] = entry["parsed"] / max(1, entry["calls"])
        entry["strict_rate"] = entry["strict"] / max(1, entry["calls"])
        lo, hi = RULE["tau_median_range"]
        entry["passes_rule"] = bool(
            entry["censored"] == 0          # undecidable while conversations are censored
            and entry["tau_median"] is not None and lo <= entry["tau_median"] <= hi
            and entry["healthy_rate"] >= RULE["min_healthy"]
            and entry["non_declining_rate"] <= RULE["max_non_declining"])

    selection = {}
    for provider, mode in sorted({(e["provider_model"], e["mode"]) for e in per_model.values()}):
        options = sorted((e["k"], name) for name, e in per_model.items()
                         if e["provider_model"] == provider and e["mode"] == mode
                         and e["passes_rule"])
        selection[f"{provider} [{mode}]"] = (
            {"k": options[0][0], "unit": options[0][1]} if options
            else {"k": None, "reason": "no k satisfies the frozen rule"})

    report = {"protocol": "LHE-RQ-v1 pilot", "rule": RULE, "failed_attempts": failures,
              "records": len(saved), "per_model_k": per_model, "selection": selection,
              "label": "pilot for k selection; not an early-warning result"}
    (output / "k_selection.json").write_text(json.dumps(report, indent=2) + "\n",
                                             encoding="utf-8")
    plot(curves, per_model, output)
    for name, e in per_model.items():
        print(f"{name:20s} complete {e['complete']}/{e['conversations']} healthy "
              f"{e['healthy_rate']:.2f} non-declining {e['non_declining_rate']:.2f} "
              f"tau median {e['tau_median']} parse {e['parse_rate']:.3f} "
              f"trunc {e['truncated']} pass {e['passes_rule']}")
    print(json.dumps(selection, indent=2))
    return 0


def plot(curves, per_model, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    providers = sorted({e["provider_model"] for e in per_model.values()})
    fig, axes = plt.subplots(1, len(providers), figsize=(4.2 * len(providers), 3.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    for ax, provider in zip(axes, providers, strict=True):
        entries = sorted(((e["mode"], e["k"], name) for name, e in per_model.items()
                          if e["provider_model"] == provider))
        for i, (mode, k, name) in enumerate(entries):
            arr = np.array(curves[name], dtype=float)
            mean = np.nanmean(arr, axis=0)
            support = np.sum(~np.isnan(arr), axis=0)
            turns = np.arange(1, arr.shape[1] + 1)
            ax.plot(turns, mean, color=palette[i % len(palette)], lw=1.4,
                    ls="-" if mode == "cot" else "--",
                    label=f"{mode} k={k} (n={len(arr)}, min support {support.min()})")
        ax.set_title(provider, fontsize=9, loc="left")
        ax.set_xlabel("turn")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("step accuracy (mean over conversations)")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"pilot_step_accuracy.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
