"""Exploratory, post-hoc look at single-call signals in LHE-RQ-v1 pilot-2 (call-free).

Pilot-2 collected primary replies only (K=1, L=0), so CS, SS and PA are not
measurable. This script computes the signals that one reply allows -- LS (real
NLI) and output tokens -- for the chain-of-thought k=16 conversations, aligns
them at each conversation's decline onset tau, and plots them with step
accuracy. Calibration conversations only; labelled exploratory; no threshold
is fitted or claimed.

  python scripts\\explore_lhe_pilot2_signals.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_lhe_early_warning import fingerprint  # noqa: E402

from reasoning_monitor import lhe_acquisition as acq  # noqa: E402
from reasoning_monitor import lhe_early_warning as lhe  # noqa: E402

CONFIG = ROOT / "configs/lhe_rq_v1_pilot2_asu.yaml"
UNITS = ("gemma3_12b_cot_k16", "gpt4o_mini_cot_k16")
OUT = ROOT / "results/lhe_rq_v1_pilot2/exploratory_signals"


def main() -> int:
    from reasoning_monitor.dialogue_six_metrics import StrictScorers

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dictionary, samples = lhe.load_dataset(ROOT / config["data_dir"])
    models = [m for m in config["models"] if m["name"] in UNITS]
    units = acq.build_units(models, samples, config["conversation_ids"], config["horizon"])
    saved, _ = acq.load_checkpoint(ROOT / config["output_dir"] / "data/records.jsonl",
                                   fingerprint(config, dictionary))
    rows = defaultdict(list)
    for row in saved.values():
        rows[(row["model"], row["dialogue_id"])].append(row)
    scorers = StrictScorers(with_ss=False)          # real NLI, no fallback
    OUT.mkdir(parents=True, exist_ok=True)
    convs = []
    for unit in units:
        scored = lhe.score_conversation(rows[(unit["model"], unit["dialogue_id"])],
                                        unit["conversation"], 1, 0, scorers=scorers)
        step = [r["step_correct"] for r in scored]
        convs.append({"model": unit["model"], "dialogue_id": unit["dialogue_id"],
                      "tau": lhe.decline_onset(step), "healthy": lhe.healthy_start(step),
                      "step": step, "LS": [r["LS"] for r in scored],
                      "tokens": [r["n_tokens"] for r in scored]})
        print(f"scored {unit['model']} {unit['dialogue_id']}", flush=True)
    (OUT / "signals.json").write_text(json.dumps(convs, default=float) + "\n", encoding="utf-8")

    summary = {}
    for model in UNITS:
        part = [c for c in convs if c["model"] == model]
        declining = [c for c in part if c["tau"] is not None]
        res = {"conversations": len(part), "declining": len(declining)}
        for sig in ("LS", "tokens"):
            pre_far, pre_near, post = [], [], []
            for c in declining:
                t = c["tau"] - 1
                vals = np.array(c[sig], dtype=float)
                far = vals[5:max(5, t - 10)]
                near = vals[max(0, t - 10):t]
                after = vals[t:t + 10]
                pre_far.append(np.nanmean(far) if far.size else math.nan)
                pre_near.append(np.nanmean(near))
                post.append(np.nanmean(after))
            res[sig] = {"mean_healthy_far": float(np.nanmean(pre_far)),
                        "mean_last10_before_tau": float(np.nanmean(pre_near)),
                        "mean_first10_after_tau": float(np.nanmean(post)),
                        "n_near_above_far": int(np.nansum(np.array(pre_near)
                                                          > np.array(pre_far))),
                        "n": len(declining)}
        summary[model] = res
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    plot(convs)
    return 0


def plot(convs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(9, 7), sharex=True, constrained_layout=True)
    rel = np.arange(-30, 21)
    for col, model in enumerate(UNITS):
        declining = [c for c in convs if c["model"] == model and c["tau"] is not None]
        for row, (sig, label) in enumerate((("step", "step accuracy"), ("LS", "LS (NLI)"),
                                            ("tokens", "output tokens"))):
            ax = axes[row, col]
            mat = np.full((len(declining), rel.size), np.nan)
            for i, c in enumerate(declining):
                for j, r in enumerate(rel):
                    t = c["tau"] - 1 + r
                    if 0 <= t < len(c[sig]):
                        mat[i, j] = c[sig][t]
                ax.plot(rel, mat[i], color="#bbbbbb", lw=0.6)
            ax.plot(rel, np.nanmean(mat, axis=0), color="#0072B2", lw=1.8)
            ax.axvline(0, color="#D55E00", ls=":", lw=1)
            ax.set_ylabel(label, fontsize=8)
            if row == 0:
                ax.set_title(f"{model} (n={len(declining)} declining)", fontsize=9, loc="left")
        axes[-1, col].set_xlabel("turns relative to decline onset tau")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"aligned_signals.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
