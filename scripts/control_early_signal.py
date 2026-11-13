"""Per-dialogue early-signal analysis on control-arm streams (EXPLORATORY).

Rule: CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md. Call-free; reads the
per-turn control metrics written by analyze_cdr_stream_control.py.

  python scripts\\control_early_signal.py [--model gpt4o_mini] [--source gpt_screen]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = ROOT / "CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md"
METRICS = ("CQ", "LS", "ES")
WINDOWS = (10, 20, 30)
BASELINE_END, Z, PERSIST = 30, 2.0, 3
CHANGE = {"S1": 34, "S2": 51, "S3": 34}
COLORS = {"CQ": "#0072B2", "LS": "#CC79A7", "ES": "#56B4E9"}
MODEL_LABELS = {"gpt4o_mini": "GPT-4o-mini", "llama3_70b": "Llama-3 70B",
                "gemma3_12b_it": "Gemma 3 12B"}


def windowed(values: pd.Series, turns: pd.Series, window: int) -> pd.Series:
    s = pd.Series(values.to_numpy(float), index=turns.to_numpy())
    return s.rolling(window, min_periods=1).mean()


def signal_turn(values: pd.Series, turns: pd.Series, window: int):
    base = values[turns <= BASELINE_END].dropna()
    if len(base) < 2:
        return None, None, None
    mu, sd = float(base.mean()), float(base.std(ddof=1))
    threshold = mu - Z * sd / np.sqrt(window)
    s = windowed(values, turns, window)
    below = [(t, v < threshold) for t, v in s.items() if t > BASELINE_END and np.isfinite(v)]
    for i in range(len(below) - PERSIST + 1):
        if all(flag for _, flag in below[i:i + PERSIST]):
            return int(below[i][0]), mu, threshold
    return None, mu, threshold


def analyze(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dialogue, scenario), part in frame.groupby(["dialogue_id", "scenario"]):
        part = part.sort_values("turn")
        for window in WINDOWS:
            row = {"dialogue_id": dialogue, "scenario": scenario, "window": window,
                   "change": CHANGE.get(scenario)}
            for metric in METRICS:
                row[f"{metric}_signal"] = signal_turn(part[metric], part["turn"], window)[0]
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ("LS", "ES"):
        out[f"{metric}_lead"] = out[f"{metric}_signal"] - out["CQ_signal"]
    return out


def classify(row, metric):
    m, c = row[f"{metric}_signal"], row["CQ_signal"]
    if pd.isna(m) and pd.isna(c):
        return "neither"
    if pd.isna(c):
        return "metric_only"
    if pd.isna(m):
        return "CQ_only"
    return "before" if m < c else ("same" if m == c else "after")


def summarize(table: pd.DataFrame) -> dict:
    drift = table[table["scenario"].isin(CHANGE)]
    null = table[~table["scenario"].isin(CHANGE)]
    out = {}
    for window, part in drift.groupby("window"):
        entry = {"n_dialogues": int(len(part)),
                 "CQ_signalled": int(part["CQ_signal"].notna().sum()),
                 "CQ_signal_minus_change_median": float(
                     (part["CQ_signal"] - part["change"]).median())
                 if part["CQ_signal"].notna().any() else None}
        for metric in ("LS", "ES"):
            classes = part.apply(lambda r, m=metric: classify(r, m), axis=1).value_counts()
            leads = part[f"{metric}_lead"].dropna()
            entry[metric] = {"classes": classes.to_dict(),
                             "median_lead": float(leads.median()) if len(leads) else None,
                             "signalled": int(part[f"{metric}_signal"].notna().sum())}
        nul = null[null["window"] == window]
        entry["null_signal_rate"] = {m: f"{int(nul[f'{m}_signal'].notna().sum())}/{len(nul)}"
                                     for m in METRICS}
        out[f"W{window}"] = entry
    return out


def plot(frame: pd.DataFrame, table: pd.DataFrame, output: Path, window: int,
         model_label: str = "GPT-4o-mini") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drift = sorted(d for d in frame["dialogue_id"].unique() if d.split("/")[0] in CHANGE)
    fig, axes = plt.subplots(6, 3, figsize=(15, 17), sharex=True, constrained_layout=True)
    for ax, dialogue in zip(axes.ravel(), drift, strict=False):
        part = frame[frame["dialogue_id"] == dialogue].sort_values("turn")
        scenario = dialogue.split("/")[0]
        row = table[(table["dialogue_id"] == dialogue) & (table["window"] == window)].iloc[0]
        for metric in METRICS:
            s = windowed(part[metric], part["turn"], window)
            ax.plot(s.index, s.values, color=COLORS[metric], lw=1.4, label=metric)
            turn, _, threshold = signal_turn(part[metric], part["turn"], window)
            if threshold is not None:
                ax.axhline(threshold, color=COLORS[metric], ls=":", lw=0.8)
            if turn is not None:
                ax.axvline(turn, color=COLORS[metric], lw=1.6, alpha=0.8)
        ax.axvline(CHANGE[scenario], color="black", ls="--", lw=1)
        ax.axvspan(0, BASELINE_END, color="#DDDDDD", alpha=0.4)
        lead_ls = row["LS_lead"]
        lead_es = row["ES_lead"]
        ax.set_title(f"{dialogue}\nCQ@{row['CQ_signal']}  LS lead {lead_ls}  ES lead {lead_es}",
                     fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{model_label} control arm, W={window} (EXPLORATORY). Grey = baseline "
                 "turns 1-30; dashed black = scheduled change; dotted = metric threshold; "
                 "solid vertical = first signal (3 turns below). Lead < 0: before CQ.",
                 fontsize=10)
    files = []
    for ext in ("png", "pdf"):
        path = output / f"early_signal_W{window}.{ext}"
        fig.savefig(path, dpi=110)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt4o_mini")
    parser.add_argument("--source", default="gpt_screen")
    args = parser.parse_args(argv)
    base = ROOT / "results/cdr_stream_v1_control_monitoring" / args.source / args.model
    frame = pd.read_csv(base / "control_turn_metrics.csv")
    output = base / "early_signal"
    output.mkdir(parents=True, exist_ok=True)
    table = analyze(frame)
    table.to_csv(output / "signal_turns.csv", index=False)
    summary = {"model": args.model, "exploratory": True,
               "addendum_sha256": hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(),
               "rule": {"baseline_turns": f"1-{BASELINE_END}", "z": Z, "persist": PERSIST},
               "by_window": summarize(table)}
    label = MODEL_LABELS.get(args.model, args.model)
    summary["figures"] = [f for w in WINDOWS for f in plot(frame, table, output, w, label)]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_window"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
