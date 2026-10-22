"""Frozen analysis of the CDR-Stream control-arm completion (call-free).

Protocol: CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md (hash-checked).

  H-C1   per-dialogue mean(CQ_ctrl - CQ_dialogue), turns 1-100, one-sided
         sign-flip across dialogues, Holm over Llama-3 70B and Gemma 3 12B
  H-C1b  same statistic over turns 21-100 (secondary, data-suggested)
  task validity: CQ_ctrl by scenario/regime (descriptive)
  GPT-4o-mini: same statistics, already seen, outside the Holm family

Then it runs the already-frozen monitoring validations unchanged:
analyze_cdr_stream_control.py --validate, control_early_signal.py and
control_aggregated_monitoring.py for each new model. Refuses to run until
Llama-3 70B and Gemma 3 12B each have all 3,000 control calls.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reasoning_monitor import cdr_stream_analysis as ca  # noqa: E402

PROTOCOL = ROOT / "CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md"
PROTOCOL_SHA256 = "d9362b9fab55ddcd6f4f348cc03ee41c3bcb59326ba0de267dd7d19661223175"
CONFIG = ROOT / "configs/cdr_stream_v1_main_asu.yaml"
OUTPUT = ROOT / "results/cdr_stream_v1_control_completion"
FAMILY = ("llama3_70b", "gemma3_12b_it")
SEEN = ("gpt4o_mini",)
LABELS = {"gpt4o_mini": "GPT-4o-mini", "llama3_70b": "Llama-3 70B",
          "gemma3_12b_it": "Gemma 3 12B"}
SIGN_SEED, BOOT_SEED, N_SIGN, N_BOOT = 20261011, 20261012, 20_000, 2_000
CONTROL_CALLS = 3000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_table(records) -> pd.DataFrame:
    """One row per (model, dialogue, turn) with valid CQ in each arm (NaN if invalid)."""
    rows = {}
    for r in records:
        if not r["ok"] or r["role"] != "primary":
            continue
        key = (r["model"], r["dialogue_id"], int(r["turn"]))
        row = rows.setdefault(key, {"model": key[0], "dialogue_id": key[1], "turn": key[2],
                                    "scenario": r["scenario"], "regime": r["regime"],
                                    "CQ_dialogue": np.nan, "CQ_ctrl": np.nan})
        value = np.nan if r["truncated"] else float(bool(r["correct"]))
        row["CQ_ctrl" if r["arm"] == "control" else "CQ_dialogue"] = value
    return pd.DataFrame(rows.values())


def dialogue_gaps(table: pd.DataFrame, first_turn: int) -> pd.DataFrame:
    part = table[(table["turn"] >= first_turn)].dropna(subset=["CQ_dialogue", "CQ_ctrl"])
    gap = part.assign(gap=part["CQ_ctrl"] - part["CQ_dialogue"])
    return gap.groupby(["model", "dialogue_id", "scenario"], as_index=False).agg(
        gap=("gap", "mean"), turns=("gap", "size"))


def bootstrap_ci(values: np.ndarray, rng) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def test(gaps: pd.DataFrame, models) -> dict:
    out = {}
    rng = np.random.default_rng(BOOT_SEED)
    for model in models:
        g = gaps[gaps["model"] == model]
        r = ca.stratified_sign_flip(g["gap"], g["model"], alternative="greater",
                                    n=N_SIGN, seed=SIGN_SEED)
        out[model] = {**r, "dialogues": int(len(g)), "paired_turns": int(g["turns"].sum()),
                      "ci95": bootstrap_ci(g["gap"].to_numpy(), rng) if len(g) else None,
                      "by_scenario": g.groupby("scenario")["gap"].mean().round(4).to_dict()}
    return out


def plot(table: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [0, 5, 10, 20, 40, 60, 80, 100]
    table = table.assign(segment=pd.cut(table["turn"], bins))
    models = [m for m in (*SEEN, *FAMILY) if m in set(table["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.8), sharey=True,
                             constrained_layout=True, squeeze=False)
    mids = [(a + b) / 2 for a, b in zip(bins, bins[1:], strict=False)]
    for ax, model in zip(axes[0], models, strict=True):
        part = table[table["model"] == model]
        for arm, color, label in (("CQ_dialogue", "#D55E00", "dialogue history"),
                                  ("CQ_ctrl", "#0072B2", "verified state summary")):
            means = part.groupby("segment", observed=False)[arm].mean().to_numpy()
            ax.plot(mids, means, "o-", color=color, label=label)
        n = part["dialogue_id"].nunique()
        ax.set_title(f"{LABELS[model]} (n={n} dialogues)")
        ax.set_xlabel("turn")
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("accuracy (CQ)")
    axes[0][0].legend(loc="lower left")
    files = []
    for ext in ("png", "pdf"):
        path = OUTPUT / f"dialogue_vs_state_summary.{ext}"
        fig.savefig(path, dpi=150)
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def main() -> int:
    if sha(PROTOCOL) != PROTOCOL_SHA256:
        raise SystemExit("Completion protocol differs from its frozen hash.")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records_path = ROOT / config["output_dir"] / "data/records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8")
               .splitlines() if line.strip()]
    from run_cdr_stream import fingerprint  # noqa: E402

    fp = fingerprint(config)
    records = [r for r in records if r["fingerprint"] == fp]
    counts = {m: sum(r["ok"] and r["arm"] == "control" and r["model"] == m for r in records)
              for m in FAMILY}
    if any(n < CONTROL_CALLS for n in counts.values()):
        raise SystemExit(f"Control arm incomplete: {counts}. Acquire it first.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = paired_table(records)
    table.to_csv(OUTPUT / "paired_cq.csv", index=False)
    result = {"protocol_sha256": PROTOCOL_SHA256, "records_sha256": sha(records_path),
              "control_calls": counts}
    for name, first in (("H_C1_turns_1_100", 1), ("H_C1b_turns_21_100", 21)):
        gaps = dialogue_gaps(table, first)
        family = test(gaps, FAMILY)
        adjusted = ca.holm({m: family[m]["p"] for m in FAMILY})
        for m in FAMILY:
            family[m]["p_holm"] = adjusted[m]
        result[name] = {"family": family,
                        "seen_outside_family": test(gaps, [m for m in SEEN
                                                           if m in set(gaps["model"])])}
    result["H_C1_supported"] = {m: bool(result["H_C1_turns_1_100"]["family"][m]["p_holm"] < 0.05)
                                for m in FAMILY}
    result["task_validity"] = {
        m: table[table["model"] == m].groupby(["scenario", "regime"])["CQ_ctrl"].mean()
        .round(3).rename(lambda k: "|".join(k) if isinstance(k, tuple) else k).to_dict()
        for m in (*SEEN, *FAMILY)}
    result["figures"] = plot(table)
    (OUTPUT / "assessment.json").write_text(json.dumps(result, indent=2, default=str) + "\n",
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "task_validity"}, indent=2,
                     default=str))

    # Already-frozen monitoring validations, unchanged.
    python = sys.executable
    steps = [[python, "scripts/analyze_cdr_stream_control.py", "--validate"]]
    for model in FAMILY:
        steps.append([python, "scripts/control_early_signal.py", "--model", model,
                      "--source", "validation"])
        steps.append([python, "scripts/control_aggregated_monitoring.py", "--model", model,
                      "--source", "validation"])
    for step in steps:
        print("running:", " ".join(step[1:]), flush=True)
        subprocess.run(step, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
