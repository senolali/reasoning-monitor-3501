"""Study 2 (CDR-Stream) and cross-study numbers, tables and figures for the merged paper.

Call-free. Writes paper/merged/generated/cdr_numbers.tex and cdr tables, and
draws the new publication figures (paired Study 1 effect, Study 2 accuracy over
turns, detector power). Study 1 assets come from
``paper_lic_rq_v2_assets.py --paper paper/merged``. Inputs are pinned by the
Stage A record-prefix hash and the frozen mappings hash; output hashes of every
input are written to the assets manifest.

  python scripts\\paper_merged_cdr_assets.py
"""

from __future__ import annotations

import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/merged"
GEN, FIG = PAPER / "generated", PAPER / "figures"
STAGE_A = ROOT / "results/cdr_stream_v1_main/analysis_stage_a"
CONTROL = ROOT / "results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini"
POWER = ROOT / "results/cdr_stream_v1_power"
SCI = ROOT / "results/lost_conversation_v2_science"
MAPPINGS = ROOT / "results/cdr_stream_v1_mappings_llama3.json"
LABEL = {"gpt4o_mini": "GPT-4o-mini", "llama3_70b": "Llama-3 70B",
         "gemma3_12b_it": "Gemma 3 12B", "llama4_scout": "Llama-4 Scout",
         "mistral_7b": "Mistral-7B"}
CDR_MODELS = ("gpt4o_mini", "llama3_70b", "gemma3_12b_it")
LIC_MODELS = ("gpt4o_mini", "llama4_scout", "mistral_7b")
TEMPLATES = {"budget_ledger": "budget", "inventory_logistics": "inventory",
             "lab_protocol": "lab", "milestone_chains": "chains"}
COLORS = {"gpt4o_mini": "#0072B2", "llama3_70b": "#D55E00", "gemma3_12b_it": "#009E73",
          "llama4_scout": "#D55E00", "mistral_7b": "#CC79A7"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(x, d=3) -> str:
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def p(x) -> str:
    return "$<$0.001" if x < 0.001 else f(x)


def macro(name, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def wilson(k, n, z=1.959964):
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def setup_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def fig_study1(conf: dict) -> None:
    plt = setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    for i, model in enumerate(LIC_MODELS):
        e = conf["primary_endpoint"][model]
        n = e["tasks"]
        for j, (key, marker, label) in enumerate(
                (("dialogue_success", "o", "dialogue history"),
                 ("fresh_success", "s", "fresh prefix (no assistant replies)"))):
            k = e[key]
            lo, hi = wilson(k, n)
            ax1.errorbar(k / n, i + (0.12 if j else -0.12), xerr=[[k / n - lo], [hi - k / n]],
                         fmt=marker, color="#D55E00" if j == 0 else "#0072B2", capsize=3,
                         label=label if i == 0 else None)
        d = e["paired_risk_difference_fresh_minus_dialogue"]
        lo, hi = e["paired_risk_difference_bootstrap_95_ci"]
        ax2.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="D", color="black", capsize=3)
        ax2.text(hi + 0.015, i, f"Holm p={p(e['holm_adjusted_p_across_models'])}",
                 va="center", fontsize=8)
    for ax in (ax1, ax2):
        ax.set_yticks(range(len(LIC_MODELS)))
        ax.set_yticklabels([LABEL[m] for m in LIC_MODELS])
        ax.invert_yaxis()
    ax1.set_xlabel("final-turn success (95% Wilson CI)")
    ax1.set_xlim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax2.axvline(0, color="#999999", lw=0.8)
    ax2.set_xlabel("paired difference, fresh minus history\n(95% task-bootstrap CI)")
    ax2.set_xlim(-0.05, 0.75)
    ax2.set_yticklabels([])
    fig.suptitle(f"Study 1: retaining assistant replies lowers final success "
                 f"(n={conf['primary_endpoint']['gpt4o_mini']['tasks']} paired tasks per model)",
                 fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"study1_paired.{ext}", dpi=200)
    plt.close(fig)


def fig_study2(turns: pd.DataFrame) -> None:
    plt = setup_plt()
    bins = [0, 5, 10, 20, 40, 60, 80, 100]
    mids = [(a + b) / 2 for a, b in zip(bins, bins[1:], strict=False)]
    turns = turns.assign(segment=pd.cut(turns["turn"], bins))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for model in CDR_MODELS:
        part = turns[turns["model"] == model]
        means = part.groupby("segment", observed=False)["CQ"].mean().to_numpy()
        ax.plot(mids, means, "o-", color=COLORS[model],
                label=f"{LABEL[model]}, dialogue history")
        ctrl = part.groupby("segment", observed=False)["CQ_ctrl"].mean()
        if ctrl.notna().any():
            ax.plot(mids, ctrl.to_numpy(), "s--", color=COLORS[model],
                    label=f"{LABEL[model]}, verified state summary")
    ax.set_xlabel("turn (binned)")
    ax.set_ylabel("accuracy (CQ)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("Study 2: accuracy over 100 turns (30 dialogues per model, all scenarios)",
                 fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"study2_accuracy_over_turns.{ext}", dpi=200)
    plt.close(fig)


def fig_power(grid: pd.DataFrame) -> None:
    plt = setup_plt()
    best = (grid[(grid["calibration"] == "n12") & (grid["window"] == 20)]
            .sort_values("detection_33", ascending=False)
            .groupby(["m", "delta", "change"], as_index=False).first())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True, constrained_layout=True)
    shades = {1: "#D55E00", 3: "#E69F00", 6: "#009E73", 18: "#0072B2"}
    for ax, change in zip(axes, (34, 51), strict=True):
        for m, color in shades.items():
            sub = best[(best["m"] == m) & (best["change"] == change)]
            ax.plot(sub["delta"], sub["detection_33"], "o-", color=color,
                    label=f"{m} dialogue{'s' if m > 1 else ''} per turn")
        ax.axhline(0.8, color="#999999", ls=":", lw=0.8)
        ax.set_title(f"change at turn {change}", fontsize=10)
        ax.set_xlabel("accuracy drop from 0.80")
    axes[0].set_ylabel("detection within 33 turns")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Simulated detector power (W=20, thresholds from 12 null streams, "
                 "FAR target 0.05)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"detector_power.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    analysis = yaml.safe_load((ROOT / "configs/cdr_stream_v1_analysis.yaml")
                              .read_text(encoding="utf-8"))
    manifest = json.loads((STAGE_A / "manifest.json").read_text(encoding="utf-8"))
    if manifest["records_prefix_sha256"] != analysis["stage_a_prefix_sha256"]:
        raise SystemExit("Stage A outputs do not belong to the pinned record prefix")
    if sha(MAPPINGS) != analysis["mappings_sha256"]:
        raise SystemExit("mappings hash mismatch")
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    m = []

    # Design: frozen per-model difficulty mappings.
    maps = json.loads(MAPPINGS.read_text(encoding="utf-8"))["models"]
    rows = []
    for model in CDR_MODELS:
        cells = [f"E{v['E']}/M{v['M']}/H{v['H']}" if (v := maps[model]["mappings"].get(t))
                 else "--" for t in TEMPLATES]
        rows.append(f"{LABEL[model]} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_cdr_mappings.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Accuracy over turns (Stage A turn metrics).
    turns = pd.read_csv(STAGE_A / "turn_metrics.csv")
    phases = {"1--5": (1, 5), "6--20": (6, 20), "21--100": (21, 100)}
    rows = []
    for model in CDR_MODELS:
        part = turns[turns["model"] == model]
        cells = []
        for lo, hi in phases.values():
            seg = part[(part["turn"] >= lo) & (part["turn"] <= hi)]
            cells.append(f(seg["CQ"].mean(), 2))
        s0e = part[(part["scenario"] == "S0-E") & (part["turn"] >= 21)]["CQ"].mean()
        ctrl = part[part["turn"] >= 21]["CQ_ctrl"]
        cells += [f(s0e, 2), f(ctrl.mean(), 2) if ctrl.notna().any() else "pending"]
        rows.append(f"{LABEL[model]} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_cdr_collapse.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    gpt = turns[turns["model"] == "gpt4o_mini"]
    m += [macro("cdrGptEarly", f(gpt[gpt["turn"] <= 5]["CQ"].mean(), 2)),
          macro("cdrGptLate", f(gpt[gpt["turn"] >= 21]["CQ"].mean(), 2)),
          macro("cdrGptCtrlLate", f(gpt[gpt["turn"] >= 21]["CQ_ctrl"].mean(), 2))]

    h1 = json.loads((STAGE_A / "h1.json").read_text(encoding="utf-8"))
    lm = h1["level_matched_context"]
    m += [macro("cdrLevelMatched", f(lm["mean"])), macro("cdrLevelMatchedP", p(lm["p"])),
          macro("cdrLevelMatchedN", lm["support"]),
          macro("cdrLevelMatchedSzeroE", f(h1["level_matched_by_scenario"]["S0-E"]["mean"]))]

    # GPT control screen and per-dialogue early signal (exploratory).
    screen = json.loads((CONTROL / "result.json").read_text(encoding="utf-8"))
    cq20 = screen["strategies"]["W20"]["distribution_based"]["CQ"]
    m += [macro("ctrlScreenDetect", f"{round(cq20['detection_rate'] * cq20['n_drift_dialogues'])}"
                f"/{cq20['n_drift_dialogues']}")]
    early = json.loads((CONTROL / "early_signal/summary.json").read_text(encoding="utf-8"))
    w20 = early["by_window"]["W20"]
    m += [macro("earlyCqSignal", f"{w20['CQ_signalled']}/{w20['n_dialogues']}"),
          macro("earlyCqNull", w20["null_signal_rate"]["CQ"]),
          macro("earlyLsSignal", f"{w20['LS']['signalled']}/{w20['n_dialogues']}"),
          macro("earlyLsNull", w20["null_signal_rate"]["LS"]),
          macro("earlyLsBefore", w20["LS"]["classes"].get("before", 0))]

    # Power (simulation).
    grid = pd.read_csv(POWER / "power_grid.csv")
    best = (grid[(grid["calibration"] == "n12") & (grid["window"] == 20)]
            .sort_values("detection_33", ascending=False)
            .groupby(["m", "delta", "change"], as_index=False).first())
    rows = []
    for mm in (1, 3, 6, 18):
        cells = []
        for change in (34, 51):
            sub = best[(best["m"] == mm) & (best["change"] == change)]
            reach = sub[sub["detection_33"] >= 0.8]["delta"]
            cells.append(f(reach.min(), 1) if len(reach) else "$>$0.6")
        d2 = best[(best["m"] == mm) & (best["delta"] == 0.2)]["detection_33"].mean()
        rows.append(f"{mm} & " + " & ".join(cells) + f" & {f(d2, 2)} \\\\")
    (GEN / "table_power.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # LIC replay direction of LS/SS (sudden, W=20).
    replay = json.loads((ROOT / "results/lost_conversation_v2_stream_replay/summary.json")
                        .read_text(encoding="utf-8"))
    h = pd.DataFrame(replay["h1"])
    h = h[(h["scenario"] == "sudden") & (h["window"] == 20)]
    ss = h[h["metric"] == "SS"].set_index("model")["median_cliffs_delta"]
    ls = h[h["metric"] == "LS"].set_index("model")["median_cliffs_delta"]
    m += [macro("replaySsDeltaGpt", f"{ss['gpt4o_mini']:+.2f}"),
          macro("replaySsDeltaLlama", f"{ss['llama4_scout']:+.2f}"),
          macro("replayLsDeltaGpt", f"{ls['gpt4o_mini']:+.2f}")]

    conf = json.loads((SCI / "confirmatory/assessment.json").read_text(encoding="utf-8"))
    fig_study1(conf)
    fig_study2(turns)
    fig_power(grid)
    crl = json.loads((SCI / "crl_incremental_posthoc/assessment.json").read_text(encoding="utf-8"))
    m += [macro("crlLastGapAuroc", f(crl["last_prefinal_gap_alone_pooled"]["auroc"])),
          macro("crlLastGapP", p(crl["last_prefinal_gap_alone_pooled"]
                                 ["task_clustered_permutation_one_sided_p"])),
          macro("crlCondAuroc", f(crl["crl_conditional_on_last_gap_pooled"]["auroc"])),
          macro("crlCondP", p(crl["crl_conditional_on_last_gap_pooled"]
                              ["task_clustered_permutation_one_sided_p"]))]
    (GEN / "cdr_numbers.tex").write_text("% Generated by scripts/paper_merged_cdr_assets.py\n"
                                         + "\n".join(m) + "\n", encoding="utf-8")
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in (
        STAGE_A / "turn_metrics.csv", STAGE_A / "h1.json", CONTROL / "result.json",
        CONTROL / "early_signal/summary.json", POWER / "power_grid.csv", MAPPINGS,
        SCI / "crl_incremental_posthoc/assessment.json")}
    (GEN / "cdr_assets_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"wrote {len(m)} macros, 3 tables, 3 figures to {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
