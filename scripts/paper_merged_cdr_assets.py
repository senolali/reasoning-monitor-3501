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
CONTROL_COMPLETION = ROOT / "results/cdr_stream_v1_control_completion"
CONTROL_VALIDATION = ROOT / "results/cdr_stream_v1_control_monitoring/validation"
POWER = ROOT / "results/cdr_stream_v1_power"
SCI = ROOT / "results/lost_conversation_v2_science"
MAPPINGS = ROOT / "results/cdr_stream_v1_mappings_llama3.json"
REPLAY = ROOT / "results/lost_conversation_v2_stream_replay"
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
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"study1_paired.{ext}", dpi=200)
    plt.close(fig)


SCENARIO_CHANGES = {"S0-E": (), "S0-H": (), "S1": (34, 68), "S2": (51,), "S3": (34, 67)}
SCENARIO_TITLES = {"all": "(a) all scenarios (30 dialogues)",
                   "S0-E": "(b) S0-E: stationary easy", "S0-H": "(c) S0-H: stationary hard",
                   "S1": "(d) S1: staircase E→M→H", "S2": "(e) S2: sudden E→H",
                   "S3": "(f) S3: E→H→E recovery"}


def fig_study2(turns: pd.DataFrame) -> None:
    """Accuracy over turns: pooled panel plus one panel per core scenario."""
    plt = setup_plt()
    bins = list(range(0, 101, 5))
    mids = [(a + b) / 2 for a, b in zip(bins, bins[1:], strict=False)]
    turns = turns.assign(segment=pd.cut(turns["turn"], bins))
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, key in zip(axes.flat, SCENARIO_TITLES, strict=True):
        sub = turns if key == "all" else turns[turns["scenario"] == key]
        for change in SCENARIO_CHANGES.get(key, ()):
            ax.axvline(change - 0.5, color="#888888", ls=":", lw=0.9)
        for model in CDR_MODELS:
            part = sub[sub["model"] == model]
            g = part.groupby("segment", observed=False)
            ax.plot(mids, g["CQ"].mean().to_numpy(), "-", color=COLORS[model], lw=1.6)
            ax.plot(mids, g["CQ_ctrl"].mean().to_numpy(), "--", color=COLORS[model], lw=1.6)
        ax.set_title(SCENARIO_TITLES[key], fontsize=9, loc="left")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel("turn")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (CQ)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[m], lw=1.8, label=LABEL[m]) for m in CDR_MODELS]
    handles += [Line2D([], [], color="#444444", lw=1.6, ls="-", label="dialogue history"),
                Line2D([], [], color="#444444", lw=1.6, ls="--",
                       label="verified state summary")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"study2_accuracy_over_turns.{ext}", dpi=200)
    plt.close(fig)


def fig_silent(turns: pd.DataFrame, bands: pd.DataFrame) -> None:
    """Descriptive: quality metrics do not fall with accuracy (S2 GPT; S1 replay)."""
    plt = setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    colors = {"CQ": "#000000", "CS": "#E69F00", "LS": "#CC79A7", "SS": "#0072B2"}
    names = {"CQ": "accuracy (CQ)", "CS": "consistency (CS)", "LS": "coherence (LS)",
             "SS": "stability (SS)"}
    bins = list(range(0, 101, 5))
    mids = [(a + b) / 2 for a, b in zip(bins, bins[1:], strict=False)]
    gpt = turns[turns["model"] == "gpt4o_mini"].assign(segment=lambda d: pd.cut(d["turn"], bins))
    g = gpt.groupby("segment", observed=False)
    for metric in ("CQ", "CS", "LS", "SS"):
        ax1.plot(mids, g[metric].mean().to_numpy(), "-", color=colors[metric], lw=1.7,
                 label=names[metric])
    ax1.plot(mids, g["CQ_ctrl"].mean().to_numpy(), "--", color="#777777", lw=1.4,
             label="CQ, verified state summary")
    ax1.set_title("(a) Study 2, GPT-4o-mini, dialogue history", fontsize=9, loc="left")
    ax1.set_xlabel("turn")
    ax1.set_ylabel("5-turn mean")
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.33))
    rep = bands[(bands["model"] == "gpt4o_mini") & (bands["scenario"] == "sudden")
                & (bands["window"] == 20)]
    for metric in ("CQ", "LS", "SS"):
        r = rep[rep["metric"] == metric].sort_values("t")
        ax2.plot(r["t"], r["median"], "-", color=colors[metric], lw=1.7, label=names[metric])
        ax2.fill_between(r["t"], r["p10"], r["p90"], color=colors[metric], alpha=0.12, lw=0)
    ax2.axvline(48.5, color="#888888", ls=":", lw=0.9)
    ax2.set_title("(b) Study 1 replay, GPT-4o-mini, fresh → history at session 49",
                  fontsize=9, loc="left")
    ax2.set_xlabel("session index in stream")
    ax2.set_ylabel("window mean, W = 20")
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=7.5, frameon=False, loc="lower left")
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"silent_metrics.{ext}", dpi=200)
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
    paired = pd.read_csv(CONTROL_COMPLETION / "paired_cq.csv")
    completion = json.loads((CONTROL_COMPLETION / "assessment.json")
                            .read_text(encoding="utf-8"))
    phases = {"1--5": (1, 5), "6--20": (6, 20), "21--100": (21, 100)}
    rows = []
    for model in CDR_MODELS:
        part = turns[turns["model"] == model]
        cells = []
        for lo, hi in phases.values():
            seg = part[(part["turn"] >= lo) & (part["turn"] <= hi)]
            cells.append(f(seg["CQ"].mean(), 2))
        s0e = part[(part["scenario"] == "S0-E") & (part["turn"] >= 21)]["CQ"].mean()
        ctrl = paired[(paired["model"] == model) & (paired["turn"] >= 21)]["CQ_ctrl"]
        cells += [f(s0e, 2), f(ctrl.mean(), 2) if ctrl.notna().any() else "pending"]
        rows.append(f"{LABEL[model]} & " + " & ".join(cells) + " \\\\")
    (GEN / "table_cdr_collapse.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    s0e = paired[(paired["scenario"] == "S0-E") & (paired["turn"] >= 21)]
    s0e = s0e.groupby("model")[["CQ_dialogue", "CQ_ctrl"]].mean()
    m += [macro("cdrSzeroEDialRange", f"{f(s0e['CQ_dialogue'].min(), 2)}--"
                f"{f(s0e['CQ_dialogue'].max(), 2)}"),
          macro("cdrSzeroECtrlRange", f"{f(s0e['CQ_ctrl'].min(), 2)}--"
                f"{f(s0e['CQ_ctrl'].max(), 2)}")]
    gpt = turns[turns["model"] == "gpt4o_mini"]
    m += [macro("cdrGptEarly", f(gpt[gpt["turn"] <= 5]["CQ"].mean(), 2)),
          macro("cdrGptLate", f(gpt[gpt["turn"] >= 21]["CQ"].mean(), 2)),
          macro("cdrGptCtrlLate", f(gpt[gpt["turn"] >= 21]["CQ_ctrl"].mean(), 2))]

    # Frozen H-C1 control-completion family and the seen GPT estimate.
    h_c1 = completion["H_C1_turns_1_100"]
    h_c1b = completion["H_C1b_turns_21_100"]
    rows = []
    for model in CDR_MODELS:
        source = (h_c1["seen_outside_family"] if model == "gpt4o_mini"
                  else h_c1["family"])
        r = source[model]
        holm = "--" if model == "gpt4o_mini" else p(r["p_holm"])
        rows.append(f"{LABEL[model]} & {f(r['mean'])} & "
                    f"[{f(r['ci95'][0])}, {f(r['ci95'][1])}] & {holm} & "
                    f"{r['dialogues']} \\\\")
    (GEN / "table_cdr_control_completion.tex").write_text("\n".join(rows) + "\n",
                                                            encoding="utf-8")
    for prefix, model in (("Llama", "llama3_70b"), ("Gemma", "gemma3_12b_it")):
        primary = h_c1["family"][model]
        late = h_c1b["family"][model]
        m += [macro(f"cdr{prefix}ControlGap", f(primary["mean"])),
              macro(f"cdr{prefix}ControlCiLo", f(primary["ci95"][0])),
              macro(f"cdr{prefix}ControlCiHi", f(primary["ci95"][1])),
              macro(f"cdr{prefix}ControlHolmP", p(primary["p_holm"])),
              macro(f"cdr{prefix}ControlLateGap", f(late["mean"]))]

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

    validation = json.loads((CONTROL_VALIDATION / "summary.json").read_text(encoding="utf-8"))
    for prefix, model in (("Llama", "llama3_70b"), ("Gemma", "gemma3_12b_it")):
        screen = validation["results"][model]["screen"]["checks"]["distribution_based"]
        e = json.loads((CONTROL_VALIDATION / model / "early_signal/summary.json")
                       .read_text(encoding="utf-8"))["by_window"]["W20"]
        agg = pd.read_csv(CONTROL_VALIDATION / model / "aggregated/aggregated_alarms.csv")
        agg = agg[(agg["window"] == 20) & (agg["metric"] == "CQ")
                  & (agg["detector"] == "cusum")]
        m += [macro(f"val{prefix}Detect", f"{round(screen['detection_rate'] * 18)}/18"),
              macro(f"val{prefix}Delay", f(screen["median_delay"], 0)),
              macro(f"val{prefix}LsSignal", f"{e['LS']['signalled']}/18"),
              macro(f"val{prefix}LsNull", e["null_signal_rate"]["LS"]),
              macro(f"val{prefix}AggDetect",
                    f"{int((agg['category'] == 'detection').sum())}/3"),
              macro(f"val{prefix}AggNull",
                    f"{int((agg['category'] == 'null_alarm').sum())}/2")]

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
    figure_turns = turns.drop(columns=["CQ_ctrl"], errors="ignore").merge(
        paired[["model", "dialogue_id", "turn", "CQ_ctrl"]],
        on=["model", "dialogue_id", "turn"], how="left")
    fig_study2(figure_turns)
    fig_silent(figure_turns, pd.read_csv(REPLAY / "trajectory_bands.csv"))
    fig_power(grid)
    # Descriptive endpoints of Figure silent_metrics(a): GPT-4o-mini dialogue arm.
    gpt_early = gpt[gpt["turn"] <= 5]
    gpt_end = gpt[gpt["turn"] >= 81]
    for metric in ("CQ", "CS", "LS", "SS"):
        m += [macro(f"silent{metric}Early", f(gpt_early[metric].mean(), 2)),
              macro(f"silent{metric}Late", f(gpt_end[metric].mean(), 2))]
    m += [macro("silentCsSupport", int(gpt["CS"].notna().sum())),
          macro("silentRsSupport", int(gpt["RS"].notna().sum()))]
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
        SCI / "crl_incremental_posthoc/assessment.json",
        CONTROL_COMPLETION / "assessment.json", CONTROL_COMPLETION / "paired_cq.csv",
        CONTROL_VALIDATION / "summary.json", REPLAY / "trajectory_bands.csv",
        *(CONTROL_VALIDATION / model / "early_signal/summary.json"
          for model in ("llama3_70b", "gemma3_12b_it")),
        *(CONTROL_VALIDATION / model / "aggregated/aggregated_alarms.csv"
          for model in ("llama3_70b", "gemma3_12b_it"))) }
    (GEN / "cdr_assets_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"wrote {len(m)} macros, 4 tables, 3 figures to {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
