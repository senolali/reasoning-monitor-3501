"""Supplementary tables and figures for the merged paper (call-free).

Writes paper/merged/generated/supp_*.tex and copies supplementary figures. Every
table is read from frozen or labelled result files; hashes are recorded in
generated/supp_assets_manifest.json.

  python scripts\\paper_merged_supplement_assets.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/merged"
GEN, FIG = PAPER / "generated", PAPER / "figures"
STAGE_A = ROOT / "results/cdr_stream_v1_main/analysis_stage_a"
CTRL = ROOT / "results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini"
LABEL = {"gpt4o_mini": "GPT-4o-mini", "llama3_70b": "Llama-3 70B",
         "gemma3_12b_it": "Gemma 3 12B", "llama4_scout": "Llama-4 Scout",
         "mistral_7b": "Mistral-7B"}
USED: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    USED[str(path.relative_to(ROOT))] = sha(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))


def f(x, d=3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def p(x) -> str:
    if x is None or x != x:
        return "--"
    return "$<$0.001" if x < 0.001 else f(x)


def write(name: str, rows: list[str]) -> None:
    (GEN / f"supp_{name}.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)

    # S-Table: Stage A frozen H1a/H1b (dialogue arm; confounded by the background collapse).
    h1 = load(STAGE_A / "h1.json")
    rows = []
    for key, r in h1["h1a"].items():
        if r["support"]:
            transition, metric = key.split("|")
            rows.append(f"{transition} & {metric} & {f(r['mean'])} & {r['support']} & "
                        f"{p(r['p'])} & {p(r['p_holm'])} \\\\")
    write("h1a", rows)
    write("h1b", [f"{k} & {f(r['mean'])} & {r['support']} & {p(r['p'])} & {p(r['p_holm'])} \\\\"
                  for k, r in h1["h1b"].items()])

    # S-Table: Stage A monitoring (dialogue arm), detection rate by strategy x W.
    mon = load(STAGE_A / "monitoring.json")
    rows = []
    for w in ("W10", "W20", "W30"):
        for strategy, per in mon["strategies"][w].items():
            cells = []
            for metric in ("CQ", "CS", "RS", "LS", "ES", "SS"):
                c = per[metric]
                cells.append("n/c" if c["parameter"] is None else f(c["detection_rate"], 2))
            rows.append(f"{w[1:]} & {strategy.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    write("stagea_monitoring", rows)

    # S-Table: GPT control-arm screen (per-dialogue detectors).
    screen = load(CTRL / "result.json")
    rows = []
    for w in ("W10", "W20", "W30"):
        for strategy, per in screen["strategies"][w].items():
            cells = [f"{f(per[m]['detection_rate'], 2)} ({f(per[m]['loo_far'], 2)})"
                     if per[m]["detection_rate"] is not None else "n/c"
                     for m in ("CQ", "LS", "ES")]
            rows.append(f"{w[1:]} & {strategy.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    write("control_screen", rows)

    # S-Table: per-dialogue early-signal rule.
    early = load(CTRL / "early_signal/summary.json")["by_window"]
    rows = []
    for w, e in early.items():
        ls = e["LS"]["classes"]
        es = e["ES"]["classes"]
        rows.append(f"{w[1:]} & {e['CQ_signalled']}/{e['n_dialogues']} & "
                    f"{e['null_signal_rate']['CQ']} & {e['LS']['signalled']} & "
                    f"{ls.get('before', 0)}/{ls.get('same', 0)}/{ls.get('after', 0)} & "
                    f"{e['null_signal_rate']['LS']} & {es.get('before', 0)}/"
                    f"{es.get('same', 0)}/{es.get('after', 0)} \\\\")
    write("early_signal", rows)

    # S-Table: aggregated (6 dialogues per turn) monitoring, exploratory.
    agg = load(CTRL / "aggregated/aggregated_alarms.csv")
    rows = []
    for w in (10, 20, 30):
        for metric in ("CQ", "LS", "ES"):
            for det in ("cusum", "page_hinkley"):
                sub = agg[(agg["window"] == w) & (agg["metric"] == metric)
                          & (agg["detector"] == det)].set_index("scenario")
                cells = []
                for scen in ("S1", "S2", "S3", "S0-E", "S0-H"):
                    r = sub.loc[scen]
                    if pd.isna(r["first_alarm"]):
                        cells.append("--")
                    else:
                        tag = "" if pd.isna(r["delay"]) else f" (+{int(r['delay'])})"
                        cells.append(f"{int(r['first_alarm'])}{tag}")
                rows.append(f"{w} & {metric} & {det.replace('_', '-')} & " + " & ".join(cells)
                            + " \\\\")
    write("aggregated", rows)

    # S-Table: full power grid summary (best detector).
    grid = load(ROOT / "results/cdr_stream_v1_power/power_grid.csv")
    best = (grid.sort_values("detection_33", ascending=False)
            .groupby(["calibration", "window", "m", "delta", "change"], as_index=False).first())
    rows = []
    for cal in ("oracle", "n36", "n12"):
        for w in (10, 20, 30):
            for mm in (1, 3, 6, 18):
                cells = []
                for change in (34, 51):
                    sub = best[(best["calibration"] == cal) & (best["window"] == w)
                               & (best["m"] == mm) & (best["change"] == change)]
                    reach = sub[sub["detection_33"] >= 0.8]["delta"]
                    cells.append(f(reach.min(), 1) if len(reach) else "$>$0.6")
                rows.append(f"{cal} & {w} & {mm} & " + " & ".join(cells) + " \\\\")
    write("power_full", rows)

    # S-Table: corrected-ADWIN power (detection by turn 100 at W=20, oracle).
    av = load(ROOT / "results/cdr_stream_v1_power/adwin_v/power_grid.csv")
    av = av[(av["calibration"] == "oracle") & (av["window"] == 20) & (av["change"] == 51)]
    rows = [f"{f(d, 1)} & " + " & ".join(f(av[(av['delta'] == d) & (av['m'] == mm)]
                                             ['detection_by_100'].iloc[0], 2)
                                           for mm in (1, 3, 6, 18)) + " \\\\"
            for d in sorted(av["delta"].unique())]
    write("adwin_v_power", rows)

    # S-Table: ADWIN sensitivity of the LIC replay.
    rows = []
    for variant, base in (("frozen (historical ADWIN)", "results/lost_conversation_v2_stream_replay"),
                          ("corrected ADWIN", "results/adwin_sensitivity/lic_replay_adwin_v"),
                          ("CUSUM for CS/SS", "results/adwin_sensitivity/lic_replay_cusum")):
        s = load(ROOT / base / "summary.json")
        o = load(ROOT / base / "ordered_outcomes.csv")
        adw = o[o["detector"].isin(["adwin", "adwin_v"])]
        cands = sum(c["replay_early_warning_candidate"] for c in s["candidates"])
        rows.append(f"{variant} & {int((adw['category'] != 'no_alarm').sum())}/{len(adw)} & "
                    f"{cands} \\\\")
    write("adwin_replay", rows)

    # S-Table: Llama-family repeat-diversity probes (technical, identical prompts).
    probes = [("Llama-4 Scout (aws)", "results/cdr_stream_v1_pilot/operational_report.json"),
              ("Llama-4 Scout (asu-air)", "results/cdr_stream_v1_kdiv_probe/report.json"),
              ("Llama-4 Maverick (aws)", "results/cdr_stream_v1_kdiv_probe_maverick/report.json"),
              ("Llama-4 Maverick (asu-air)",
               "results/cdr_stream_v1_kdiv_probe_maverick_asuair/report.json"),
              ("Llama-3 8B", "results/cdr_stream_v1_kdiv_probe_llama3_8b/report.json"),
              ("Llama-3 70B", "results/cdr_stream_v1_kdiv_probe_llama3_70b/report.json")]
    rows = []
    for name, rel in probes:
        r = load(ROOT / rel)
        if "by_model" in r:
            rows.append(f"{name} & pilot, 100 turns & "
                        f"{f(r['by_model']['llama4_scout']['k_repeat_text_diversity'], 2)} & "
                        f"{f(r['by_model']['llama4_scout']['contract_validity_dialogue'], 2)} & "
                        f"{r['by_model']['llama4_scout']['truncated']} \\\\")
        else:
            rows.append(f"{name} & {r['turns_complete']} probe turns & "
                        f"{f(r['candidate_k_diversity'], 2)} & "
                        f"{f(r['candidate_contract_validity'], 2)} & "
                        f"{r['candidate_truncated']} \\\\")
    write("kdiv", rows)

    # S-Table: reproducibility hashes.
    sci = yaml.safe_load((ROOT / "configs/lost_conversation_v2_science_asu.yaml")
                         .read_text(encoding="utf-8"))
    ana = yaml.safe_load((ROOT / "configs/cdr_stream_v1_analysis.yaml").read_text(encoding="utf-8"))
    pairs = [("Study 1 raw records", sci["records_sha256"]),
             ("Study 1 confirmatory assessment", sci["confirmatory_assessment_sha256"]),
             ("Study 2 frozen mappings", ana["mappings_sha256"]),
             ("Study 2 main manifest", ana["manifest_sha256"]),
             ("Study 2 Stage A record prefix", ana["stage_a_prefix_sha256"]),
             ("Study 2 analysis protocol", ana["protocol_sha256"]),
             ("Study 2 acquisition fingerprint", ana["fingerprint"])]
    write("hashes", [f"{k} & \\texttt{{{v[:16]}\\ldots}} \\\\" for k, v in pairs])

    # S-Table: the frozen v2 difficulty ladder (design constant, read from the generator).
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from reasoning_monitor.data.cdr_stream import LADDER_V2

    dagger = r"$^\dagger$"
    write("ladder", [str(level) + (dagger if level > 12 else "") + " & "
                     + " & ".join(str(v) for v in spec) + r" \\"
                     for level, spec in LADDER_V2.items()])

    figs = {ROOT / "results/lost_conversation_v2_science/confirmatory/crl_discrimination.pdf":
            "supp_crl_discrimination.pdf",
            ROOT / "results/lost_conversation_v2_science/figures/cumulative_resolution_lag.pdf":
            "supp_crl_trajectory.pdf",
            ROOT / "results/lost_conversation_v2_science/robustness_accuracy_loss/"
                   "crl_accuracy_loss.pdf": "supp_crl_accuracy_loss.pdf",
            **{ROOT / f"results/lost_conversation_v2_science/figures/"
                      f"seven_metrics_semantic_{m}.pdf": f"supp_s1_metrics_{m}.pdf"
               for m in ("gpt4o_mini", "llama4_scout", "mistral_7b")},
            **{STAGE_A / f"trajectories_{m}.pdf": f"supp_s2_metrics_{m}.pdf"
               for m in ("gpt4o_mini", "llama3_70b", "gemma3_12b_it")},
            CTRL / "trajectories_gpt4o_mini.pdf": "supp_s2_control_gpt.pdf",
            ROOT / "results/cdr_stream_v1_control_completion/"
                   "dialogue_vs_state_summary.pdf": "supp_s2_control_completion.pdf",
            CTRL / "early_signal/early_signal_W20.pdf": "supp_s2_early_signal_W20.pdf",
            CTRL / "aggregated/aggregated_W20.pdf": "supp_s2_aggregated_W20.pdf"}
    for model, short in (("llama3_70b", "llama"), ("gemma3_12b_it", "gemma")):
        base = ROOT / "results/cdr_stream_v1_control_monitoring/validation" / model
        # Model-labelled redraws: scripts/relabel_control_validation_figures.py
        figs[base / "relabelled/early_signal_W20.pdf"] = \
            f"supp_s2_validation_{short}_early.pdf"
        figs[base / "relabelled/aggregated_W20.pdf"] = \
            f"supp_s2_validation_{short}_agg.pdf"
    for src, name in figs.items():
        USED[str(src.relative_to(ROOT))] = sha(src)
        shutil.copyfile(src, FIG / name)
    (GEN / "supp_assets_manifest.json").write_text(json.dumps(USED, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"wrote supplementary tables and {len(figs)} figures; {len(USED)} inputs hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
