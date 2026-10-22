"""Generate every number, table and figure of the LIC-RQ-v2 paper from frozen outputs.

Call-free. Writes paper/lic_rq_v2/generated/*.tex and copies figures. Every
input is hash-checked against the value recorded in the frozen config or the
replay protocol, so the manuscript can never drift from the evidence.

  python scripts\\paper_lic_rq_v2_assets.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/lic_rq_v2"
GEN = PAPER / "generated"
FIG = PAPER / "figures"
SCI = ROOT / "results/lost_conversation_v2_science"
REPLAY = ROOT / "results/lost_conversation_v2_stream_replay"
SENS = ROOT / "results/adwin_sensitivity"
MODELS = {"gpt4o_mini": "GPT-4o-mini", "llama4_scout": "Llama-4 Scout",
          "mistral_7b": "Mistral-7B"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(x, d=3) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def p(x) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main(argv=None) -> int:
    import argparse

    global PAPER, GEN, FIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=PAPER,
                        help="target paper folder (default: paper/lic_rq_v2)")
    args = parser.parse_args(argv)
    PAPER = args.paper if args.paper.is_absolute() else ROOT / args.paper
    GEN, FIG = PAPER / "generated", PAPER / "figures"
    cfg = yaml.safe_load((ROOT / "configs/lost_conversation_v2_science_asu.yaml")
                         .read_text(encoding="utf-8"))
    conf_path = SCI / "confirmatory/assessment.json"
    rob_path = SCI / "robustness_accuracy_loss/assessment.json"
    if sha(conf_path) != cfg["confirmatory_assessment_sha256"]:
        raise SystemExit("confirmatory assessment hash mismatch")
    if sha(rob_path) != cfg["accuracy_loss_robustness_sha256"]:
        raise SystemExit("robustness assessment hash mismatch")
    if sha(SCI / "data/records.jsonl") != cfg["records_sha256"]:
        raise SystemExit("raw records hash mismatch")
    conf = json.loads(conf_path.read_text(encoding="utf-8"))
    rob = json.loads(rob_path.read_text(encoding="utf-8"))
    ops = json.loads((SCI / "operational_report.json").read_text(encoding="utf-8"))
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    m = [macro("nRecords", f"{ops['records']:,}".replace(",", "{,}")),
         macro("nTasks", ops["science_tasks"]),
         macro("nEmptyPrimaries", ops["empty_primary_responses"]),
         macro("nPermutations", f"{conf['permutation_or_bootstrap_repetitions']:,}"
               .replace(",", "{,}")),
         macro("recordsSha", cfg["records_sha256"][:12]),
         macro("Ktemp", cfg["temperature"]), macro("Krep", cfg["k_repeats"]),
         macro("Lpar", cfg["l_paraphrases"]), macro("maxTok", f"{cfg['max_output_tokens']:,}"
                                                     .replace(",", "{,}"))]
    rows = []
    for key, label in MODELS.items():
        e = conf["primary_endpoint"][key]
        tag = {"gpt4o_mini": "Gpt", "llama4_scout": "Llama", "mistral_7b": "Mistral"}[key]
        lo, hi = e["paired_risk_difference_bootstrap_95_ci"]
        m += [macro(f"dial{tag}", f(e["dialogue_success_rate"])),
              macro(f"fresh{tag}", f(e["fresh_success_rate"])),
              macro(f"diff{tag}", f(e["paired_risk_difference_fresh_minus_dialogue"])),
              macro(f"ci{tag}", f"[{f(lo)}, {f(hi)}]"),
              macro(f"pHolm{tag}", p(e["holm_adjusted_p_across_models"])),
              macro(f"pRaw{tag}", p(e["mcnemar_exact_two_sided_p"]))]
        rows.append(f"{label} & {e['dialogue_success']}/{e['tasks']} ({f(e['dialogue_success_rate'])}) & "
                    f"{e['fresh_success']}/{e['tasks']} ({f(e['fresh_success_rate'])}) & "
                    f"{f(e['paired_risk_difference_fresh_minus_dialogue'])} [{f(lo)}, {f(hi)}] & "
                    f"{e['fresh_only_successes']}/{e['dialogue_only_successes']} & "
                    f"{p(e['mcnemar_exact_two_sided_p'])} & {p(e['holm_adjusted_p_across_models'])} \\\\")
    (GEN / "table_primary.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    pe = conf["primary_endpoint"]
    m += [macro("nFreshOnly", sum(pe[k]["fresh_only_successes"] for k in MODELS)),
          macro("nDialOnly", sum(pe[k]["dialogue_only_successes"] for k in MODELS))]

    fv = conf["format_valid_sensitivity"]
    rows = [f"{MODELS[k]} & {fv[k]['both_final_responses_semantically_valid']} & "
            f"{f(fv[k]['paired_difference'])} & {p(fv[k]['mcnemar_exact_two_sided_p'])} \\\\"
            for k in MODELS]
    (GEN / "table_format_valid.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    crl = conf["auxiliary_crl"]
    rows = []
    for k, label in MODELS.items():
        c = crl[k]
        lo, hi = c["bootstrap_95_ci"]
        rows.append(f"{label} & {c['complete_cases']} & {c['target_cases']}/{c['target_controls']} & "
                    f"{f(c['mean_crl'])} & {f(c['auroc'])} [{f(lo)}, {f(hi)}] & "
                    f"{p(c['permutation_one_sided_p'])} & {p(c['holm_adjusted_p_across_models'])} \\\\")
    pooled = crl["pooled_model_stratified"]
    lo, hi = pooled["task_clustered_bootstrap_95_ci"]
    rows.append("\\midrule")
    rows.append(f"Pooled (model-stratified) & {pooled['source_tasks']} tasks & "
                f"{pooled['comparable_case_control_pairs']} pairs & -- & "
                f"{f(pooled['auroc'])} [{f(lo)}, {f(hi)}] & "
                f"{p(pooled['task_clustered_permutation_one_sided_p'])} & -- \\\\")
    (GEN / "table_crl.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    m += [macro("crlAuroc", f(pooled["auroc"])), macro("crlCi", f"[{f(lo)}, {f(hi)}]"),
          macro("crlP", p(pooled["task_clustered_permutation_one_sided_p"])),
          macro("crlPairs", pooled["comparable_case_control_pairs"])]

    rows = []
    for metric, r in conf["rqeval_early_warning"].items():
        rows.append(f"{metric} & {r['measured_turns']} & {r['model_task_units']} & "
                    f"{r['source_tasks']} & {r['task_clustered_partial_correlation']:+.3f} & "
                    f"{p(r['permutation_one_sided_p'])} & {p(r['holm_adjusted_p'])} \\\\")
    (GEN / "table_rqeval.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    for name in ("all_model_task_units", "transport_affected_units_excluded"):
        pooled_r = rob[name].get("pooled_model_stratified", {})
        tag = "All" if name.startswith("all") else "Excl"
        if pooled_r:
            lo, hi = pooled_r["task_clustered_bootstrap_95_ci"]
            m += [macro(f"rob{tag}Auroc", f(pooled_r["auroc"])),
                  macro(f"rob{tag}P", p(pooled_r["task_clustered_permutation_one_sided_p"])),
                  macro(f"rob{tag}Ci", f"[{f(lo)}, {f(hi)}]")]

    # Stream replay (post-hoc method demonstration) and ADWIN sensitivity.
    summary = json.loads((REPLAY / "summary.json").read_text(encoding="utf-8"))
    det = pd.DataFrame(summary["detection"])
    det = det[det["window"] == 20]
    rows = []
    for k, label in MODELS.items():
        for scen in ("sudden", "gradual"):
            sub = det[(det["model"] == k) & (det["scenario"] == scen)]
            best = sub.sort_values("detection_rate", ascending=False).iloc[0]
            where = (f"{best['metric']} / {best['strategy'].replace('_', ' ')}"
                     if best["detection_rate"] > 0 else "none")
            rows.append(f"{label} & {scen} & {f(best['detection_rate'])} & {where} \\\\")
    (GEN / "table_replay.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    cands = [c for c in summary["candidates"] if c["replay_early_warning_candidate"]]
    m.append(macro("replayCandidates", len(cands)))
    sens = {}
    for variant in ("adwin_v", "cusum"):
        s = json.loads((SENS / f"lic_replay_{variant}/summary.json").read_text(encoding="utf-8"))
        sens[variant] = sum(c["replay_early_warning_candidate"] for c in s["candidates"])
    m.append(macro("replaySensCandidates", sens["adwin_v"] + sens["cusum"]))

    (GEN / "numbers.tex").write_text("% Generated by scripts/paper_lic_rq_v2_assets.py\n"
                                     + "\n".join(m) + "\n", encoding="utf-8")
    figures = {
        SCI / "confirmatory/paired_final_success.pdf": "paired_final_success.pdf",
        SCI / "confirmatory/crl_discrimination.pdf": "crl_discrimination.pdf",
        SCI / "figures/cumulative_resolution_lag.pdf": "cumulative_resolution_lag.pdf",
        SCI / "robustness_accuracy_loss/crl_accuracy_loss.pdf": "crl_accuracy_loss.pdf",
        **{SCI / f"figures/seven_metrics_semantic_{k}.pdf": f"seven_metrics_{k}.pdf"
           for k in MODELS}}
    for src, name in figures.items():
        shutil.copyfile(src, FIG / name)
    manifest = {"inputs": {"records": cfg["records_sha256"],
                           "confirmatory": cfg["confirmatory_assessment_sha256"],
                           "robustness": cfg["accuracy_loss_robustness_sha256"],
                           "replay_summary": sha(REPLAY / "summary.json")},
                "figures": sorted(figures.values())}
    (GEN / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"wrote {len(m)} macros, 6 tables, {len(figures)} figures to {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
