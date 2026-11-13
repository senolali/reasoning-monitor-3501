"""Exploratory sliding-window early-warning screen on the LHE-RQ-v1 pilots (call-free).

Why these data: the k in an LHE conversation is fixed, so difficulty does not
rise on a schedule (unlike Elle S2). The pilots also already contain
spontaneous declines. Only primary replies exist (K=1, L=0), so CS, SS and RS
cannot be measured. The measurable signals are listed below. This is a
post-hoc screen on calibration conversations. It is not confirmatory and no
threshold is claimed.

Signals (risk orientation fixed in advance; higher = riskier)
  LS   1 - LS (real NLI, rqeval consecutive-step formula; gold-free)
  LEN  log output tokens (gold-free)
  FMT  1 if the reply is non-strict (no single bare-integer answer tag; gold-free)
  ES   1 - harmonic(step correctness, conciseness) (gold-dependent; reference only)
  CQ   1 - step correctness (the target)
  Task-side references (known without the model): TURN index, MAG = log10(|true sum|+1).

Normalisation: z against a leave-one-conversation-out reference. The
reference is the same unit's OTHER conversations, turns 1-10, so it uses no
information about the monitored conversation.
Windows W in {10, 20, 30}: trailing mean over min(W, turns so far), with at
least 5 turns.
Detectors from turn 6: threshold z_W >= 3, and CUSUM on z_W (k = 0.5, h = 5).
Outputs: lead = tau - alarm; alarm before the CQ alarm; false alarms in
complete non-declining conversations of the same unit (a constant-difficulty
null); and prospective AUROC on healthy-now turns (error in the next 5 turns),
raw and within turn-quintiles.

  python scripts\\explore_lhe_windows.py
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

PILOTS = ("configs/lhe_rq_v1_pilot2_asu.yaml", "configs/lhe_rq_v1_pilot3_asu.yaml")
OUT = ROOT / "results/lhe_rq_v1_window_screen"
WINDOWS = (10, 20, 30)
SIGNALS = ("cq", "es", "ls", "len", "fmt", "turn", "mag")
GOLD_FREE = {"ls", "len", "fmt"}
MIN_DECLINING = 3


def load_conversations():
    from reasoning_monitor.dialogue_six_metrics import StrictScorers
    from reasoning_monitor.models.clients import reasoning_steps_from_text

    cache = OUT / "turns.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    scorers = StrictScorers(with_ss=False)
    convs = []
    for path in PILOTS:
        config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        dictionary, samples = lhe.load_dataset(ROOT / config["data_dir"])
        models = [m for m in config["models"] if m.get("mode", "cot") == "cot"]
        units = acq.build_units(models, samples, config["conversation_ids"], config["horizon"])
        saved, _ = acq.load_checkpoint(ROOT / config["output_dir"] / "data/records.jsonl",
                                       fingerprint(config, dictionary))
        rows = defaultdict(list)
        for row in saved.values():
            rows[(row["model"], row["dialogue_id"])].append(row)
        pending = []
        for unit in units:
            cell = rows[(unit["model"], unit["dialogue_id"])]
            if not cell:
                continue
            scored = lhe.score_conversation(cell, unit["conversation"], 1, 0, scorers=None)
            primaries = {int(r["turn"]): r for r in cell if r["role"] == "primary"}
            observed = max(primaries) if primaries else 0
            scored = [r for r in scored if r["turn"] <= observed]
            steps = [reasoning_steps_from_text(lhe.reasoning_text(primaries[r["turn"]]["raw_text"]))
                     if r["turn"] in primaries and lhe.valid_call(primaries[r["turn"]]) else []
                     for r in scored]
            pending.append((path, unit, scored, steps))
        flat = [s for _, _, _, steps in pending for s in steps]
        print(f"{path}: LS NLI on {len(flat)} replies ...", flush=True)
        ls_values = scorers.coherence_many(flat)
        pos = 0
        for path_, unit, scored, steps in pending:
            ls_part = ls_values[pos:pos + len(steps)]
            pos += len(steps)
            convs.append({
                "pilot": Path(path_).stem, "unit": unit["model"], "dialogue_id": unit["dialogue_id"],
                "complete": len(scored) == unit["conversation"]["horizon"]
                if "horizon" in unit["conversation"] else len(scored) >= 100,
                "step": [r["step_correct"] for r in scored],
                "strict": [bool(r["strict"]) for r in scored],
                "tokens": [r["n_tokens"] for r in scored],
                "true_sum": [r["true_sum"] for r in scored],
                "ls": [float(v) if v is not None else math.nan for v in ls_part]})
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(convs, default=float), encoding="utf-8")
    return convs


def raw_signals(c):
    step = np.array(c["step"], float)
    tokens = np.array(c["tokens"], float)
    finite = tokens[np.isfinite(tokens)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
    concise = 1 - (tokens - lo) / (hi - lo) if hi > lo else np.zeros_like(tokens)
    es = np.where(np.isnan(step), np.nan,
                  np.where(step + concise > 0, 2 * step * concise / np.maximum(step + concise, 1e-12), 0))
    return {"cq": 1 - step, "es": 1 - es, "ls": 1 - np.array(c["ls"], float),
            "len": np.log(np.maximum(tokens, 1)),
            "fmt": np.array([0.0 if s else 1.0 for s in c["strict"]]),
            "turn": np.arange(1, len(step) + 1, dtype=float),
            "mag": np.log10(np.abs(np.array(c["true_sum"], float)) + 1)}


def trailing(x, w):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= min(5, i + 1) and len(seg):
            out[i] = seg.mean()
    return out


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    keep = np.isfinite(scores)
    s, y = scores[keep], labels[keep]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_auroc(s, y, g):
    num = den = 0.0
    for k in np.unique(g):
        m = g == k
        a = auroc(s[m], y[m])
        n = (y[m] == 1).sum() * (y[m] == 0).sum()
        if np.isfinite(a) and n:
            num, den = num + a * n, den + n
    return num / den if den else math.nan


def first_alarm(z, kind):
    s = 0.0
    for t in range(5, len(z)):
        if not np.isfinite(z[t]):
            continue
        if kind == "z>=3" and z[t] >= 3:
            return t + 1
        if kind == "cusum":
            s = max(0.0, s + z[t] - 0.5)
            if s >= 5:
                return t + 1
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    convs = load_conversations()
    for c in convs:
        c["tau"] = lhe.decline_onset(c["step"])
        c["healthy"] = lhe.healthy_start(c["step"])
        c["raw"] = raw_signals(c)
    by_unit = defaultdict(list)
    for c in convs:
        by_unit[(c["pilot"], c["unit"])].append(c)
    eligible = {k: v for k, v in by_unit.items()
                if sum(1 for c in v if c["tau"] and c["healthy"]) >= MIN_DECLINING}
    overview = [{"pilot": p, "unit": u, "conversations": len(v),
                 "declining_healthy": sum(1 for c in v if c["tau"] and c["healthy"]),
                 "complete_non_declining": sum(1 for c in v if not c["tau"] and c["complete"]),
                 "taus": sorted(c["tau"] for c in v if c["tau"]),
                 "eligible": (p, u) in eligible} for (p, u), v in sorted(by_unit.items())]
    report = {"label": "EXPLORATORY post-hoc LHE window screen; K=1 so CS/SS/RS unmeasurable",
              "units": overview, "detectors": [], "auroc": []}

    # leave-one-conversation-out reference (turns 1-10 of the other conversations)
    for (p, u), cs in eligible.items():
        for c in cs:
            c["z"] = {}
            for w in WINDOWS:
                for sig in SIGNALS:
                    ref = np.concatenate([trailing(o["raw"][sig], w)[:10] for o in cs if o is not c])
                    ref = ref[np.isfinite(ref)]
                    mu, sd = (ref.mean(), ref.std()) if ref.size else (0.0, 1.0)
                    sd = max(sd, 0.02 if sig in ("cq", "es", "ls", "fmt") else 0.05)
                    c["z"][(w, sig)] = (trailing(c["raw"][sig], w) - mu) / sd

    for w in WINDOWS:
        for kind in ("z>=3", "cusum"):
            for sig in SIGNALS:
                leads, before_cq, missed, false_alarms, nulls = [], 0, 0, 0, 0
                for cs in eligible.values():
                    for c in cs:
                        t = first_alarm(c["z"][(w, sig)], kind)
                        if c["tau"] and c["healthy"]:
                            if t is None:
                                missed += 1
                                continue
                            leads.append(c["tau"] - t)
                            cq = first_alarm(c["z"][(w, "cq")], kind)
                            before_cq += int(sig != "cq" and (cq is None or t < cq))
                        elif not c["tau"] and c["complete"]:
                            nulls += 1
                            false_alarms += t is not None
                n = len(leads) + missed
                report["detectors"].append({
                    "W": w, "detector": kind, "signal": sig, "gold_free": sig in GOLD_FREE,
                    "declining": n, "alarmed": len(leads), "early": sum(v > 0 for v in leads),
                    "median_lead": float(np.median(leads)) if leads else None,
                    "before_cq": before_cq if sig != "cq" else None,
                    "false_alarms": false_alarms, "null_conversations": nulls, "leads": leads})

    rng = np.random.default_rng(20261109)
    for w in WINDOWS:
        for scope in ["pooled"] + [u for _, u in eligible]:
            cs_sel = [c for (p, u), cs in eligible.items() for c in cs if scope in ("pooled", u)]
            def collect(sel):
                rows = defaultdict(list)
                for c in sel:
                    st = c["step"]
                    for t in range(4, len(st) - 5):
                        if all(st[t - k] == 1 for k in range(5)):
                            rows["y"].append(int(any(v != 1 for v in st[t + 1:t + 6])))
                            rows["turn_raw"].append(t + 1)
                            for sig in SIGNALS:
                                rows[sig].append(c["z"][(w, sig)][t])
                return {k: np.array(v, float) for k, v in rows.items()}
            base = collect(cs_sel)
            if not len(base.get("y", [])) or base["y"].sum() == 0:
                continue
            q = np.digitize(base["turn_raw"], np.quantile(base["turn_raw"], [.2, .4, .6, .8]))
            for sig in SIGNALS:
                if sig == "cq":
                    continue
                boots = []
                for _ in range(1000):
                    b = collect([cs_sel[i] for i in rng.integers(0, len(cs_sel), len(cs_sel))])
                    if len(b.get("y", [])):
                        boots.append(auroc(b[sig], b["y"]))
                boots = [v for v in boots if np.isfinite(v)]
                report["auroc"].append({
                    "W": w, "scope": scope, "signal": sig, "positives": int(base["y"].sum()),
                    "n": len(base["y"]), "auroc": auroc(base[sig], base["y"]),
                    "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))] if boots else None,
                    "within_turn_quintiles": None if sig == "turn" else strat_auroc(base[sig], base["y"], q)})
    (OUT / "window_screen_report.json").write_text(json.dumps(report, indent=2, default=float),
                                                   encoding="utf-8")
    plot(eligible)
    for r in overview:
        print(r)
    print()
    for r in report["detectors"]:
        print(f"W={r['W']:2d} {r['detector']:5s} {r['signal']:4s} alarm {r['alarmed']}/{r['declining']} "
              f"early {r['early']} medLead {r['median_lead']} beforeCQ {r['before_cq']} "
              f"FA {r['false_alarms']}/{r['null_conversations']}")
    print()
    for r in report["auroc"]:
        ci = r["ci"] or [math.nan, math.nan]
        wt = r["within_turn_quintiles"]
        print(f"W={r['W']:2d} {r['scope']:22s} {r['signal']:4s} AUROC {r['auroc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"within-turn {'-' if wt is None else f'{wt:.3f}'} (+{r['positives']}/{r['n']})")
    return 0


def plot(eligible) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"cq": "#111827", "es": "#059669", "ls": "#2563eb", "len": "#d97706", "fmt": "#9333ea",
              "turn": "#9ca3af", "mag": "#a3a3a3"}
    decl = [c for cs in eligible.values() for c in cs if c["tau"] and c["healthy"]]
    rel = np.arange(-40, 21)
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(15, 4), sharey=True)
    for ax, w in zip(axes, WINDOWS):
        for sig in SIGNALS:
            mat = np.full((len(decl), rel.size), np.nan)
            for i, c in enumerate(decl):
                z = c["z"][(w, sig)]
                for j, r in enumerate(rel):
                    t = c["tau"] - 1 + r
                    if 0 <= t < len(z):
                        mat[i, j] = z[t]
            ax.plot(rel, np.nanmedian(mat, 0), color=colors[sig],
                    ls="--" if sig in ("turn", "mag") else "-", lw=2 if sig == "cq" else 1.4,
                    label=sig.upper() + (" (task ref.)" if sig in ("turn", "mag") else ""))
        ax.axvline(0, color="#ef4444", ls="--", lw=1)
        ax.axhline(3, color="#6b7280", ls=":", lw=.8)
        ax.set_ylim(-4, 15)
        ax.set_title(f"W={w}", fontsize=10)
        ax.set_xlabel("turns relative to onset tau")
    axes[0].set_ylabel(f"median risk z (n={len(decl)} declining conversations)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("LHE pilots (constant k): windowed signals aligned at decline onset", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"window_event_locked.{ext}", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
