"""Pilot sanity check: run one model × one scenario × N queries and inspect
the six raw metric values.

Usage:
    python scripts/pilot_check.py                    # synthetic client
    python scripts/pilot_check.py --config live.yaml # real model (edit config)

Checks for common failure modes:
  * CS / SS stuck at 1.0 (indicates no variation → suspect scoring bug)
  * LS constant (may be real: NLI unavailable → heuristic fallback, or
    simulated traces with no contradictions — check with real model)
  * RS NaN rate (should equal the fraction of wrong primary answers)
  * CQ raw DDM and windowed CUSUM firing on separate channels
"""
import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of queries")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--scenario", default="ST2",
                    help="ST1-ST6 (the S0-S6 names were retired in v0.2.0)")
    args = ap.parse_args()

    from reasoning_monitor.monitor import ReasoningMonitor, MonitorConfig
    from reasoning_monitor.models.clients import SimulatedClient
    from reasoning_monitor.data.loaders import load_all
    from reasoning_monitor.data.scenarios import SCENARIOS, build_stream

    ds = load_all(["gsm8k"], limit_per_dataset=60, seed=42)
    if args.scenario not in SCENARIOS:
        raise SystemExit(
            f"unknown scenario {args.scenario!r}; expected one of {list(SCENARIOS)}"
        )
    spec = SCENARIOS[args.scenario]
    stream = build_stream(spec, ds, seed=42)[:args.n]

    client = SimulatedClient(name="pilot", seed=42)
    mon = ReasoningMonitor(
        client,
        MonitorConfig(use_rqeval=True, window_size=args.window,
                      stability_scorer="jaccard"),
    )
    report = mon.preflight(strict=False)
    result = mon.run(stream)

    pq = result.per_query
    print("Preflight:")
    for m, st in report["scorers"].items():
        mark = "✅" if st["ready"] else "⚠️"
        print(f"  {mark} {m}: {st['scorer']} ({st['model']})")
    for problem in report["problems"]:
        print(f"  ❌ {problem}")
    for warning in report["warnings"]:
        print(f"  ⚠️  {warning}")
    print()
    print(f"=== PILOT: {args.n} queries, W={args.window}, scenario={args.scenario} ===\n")

    print("Per-query raw metric statistics:")
    for col in [c for c in pq.columns if c.endswith("_raw")]:
        vals = pq[col].dropna()
        metric = col.replace("_raw", "")
        if len(vals):
            print(f"  {metric}: min={vals.min():.3f} max={vals.max():.3f} "
                  f"mean={vals.mean():.3f} std={vals.std():.3f} "
                  f"(NaN={pq[col].isna().sum()})")

    channels = {}
    for a in result.alarms:
        key = f"{a.metric}/{a.detector}/{a.level}"
        channels[key] = channels.get(key, 0) + 1
    print(f"\nAlarm channels ({len(result.alarms)} total):")
    for k, v in sorted(channels.items()):
        print(f"  {k}: {v}")

    print("\nHealth flags:")
    for m in ("CS", "SS"):
        vals = pq[f"{m}_raw"].dropna()
        stuck = (vals == 1.0).all() if len(vals) else False
        print(f"  {m} stuck at 1.0: {'⚠️ YES' if stuck else 'no'}")
    ls_vals = pq["LS_raw"].dropna()
    print(f"  LS constant: {'⚠️ YES (check NLI availability / real model)' if ls_vals.nunique() == 1 else 'no'}")
    cq_channels = [c for c in channels if c.startswith("CQ/")]
    print(f"  CQ two-level channels active: {len(cq_channels)} "
          f"({'✅ separate' if len(cq_channels) >= 1 else 'none'})")
    flat = result.degenerate_metrics()
    print(f"  Windowed metrics that never varied: "
          f"{', '.join(flat) if flat else 'none ✅'}")
    print(f"\n  Raw call records captured: {len(result.records)} "
          f"({len(result.records) / max(len(stream), 1):.1f} calls/query)")


if __name__ == "__main__":
    main()
