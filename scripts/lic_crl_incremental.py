"""POST-HOC: does CRL add information beyond the last pre-final readiness gap?

Reviewer concern: the frozen CRL target (a positive final readiness gap) is the
final value of the same readiness process that CRL averages, so CRL may only
measure persistence. This call-free analysis, labelled post-hoc, compares:

  (a) the last comparable pre-final positive readiness gap alone, and
  (b) CRL conditional on that last gap (AUROC stratified by model x last gap).

It reuses the frozen ``crl_discrimination`` routine (same seed and repetitions)
by substituting the score or the strata. Frozen outputs are not modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.lost_conversation_science_analysis import (  # noqa: E402
    crl_discrimination,
)

SCI = ROOT / "results/lost_conversation_v2_science"
OUT = SCI / "crl_incremental_posthoc"
SEED, REPS = 20261002, 10_000


def main() -> int:
    crl = pd.read_csv(SCI / "crl_by_task.csv")
    traj = pd.read_csv(SCI / "crl_trajectory.csv")
    pre = traj[(~traj["is_final"]) & (traj["comparable"])]
    last = (pre.sort_values("turn").groupby(["model", "task_id"]).tail(1)
            .set_index(["model", "task_id"])["positive_gap"])
    crl["last_prefinal_gap"] = [last.get((m, t)) for m, t in zip(crl.model, crl.task_id,
                                                                  strict=True)]
    crl = crl.dropna(subset=["last_prefinal_gap"])

    reproduced = crl_discrimination(crl, seed=SEED, repetitions=REPS)["pooled_model_stratified"]
    last_only = crl_discrimination(crl.assign(CRL=crl["last_prefinal_gap"]), seed=SEED,
                                   repetitions=REPS)["pooled_model_stratified"]
    conditional = crl_discrimination(
        crl.assign(model=crl["model"] + "|last_gap=" + crl["last_prefinal_gap"].astype(int)
                   .astype(str)), seed=SEED, repetitions=REPS)["pooled_model_stratified"]
    table = (crl.groupby(["model", "last_prefinal_gap"])["final_positive_readiness_gap"]
             .agg(["count", "sum"]).reset_index()
             .rename(columns={"count": "units", "sum": "target_cases"}))
    result = {
        "analysis": "LIC-RQ-v2 CRL incremental value (POST-HOC)",
        "post_hoc": True,
        "inputs_sha256": {name: hashlib.sha256((SCI / name).read_bytes()).hexdigest()
                          for name in ("crl_by_task.csv", "crl_trajectory.csv")},
        "units_with_prefinal_turns": int(len(crl)),
        "reproduced_frozen_crl_pooled": reproduced,
        "last_prefinal_gap_alone_pooled": last_only,
        "crl_conditional_on_last_gap_pooled": conditional,
        "cross_tab": table.to_dict(orient="records"),
        "interpretation_rule": ("CRL adds information beyond persistence only if its "
                                "AUROC conditional on the last pre-final gap exceeds 0.5 "
                                "with task-clustered permutation p < 0.05"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assessment.json").write_text(json.dumps(result, indent=2, default=float) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
