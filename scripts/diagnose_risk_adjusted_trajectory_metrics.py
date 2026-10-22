"""Run the frozen, call-free RAT-v1 metric-subset ablation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.evaluation.risk_adjusted_trajectory_ablation import (  # noqa: E402
    assess_metric_ablation,
)
from reasoning_monitor.evaluation.risk_adjusted_trajectory_analysis import (  # noqa: E402
    build_risk_set,
)

GATE_PATH = ROOT / "configs" / "risk_adjusted_trajectory_v1_metric_ablation.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    gate = yaml.safe_load(GATE_PATH.read_text(encoding="utf-8"))
    for relative_path, expected_digest in gate["expected_source_sha256"].items():
        digest = _sha256(ROOT / relative_path)
        if digest != expected_digest:
            raise ValueError(f"source hash mismatch for {relative_path}: {digest}")

    frames = []
    source_sha256 = {}
    for model in gate["expected_models"]:
        source = gate["sources"][model]
        path = ROOT / source["path"]
        digest = _sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"source hash mismatch for {model}: {digest}")
        frame = pd.read_csv(path)
        frame["model"] = model
        if len(frame) != int(gate["expected_queries_per_model"]):
            raise ValueError(f"unexpected per-query count for {model}: {len(frame)}")
        frames.append(frame)
        source_sha256[str(source["path"])] = digest

    risk_set = build_risk_set(pd.concat(frames, ignore_index=True))
    report = assess_metric_ablation(risk_set, gate)
    report["source_sha256"] = source_sha256
    report["gate"] = gate
    output_path = ROOT / gate["output_path"]
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(not report["screen_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
