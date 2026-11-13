"""Redraw the Llama-3 70B and Gemma 3 12B control-arm validation figures with
correct model labels (call-free, plot-only).

The original figures carried a hard-coded "GPT control arm" title. This script
re-reads the frozen CSV outputs, recomputes nothing, and writes the corrected
figures to a new ``relabelled/`` folder next to each model's results, so no
frozen artefact is overwritten.

  python scripts\relabel_control_validation_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import control_aggregated_monitoring as agg  # noqa: E402
import control_early_signal as early  # noqa: E402

VALIDATION = ROOT / "results/cdr_stream_v1_control_monitoring/validation"


def main() -> int:
    for model, label in (("llama3_70b", "Llama-3 70B"), ("gemma3_12b_it", "Gemma 3 12B")):
        base = VALIDATION / model
        out = base / "relabelled"
        out.mkdir(exist_ok=True)
        frame = pd.read_csv(base / "control_turn_metrics.csv")
        signal_table = pd.read_csv(base / "early_signal/signal_turns.csv")
        for window in early.WINDOWS:
            early.plot(frame, signal_table, out, window, model_label=label)
        alarms = pd.read_csv(base / "aggregated/aggregated_alarms.csv")
        streams = {s: agg.aggregate(frame, s) for s in ("S1", "S2", "S3", "S0-E", "S0-H")}
        agg.plot(streams, alarms, output=out, model_label=label)
        print(f"{model}: wrote {sorted(p.name for p in out.glob('*.pdf'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
