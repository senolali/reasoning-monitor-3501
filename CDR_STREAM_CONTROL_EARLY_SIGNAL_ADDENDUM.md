# Addendum: per-dialogue early-signal analysis on the GPT control arm (EXPLORATORY)

Fixed 2026-10-04, before computing, at the investigator's request. It
complements `CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md` and does not change
it. It reuses that protocol's per-turn control-arm metrics,
`results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini/control_turn_metrics.csv`.

**Dialogues.** All 18 GPT-4o-mini control-arm drift dialogues: S1, S2 and S3,
6 each. They are selected by design, not by outcome. The 12 S0 dialogues
(S0-E and S0-H) measure how often the rule signals when nothing changes.

**Metrics.** CQ, LS and ES; the control arm has K=1, so there is no CS, SS or
RS.

**Signal rule**, fixed in advance and applied identically to every metric,
dialogue and W ∈ {10, 20, 30}:

1. The windowed value s(t) is the mean of the valid values in turns
   t−W+1 … t.
2. The baseline is computed from the per-turn values in turns 1–30, which
   precede every scheduled change (the earliest is at 34):
   - μ₀ is their mean;
   - σ₀ is their standard deviation.
3. Signal turn: the first t ≥ 31 at which s(t) < μ₀ − 2·σ₀/√W holds on 3
   consecutive turns. The signal turn is the first of those three turns.
4. A metric with no such turn by turn 100 has "no signal".
5. Lead = signal turn(metric) − signal turn(CQ). A negative lead means the
   metric signalled earlier than CQ.

**Reporting.**

- Per dialogue: the signal turns of CQ, LS and ES, and the scheduled change
  point.
- Per W: how often LS or ES signalled before, at the same turn as, or after
  CQ, with median leads.
- On the S0 dialogues: the signal rate of each metric, which acts as the false
  signal rate.
- Figures: one panel per dialogue showing CQ, LS and ES windowed trajectories,
  baselines, thresholds and signal turns.

**Status.** GPT's control data were seen in earlier figures, so everything
here is exploratory. Any claim needs the same rule applied, unchanged, to
unseen data (the Llama and Gemma control arms).

## Part 2: aggregated monitoring of 6 dialogues per turn (EXPLORATORY)

Fixed 2026-10-04, before computing, at the investigator's request.

**Streams.**

- For each of S1, S2 and S3, the per-turn mean of the 6 GPT control dialogues
  of that scenario, for CQ, LS and ES. This gives 3 drift streams.
- S0-E and S0-H are aggregated the same way and reported as sanity streams.

**Null for calibration.** The control arm carries no history, so turn order
within a stationary regime is exchangeable. Each null stream is built as
follows:

1. For each of the 6 S0-E dialogues, permute its turns independently, with
   seed 20261010.
2. Recompute ES on the permuted order. Each null stream needs its own
   permutation.
3. Average the 6 dialogues per turn.

This yields 2,000 null streams. S0-E is used because every drift scenario
starts in the easy regime.

**Detectors.**

- The unchanged reference detectors and grids: cusum, page_hinkley, ewma,
  adwin and ddm.
- W ∈ {10, 20, 30}, with signals from `build_window_signal`.
- For each (metric, detector, W), the threshold is the most sensitive grid
  value whose null first-alarm rate is ≤ 0.05.

**Outcomes.** For each drift stream, the first alarm, classified against each
scheduled change:

| Stream | Change(s) |
|---|---|
| S1 | 34 (E→M), and 68 (M→H) for alarms at or after 68 |
| S2 | 51 |
| S3 | 34 |

- A detection is the first alarm in [c, c+33]. The delay is the alarm turn
  minus c.
- Lead is LS or ES versus CQ under the same detector type and W.

**Limitations.**

- There are only 3 drift streams, so the results are descriptive.
- The nulls come from the same S0-E dialogues, permuted.
- Any claim needs Llama and Gemma control data.
