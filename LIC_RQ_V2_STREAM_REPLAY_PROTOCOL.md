# LIC-RQ-v2-SR: Session-Stream Detector Replay Protocol

Status: **FROZEN (2026-10-02)**, before any replay computation. The investigator
approved the draft and accepted every recommendation in Section 12 (D1–D4). The
freeze is recorded in `DEVIATIONS_FROM_PROPOSAL.md`. The analysis is entirely
call-free. It makes no model, gateway or network call.

Clarifications fixed at freeze time, before any computation:

- "Code default warm-up" means each detector's own default: 20 updates for
  CUSUM, Page-Hinkley, EWMA and ADWIN, and 30 updates with `min_count=30` for DDM.
  This is the same convention as Sections 36 and 39.
- H2 paired intervals use 2,000 bootstrap resamples of realizations, seed 121042.
- The same index permutations are shared across models as well as across metrics,
  strategies and W, as in Section 36.
- Strategy-to-detector pairs are read from `assign_detector` at run time, not
  re-typed. A (metric, detector, W) cell shared by two strategies is calibrated
  once, because the null streams and selection rule are identical.
- Validation and ordered streams are evaluated only for parameters already
  locked by calibration. Ordered streams are evaluated only for cells that
  passed validation in all three models.
- The final-turn sensitivity analysis (Section 4) repeats calibration,
  validation and ordered evaluation, without the task bootstrap.
- Null and validation permutations come from the existing
  `generate_matched_permutations` (unique, non-identity). Validation
  permutations that duplicate a calibration permutation are skipped, so the two
  sets stay disjoint.
- Operationalization of the H3 directional predictions: within one model,
  scenario, strategy and W, a metric's "first-to-alarm share" is the fraction of
  realizations in which its first post-`t_s` alarm is strictly earlier than every
  other validated metric's. Ties give no credit to anyone. A prediction ("CS and
  LS lead" for LIC-G; "CQ and RS lead" for LIC-S) holds in that model only if
  the two predicted metrics are validated and hold the two largest shares, both
  greater than zero.

## 1. Purpose and relation to the funded proposal

LIC-RQ-v2 established a paired history penalty at the level of single dialogues,
but it never used the proposal's streaming layer: no sliding window `W`, no
metric-specific drift detectors, no detector-strategy comparison and no window
ablation. This protocol connects the completed LIC-RQ-v2 records to that layer.

A deployed assistant serves a stream of conversation sessions. The replay treats
each completed LIC-RQ-v2 session as one stream item and builds streams in which
the operating regime changes from `fresh_prefix` (no accumulated assistant
history) to `dialogue_history` (accumulated history). This is a
**context-accumulation drift**, a specific form of the proposal's reasoning drift.
The input and task distribution is identical before and after the change, because
the same 48 source tasks appear once in each regime. Any change the monitor sees
therefore comes from history, not from harder questions. This is the
difficulty-confound control the proposal promised.

Proposal elements addressed:

| Proposal element | Addressed here |
| --- | --- |
| H1: metrics change after drift | Pre/post window distributions on ordered streams |
| H2: metric-specific detectors vs generic ADWIN | Three frozen strategies on identical streams |
| H3: gradual drift -> CS/LS first; sudden drift -> CQ/RS first | Sudden and gradual scenarios, first-alarm order |
| W in {10, 20, 30} ablation | All analyses repeated per W |
| Signed delay, false-alarm / early / post-threshold categories | Section 7 |
| H5 (ES + CQ) | **Not addressable**: ES is structurally unobservable in LIC-RQ-v2 |
| S2–S6 scale, 5K–10K queries | **Not addressed**; requires a separate paid protocol |

This is not Section 70 and does not reopen Section 69, RAT-v1, CDR or DBQR. It
does not alter any LIC-RQ-v2 confirmatory result.

## 2. What this analysis can and cannot show

The context effect is already known from the confirmatory LIC-RQ-v2 report.
The replay therefore produces **no new evidence that the effect exists**. It
measures something different: whether, and how quickly, the proposal's
window-plus-detector monitor would flag this known drift at a controlled false-alarm
rate, and which metric would flag it first.

All results are conditional on the 288 observed sessions (3 models x 48 tasks x
2 conditions). The stream timing is a construction, so an "early" alarm here means
before the declared change point in session order. It does not mean the warning
precedes failure within a dialogue. Results must be labelled **post-hoc
replay / method demonstration**, never confirmatory or prospective.

Disclosure: this draft was written after the investigator and the drafting agent
had seen the LIC-RQ-v2 confirmatory report, the CRL figure and the GPT-4o-mini
seven-panel trajectory figure. In particular, dialogue-history LS/SS appeared
visually higher than fresh-prefix LS/SS for GPT-4o-mini. To prevent tuning on
that observation, the detector direction stays the proposal's frozen
"higher is better, alarm on decrease" for every metric (Section 5). Upward
movement is reported descriptively only.

## 3. Frozen inputs

All inputs are verified by SHA-256 before any computation. The analysis aborts on
any mismatch.

| Input | SHA-256 |
| --- | --- |
| `results/lost_conversation_v2_science/data/records.jsonl` | `e313f1ce59854d2590efe5602cee4799a917664f6f389426412fe433ea1772ba` |
| `results/lost_conversation_v2_science/metrics.csv` | `7a9331e678b4f37b7fa9d62336a1634a3f1af42ce00c7e410324c68ba122c1c9` |
| `results/lost_conversation_v2_science/confirmatory/assessment.json` | `1be56fb6bb98e6f8348202dfea9804d2c2ed57582fbf78a1586e07790589bcf1` |

`metrics.csv` is the frozen semantic-contract rescore with 271 turn rows per
model and condition (48 sessions each). Raw records are read only for hash
verification and the empty-primary audit. They are never rescored.

## 4. Session-level values

One session is one (model, task, condition). Each metric gets one value per
session, fixed as follows:

| Metric | Session value | Missing when |
| --- | --- | --- |
| CQ | Final-turn CQ. An invalid or absent final answer is counted as **0**, matching the confirmatory primary rule. | never |
| RS | Final-turn RS (correct-primary C-subset only). | final primary incorrect or RS not measured -> NaN |
| CS | Mean of measured CS over all turns of the session. | no measured turn -> NaN |
| LS | Mean of measured LS over all turns of the session. | no measured turn -> NaN |
| SS | Mean of measured SS over all turns of the session. | no measured turn -> NaN |
| ES | Excluded. Zero measured values in the science run. | always |
| RQ6_mean | Excluded. Zero complete cases. | always |

Missing values are never imputed, set to zero or carried forward. As in the
existing `WindowMeanMetric`, a NaN does not enter the window and does not advance
that metric's detector. The proportion of sessions missing for each metric and
condition is reported. CRL is not a stream metric: it is defined only on paired
arms, and a single-arm monitor cannot observe it.

Sensitivity analysis (one only): CS/LS/SS taken from the final turn instead of the
session mean.

## 5. Monitor configuration

- Window: `WindowMeanMetric` with W in {10, 20, 30}. W=20 is the proposal's
  primary window. W=10 matches Sections 36/39. The results for all three are
  reported together; none is selected after seeing ordered outcomes.
- Detector warm-up: the code default of 20 updates, as in Sections 36 and 39
  (see Decision D1). Structural earliest-alarm index is about W + 20. With W=30
  it falls after the change point, so the gradual "early" category is impossible
  by construction. That is a finding of the ablation, not a defect.
- Direction: every metric is treated as higher-is-better. All detectors alarm on
  decrease, which is the existing code convention. No direction is flipped.
- The CQ raw-level DDM channel (`level="drift_raw"`) runs on the per-session 0/1
  CQ stream and is reported separately. It is never merged with windowed CQ.
- Strategies, from `detectors/assignment.py`, unchanged:
  - `distribution_based`: CQ, CS, RS, LS, SS -> CUSUM (windowed)
  - `proposal_mapping`: CQ->CUSUM, CS->ADWIN, RS->DDM, LS->Page-Hinkley, SS->ADWIN
  - `adwin_all`: every metric -> ADWIN

Only one sensitivity parameter per detector may vary. The frozen grids are
inherited from Sections 36 and 39 and listed from most to least sensitive:

| Detector | Parameter | Grid |
| --- | --- | --- |
| CUSUM | threshold | 0.15, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00 |
| ADWIN | delta | 0.05, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001 |
| DDM | drift level | 2, 2.5, 3, 3.5, 4, 5, 6, 8 |
| Page-Hinkley | threshold | 0.1, 0.15, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10 |

All other parameters keep their code defaults. CUSUM drift allowance stays 0.01.

## 6. Streams

Per model, the 96 sessions (48 tasks x 2 conditions) are arranged as follows.
Indices are 1-based.

**LIC-S (sudden).** Sessions 1–48 are the 48 `fresh_prefix` sessions in random
order. Sessions 49–96 are the 48 `dialogue_history` sessions in an independent
random order. Change point `t_s = t* = 49`.

**LIC-G (gradual).** Six blocks of 16 sessions. The number of
`dialogue_history` sessions per block is 0, 3, 6, 10, 13, 16, which uses all 48
from each condition. Assignment to blocks and order within blocks are random.
Transition start `t_s = 17`. Effective threshold `t* = 49`, where the cumulative
dialogue share crosses one half.

**Null (order-uninformative).** A uniform random permutation of all 96 sessions.
It is used for calibration and validation of false-alarm control.

**Stationary controls (descriptive).** Random orders of the 48 fresh-prefix
sessions only and of the 48 dialogue-history sessions only.

Frozen seeds and counts:

| Set | Seed | Streams per model |
| --- | ---: | ---: |
| Null calibration | 71042 | 2,000 |
| Null validation (disjoint from calibration) | 81042 | 2,000 |
| Ordered LIC-S realizations | 91042 | 1,000 |
| Ordered LIC-G realizations | 91043 | 1,000 |
| Stationary controls (each) | 91044 | 1,000 |
| Task-bootstrap sensitivity | 101042 | 1,000 |

The same index permutations are used for every metric, strategy and W within a
model, so all comparisons are paired. In the task-bootstrap sensitivity, 48
tasks are resampled with replacement before building each ordered stream, so that
a task's two sessions stay paired. This propagates task-sampling uncertainty.

## 7. Calibration and error control

For each (strategy, metric, detector, W) cell, one common parameter is selected
across all three models using **calibration null streams only**. The rule is:
the first, most sensitive grid value whose two-sided 95% Wilson upper bound on
the stream-level any-alarm probability is at most `alpha_cell` for every model.

Family per strategy and W: 5 metrics x 3 models = 15 cells. Hence
`alpha_cell = 0.05 / 15 = 0.00333`, a Bonferroni family-wise bound of 0.05.
2,000 null streams per set make this bound attainable: 0/2,000 gives a Wilson
upper bound of about 0.0019. The CQ raw DDM channel is calibrated by the same
rule, as a separate one-metric family per model (`alpha_cell = 0.05/3`).

The selected parameter is locked before validation or any ordered stream is
evaluated. It must independently meet the same bound on the validation null set
for every model. A cell that has no qualifying grid value, or fails validation,
is reported as "no specificity-controlled detector". It takes no further part in
H2/H3, and no grid expansion or relaxation is allowed.

## 8. Outcomes on ordered streams

For each model, scenario, strategy, metric and W, using the locked parameters:

- **Detection rate:** fraction of ordered realizations whose first alarm is at
  `t >= t_s`.
- **Pre-change alarm rate:** fraction with first alarm at `t < t_s`.
- **Categories:** `false_alarm` (t < t_s), `early_detection`
  (t_s <= t < t*, LIC-G only), `post_threshold` (t >= t*).
- **Signed delay** `D* = t_alarm - t*`, summarized as median and IQR over the
  realizations that detected the drift.
- **Stationary false-alarm rate** on the fresh-only and dialogue-only controls.
- **H1 (descriptive):** for each metric, the window mean over the last window
  before `t_s` versus the last window of the stream, with Mann-Whitney U and
  Cliff's delta. This is a conditional description, not an independent test,
  because these are reorderings of the same sessions.

H2 comparison, within each model, scenario and W: paired across realizations,
the difference between strategies in detection rate and in median signed delay
for each metric. A strategy is called better for a metric only when it has both
(a) a validated specificity-controlled detector, and (b) a higher detection rate
with a paired-realization 95% bootstrap interval excluding zero. Differences are
reported for all three models. A cross-model claim requires the same direction in
all three.

H3 lead analysis, within each model, scenario, strategy and W: for each non-CQ
metric m, the proportion of realizations in which m's first post-`t_s` alarm
strictly precedes windowed CQ's first post-`t_s` alarm. A metric with no alarm
has time infinity; ties do not count as leading. The proposal's directional
predictions are evaluated as stated:

- LIC-G: CS and LS lead the other metrics.
- LIC-S: CQ and RS lead the other metrics.

A metric is a **replay early-warning candidate** only if, under one strategy and
W, it (a) has a validated detector, (b) has detection rate >= 0.50, and (c) leads
CQ in >= 0.50 of realizations, in **all three** models. Cross-realization
consistency of the first-alarm order is reported as Kendall's W. Since all
realizations reuse the same sessions, Kendall's W measures robustness to order,
not reproducibility across samples.

## 9. Outputs

Everything is written under `results/lost_conversation_v2_stream_replay/`:

- `input_verification.json`: hashes, session counts, missingness per metric and
  condition, and the six empty-primary sessions.
- `calibration.json`: per-cell grid, null alarm counts, Wilson bounds, locked
  parameters and validation outcome.
- `ordered_outcomes.csv`: one row per model, scenario, strategy, metric, W and
  realization, giving first alarm time and category.
- `summary.json`: detection rates, delays, H2 contrasts, H3 lead proportions,
  candidate decisions, Kendall's W.
- Figures, each in PNG and PDF:
  - Windowed metric trajectories per model for LIC-S and LIC-G (median and
    10–90% band over realizations), with `t_s` and `t*` marked and missing-support
    counts shown.
  - First-alarm time distributions per metric and strategy.
  - H2 detection-rate and delay comparison.
  - W ablation (detection rate and delay vs W).

All plots show all five monitored metrics, including any metric without a
validated detector, and state that ES and RQ6_mean are unobservable.

## 10. Implementation and checks

- Script: `scripts/replay_lost_conversation_v2_streams.py` (new). It only reads
  the frozen inputs and writes the output directory above.
- The window and detector logic reuses `WindowMeanMetric`,
  `detectors/detectors.py` and `detectors/assignment.py` unchanged. A vectorized
  fast path is permitted only if a test proves it gives identical first-alarm
  times to the reference classes on fixed fixture streams.
- Tests (offline, no network/GPU): session-value construction and NaN handling,
  stream composition (each session exactly once, LIC-G block counts), seed
  determinism, calibration-rule selection on a synthetic fixture, and
  fast-path equivalence.
- The script prints progress (completed/total stream evaluations, percentage,
  rate, ETA) and is resumable per calibration cell.
- Before delivery: `ruff check src/ tests/`, `pytest`, `--quick` smoke.

Windows Miniconda Prompt command (after approval):

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\replay_lost_conversation_v2_streams.py
```

## 11. Permitted claims

Permitted, if supported:

- "Under a constructed context-accumulation drift, with family-wise false-alarm
  control, the proposal's window-and-detector monitor detects / fails to detect
  the change at delay D for metric m."
- Strategy and W comparisons, stated as conditional on the observed sessions.

Not permitted:

- Any new claim that dialogue history harms accuracy. That claim rests only on
  the LIC-RQ-v2 confirmatory analysis.
- Prospective or within-dialogue early-warning claims.
- Generalization beyond the three models, LiC math and this session count.
- Treating a replay candidate as validated without a fresh held-out stream.

## 12. Investigator decisions (resolved 2026-10-02: all recommendations accepted)

- **D1. Detector warm-up.** Recommendation: keep 20 (consistent with Sections
  36/39). Alternative: 10, which allows earlier alarms with W=20/30 in streams of
  96 sessions, but departs from the earlier calibrations.
- **D2. Primary window.** Recommendation: designate W=20 (the proposal's value)
  as primary and report W=10/30 as ablation. Alternative: no primary, all three
  equal.
- **D3. Session value for CS/LS/SS.** Recommendation: session mean (more
  support), with final-turn-only as the single sensitivity analysis.
- **D4. Ordered-realization inference.** Recommendation: conditional
  within-phase permutation as primary, task-bootstrap as sensitivity.

## 13. Observed result (2026-10-03; post-hoc replay, not confirmatory)

The frozen replay ran call-free to completion in both modes. Input hashes
matched. Outputs are under `results/lost_conversation_v2_stream_replay/`:
`summary.json` SHA-256
`7c376dc77ebdb8ae69874143d5e799936047ed72311efca576efeb10f901aa16`,
`calibration.json` `e0c109dddecd34f63482f881d76243e577c3f8147ba580c9361a72f7dcf559cc`,
`ordered_outcomes.csv` `2c27ede90ce17f87838ef324212469fe92657fdaceab3f10227aa113f373c9e7`.
`summary.json` embeds a generation timestamp, so its hash changes on rerun.

**Error control.** Calibration and validation passed for every ADWIN cell
(delta 0.05, the most sensitive grid value) and for most CUSUM and Page-Hinkley
cells. No common parameter qualified for RS-DDM at any W, or for CQ-CUSUM at
W=10. The raw CQ-DDM channel locked drift level 6. Every validated windowed cell
met the 0.0033 Wilson bound on the disjoint validation nulls; the raw channel,
which has its own 0.0167 family bound, had at most 5/2,000. Stationary-control
alarm rates were at most 0.001. The code-default CUSUM threshold of 0.15 alarmed on 90–94% of null
orderings, so only thresholds of 7.5–10 controlled false alarms.

**Detection (H1/H2/H3).** Under that control, the monitor almost never detected
the context-accumulation drift. At primary W=20, only windowed CQ-CUSUM detected
anything, and only for Mistral-7B: LIC-S 0.067 (W=30: 0.101), LIC-G 0.012,
with median signed delay about +44 (first alarm near session 93 of 96). For
GPT-4o-mini and Llama-4 Scout, every validated windowed detector had detection
rate 0.000 in both scenarios, except GPT CS-CUSUM in LIC-G (0.002–0.003). The
raw CQ-DDM channel detected 0.069 (LIC-S) and 0.045 (LIC-G) for GPT, 0.008/0.021
for Llama and 0.000 for Mistral. Pre-change alarm rates were zero. The task
bootstrap raised the largest rate to 0.226 (Mistral CQ-CUSUM, LIC-S, W=30),
still far below the 0.50 requirement.

No metric qualified as a replay early-warning candidate. The H3 directional
predictions (LIC-G: CS and LS first; LIC-S: CQ and RS first) held in no model,
strategy or W. No H2 strategy contrast showed a consistent cross-model
direction, because detection rates were at or near zero for all strategies.
Kendall's W was undefined where no metric alarmed.

**Direction of the non-CQ metrics (H1, descriptive).** Sudden scenario, W=20,
median over orderings of the last pre-change versus last stream window:

| Model | CQ | CS | LS | SS |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | 0.90 -> 0.70 (delta -0.20) | 0.98 -> 0.92 (-0.11) | 0.68 -> 0.75 (+0.31) | 0.93 -> 0.94 (+0.46) |
| Llama-4 Scout | 0.95 -> 0.85 (-0.10) | 0.98 -> 0.99 (+0.05) | 0.67 -> 0.68 (+0.05) | 0.98 -> 0.99 (+0.42) |
| Mistral-7B | 0.35 -> 0.05 (-0.30) | 0.97 -> 0.99 (+0.01) | 0.73 -> 0.77 (+0.20) | 0.998 -> 0.998 (+0.16) |

Values are window means, and delta is the median Cliff's delta. For SS, the
two-sided Mann-Whitney test was below 0.05 in 78% (GPT) and 64% (Llama) of
orderings. Under accumulated history, LS and SS therefore moved **upward**:
responses became more locally coherent and more mutually similar while accuracy
fell. A decrease-alarming detector cannot flag that by construction. This
direction was not tested confirmatorily and must not be retro-fitted into a
two-sided detector on these data.

**Sensitivity.** Final-turn CS/LS/SS values gave the same conclusion: no
candidate, the same CQ-CUSUM Mistral rates, and the same upward direction for
LS/SS, with smaller magnitudes.

**Interpretation.** With 48 sessions per regime and family-wise false-alarm
control, the proposal's window-and-detector layer cannot detect even a
confirmed paired accuracy loss of 0.21–0.29 in time, and RQEval process metrics
do not warn before CQ. This replicates the Section 36/38/39 stream finding in a
setting where task difficulty is held fixed, so it cannot be blamed on
difficulty confounding. It is a negative result about detector sensitivity at
this sample size, not evidence that the history effect is absent.
