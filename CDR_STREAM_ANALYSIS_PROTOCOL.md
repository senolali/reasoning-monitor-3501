# CDR-Stream main-run analysis protocol (DRAFT, call-free)

Status: **FROZEN 2026-10-04 by investigator approval** (Section 11 decisions
accepted as recommended; Stage A/B split in Section 12). Written before any
main-run accuracy, metric value or alarm was inspected. It operationalizes
CDR_STREAM_DESIGN §5–§6 and the truncation policy of §22. Code:
`src/reasoning_monitor/cdr_stream_analysis.py`,
`scripts/analyze_cdr_stream.py`, and tests in `tests/test_cdr_stream_analysis.py`
that use synthetic records only.

## 1. Inputs and integrity

- **Records.** `results/cdr_stream_v1_main/data/records.jsonl`. Analysis starts
  only when all 54,000 planned calls are present under fingerprint `e5a9895b…`.
  The SHA-256 of the records is written into every output.
- **Manifest and mappings.** The manifest (`5351be89…`) and the mappings
  (`4ab2af3d…`) are hash-checked.
- **Unit of inference.** The dialogue is the unit of inference; turns are not
  independent.

## 2. Per-turn scoring (dialogue arm)

Each (model, dialogue, turn) cell has 1 primary, K−1=2 repeats and L=2
paraphrases. A response is **valid** if `ok` is true and it is not truncated.

| Metric | Definition | Missing when |
|---|---|---|
| CQ | primary `correct` under the frozen G1 semantic parser; contract-invalid (no marker) counts as incorrect | primary truncated |
| CS | mean pairwise answer agreement over the K=3 runs | any run invalid |
| SS | mean pairwise BERTScore F1 (distilbert-base-uncased, unrescaled, clipped to [0,1]) over the K=3 reasoning traces | any run invalid or empty trace |
| LS | 1 − share of NLI contradictions between consecutive reasoning steps (cross-encoder/nli-deberta-v3-small, rqeval formula) | primary truncated or fewer than 2 steps |
| RS | C-subset: share of L paraphrase answers equal to gold, only when the primary is correct | primary incorrect (excluded by design) or any paraphrase invalid |
| ES | rqeval efficiency (`_ESMetric`) over the trailing 5 valid primaries (CQ and output tokens) | primary truncated or warm-up (fewer than 5 valid primaries) |

The **control arm** gives only CQ_ctrl, the `state_summary` primary
correctness, with the same truncation rule.

Scorers are the real local NLI and BERTScore models. A fallback scorer is an
error, never a silent substitute. Missing values are never imputed.

**Sensitivity analysis (prespecified, §22).** CQ, and therefore ES, is rescored
with truncated cells counted as incorrect. Truncation counts are reported per
model, scenario, regime and turn.

## 3. Change points and windows

Change points are fixed by the generator schedules:

- S1: 34 (E→M) and 68 (M→H);
- S2: 51 (E→H);
- S3: 34 (E→H) and 67 (H→E, a recovery);
- S0-E and S0-H: none.

**Degradation transitions:** S1-a@34, S1-b@68, S2@51, S3-a@34.
**Recovery transition:** S3-b@67.

## 4. H1: degradation under drift

For each dialogue and transition at change point c, take the mean of the
valid values in two windows:

- pre-window: turns c−20 … c−1;
- post-window: turns c … c+19.

Δ is post minus pre.

- **H1a, task drift in the dialogue arm.** Δ < 0 for each metric and each
  degradation transition. That is 6 metrics × 4 transitions = 24 one-sided
  tests.
- **H1b, context net of task.** ΔCQ(dialogue) − ΔCQ(control) < 0, the extra
  decline under accumulated history, for each of the 4 degradation
  transitions.
- **Test.** An exact or Monte-Carlo (20,000 draws, seed 20261004) sign-flip
  test of the mean per-dialogue Δ. Signs are flipped within model strata, so
  the primary test pools the three models with model-stratified flips.
  Per-model results are secondary and descriptive.
- **Missing support.** A dialogue with no valid value in either window is
  missing for that test, and support is reported.
- **Multiplicity.** Holm correction within H1a (24 tests) and within H1b
  (4 tests).
- **Within-arm, level-matched contrast (secondary).** For each dialogue,
  compare CQ for turns ≤ 50 and turns > 50 at the same difficulty level. Take
  the mean of the level-stratified differences, weighting each level by its
  smaller cell count. Test it with the same sign-flip test.

## 5. Monitoring layer (H2, H3)

**Signal.** The detector input is the existing `build_window_signal`: a
trailing window mean over valid values, with W ∈ {10, 20, 30} and W=20 as the
primary window. Missing turns neither enter nor advance the window, which is
the WindowMeanMetric C-subset behaviour. ES uses the rqeval efficiency over
the window.

**Detectors and strategies.** The unchanged reference detectors and grids
(`stream_replay.GRIDS`) are used, with the three strategies
`distribution_based`, `proposal_mapping` and `adwin_all` from the assignment
layer. The direction is the proposal's primary direction, a decrease.

**Detector warm-up (property of the unchanged detectors).** Each detector
ignores alarms during its first 20 window values; DDM ignores its first 30.
The earliest possible alarm is therefore about turn W+20, which is turn 40 at
W=20. With W=20, the warm-up reference for S1-a and S3-a (turns 20–40) spans
the change at 34. This can reduce sensitivity to the first transition. With
W=10 the reference is turns 10–30, which is clean. The detectors are NOT
modified; this limitation is reported alongside the W ablation.

**Null calibration (deviation from §6, see §10).** All main-run S0 dialogues
(S0-E and S0-H, pooled across models; 36 dialogues) form the null. For each
(metric, detector, W), the grid value chosen is the most sensitive one whose
pooled null first-alarm rate is ≤ α_cell = 0.05.

- **Held-out FAR.** It is estimated by leave-one-dialogue-out. Each S0
  dialogue is evaluated with a threshold calibrated on the other 35.
- **Reporting.** FAR is reported per metric and as the any-metric rate, with
  Wilson intervals.
- **Drift dialogues.** These use the threshold calibrated on all 36 nulls.

**Detection** is defined at the first degradation change point c₁ of each
drift dialogue:

| First alarm | Category |
|---|---|
| before c₁ | false alarm |
| in [c₁, c₁ + 33] | detection |
| none | miss |

The signed delay is the alarm turn minus c₁. S1-b@68 is reported separately,
counting the first alarm at or after 68.

- **H2.** Compare the three strategies on identical scored streams: detection
  rate, median signed delay and held-out FAR. Report a paired bootstrap over
  dialogues (2,000 draws, seed 20261005) for differences against `adwin_all`.
- **H3.** For each S1 and S2 dialogue, under `proposal_mapping` with W=20, take
  each metric's first-alarm turn at or after c₁, with no alarm counted as +∞.
  Report:
  - each metric's share of dialogues in which it alarms strictly before CQ;
  - the first-to-alarm shares;
  - Kendall's W across dialogues.

  The proposal's predictions, frozen from the replay protocol, are that CS and
  LS lead in gradual drift (S1) and CQ and RS lead in sudden drift (S2). A
  prediction "holds" under the replay rule (`prediction_holds`). H3 is
  descriptive unless at least 10 dialogues per model alarm on CQ.

**Two-sided secondary arm (D5).** LS and SS are also monitored for an
increase, by feeding 1 − value to the same detectors. These detectors are
calibrated separately on the same nulls and reported apart from the primary
results.

## 6. H4 and H5 (descriptive)

- **H4.** A model × metric matrix of mean Δ (H1a windows), with dialogue
  bootstrap intervals.
- **H5.** A per-dialogue Spearman correlation of ΔCQ and ΔES, and the share of
  dialogues with ΔCQ < 0 and ΔES < 0.

## 7. Figures (PNG and PDF)

- **Turn trajectories.** For each model × scenario: the mean per turn of the
  six metrics, plus CQ_ctrl, with change-point lines. Missing-coverage panels
  are drawn under each metric.
- **Alarm raster.** First alarms per dialogue, by metric and strategy.
- **Null calibration.** Null first-alarm curves over the grids.

Every figure states n dialogues and the coverage of each metric, and whether
each signal is contemporaneous with the change or precedes it.

## 8. What will not be done

- No threshold tuning on drift dialogues.
- No exclusion of dialogues after inspection.
- No imputation.
- No reinterpretation of S0 alarms as findings.
- No new metric fitted to the data.

## 9. Outputs

`results/cdr_stream_v1_main/analysis/`:

- `turn_metrics.csv`
- `h1.json`
- `monitoring.json`
- `h3.json`
- `descriptive.json`
- `manifest.json`, with input hashes and scorer revisions
- the figures

## 10. Deviations to record when this is frozen

The design (§6) calibrated thresholds on calibration-split S0 dialogues, and
the staged gate G3 was to run those calibration streams. G3 was never run, so
no calibration-split stream data exist. This protocol instead calibrates on
the main-run S0 dialogues, with leave-one-out FAR evaluation.

The consequences:

- The nulls are 36 dialogues, not a separate set.
- FAR resolution is about 0.03.
- A per-cell α of 0.05 cannot be Bonferroni-corrected across metrics, so the
  any-metric FAR is reported instead.

## 11. Decisions for the investigator before freezing

1. Approve the null calibration of §5 (pooled S0, α_cell = 0.05, LOO FAR), or
   choose per-model calibration. Per-model calibration has 12 nulls per
   model, so FAR resolution would be about 0.08.
2. Approve pooled-model primary tests with model-stratified sign flips. The
   alternative is per-model primary tests, with only 6 dialogues per
   scenario, so the minimum one-sided p is 1/64.
3. Approve the H1 window of 20 turns either side of each change point.

## 12. Staged analysis after the budget pause (frozen 2026-10-04)

The ASU budget ran out at 25,039/54,000 calls (46.4%). The runner schedules
primaries first, so the acquisition is uneven:

- **Primary chains.** Complete for all 90 dialogues. The one exception is a
  single missing Llama turn-100 primary.
- **GPT-4o-mini.** Almost complete: 17,880 of 18,000 calls.
- **Llama-3 70B.** Repeats, paraphrases and control calls exist for only
  about 8% of turns.
- **Gemma 3 12B.** Primaries only.

Packed history consists of primaries only. Completing the remaining calls
later therefore does not change any history and gives the same design.

**Stage A, now (records SHA-256 `5e1b66cc…`).**

- CQ, LS and ES for all three models. These depend only on primaries and are
  final.
- All six metrics and CQ_ctrl for GPT-4o-mini.
- Llama-3 70B's partial repeats, paraphrases and control calls are excluded
  from Stage A, so incomplete acquisition is never mixed with missingness.
- A metric that is not acquired is "not acquired". It is not "missing" in the
  scoring sense: its dialogues are excluded from the tests and from the
  monitoring for that metric. They are not counted as no-alarm.
- Stage A runs every analysis in Sections 4–6 that is supported by the
  available model-metric pairs:
  - H1a for CQ, LS and ES in three models, and for CS, SS and RS in GPT only;
  - H1b in GPT only;
  - null calibration per metric on the S0 dialogues of the models that have
    that metric. For CS, SS and RS this means 12 GPT nulls, so the FAR
    resolution is about 0.08.
  - H3 restricted to the available metrics, labelled incomplete.

**Stage B, later.** CS, SS, RS and CQ_ctrl for Llama-3 70B and Gemma 3 12B,
followed by the complete analysis of Sections 4–6.

**Commitment, made before Stage A results.** Stage B will be completed with
the same frozen runner whenever budget allows. This does not depend on any
Stage A result. Stage A results are reported as Stage A, and the final report
presents Stage A and Stage B together.
