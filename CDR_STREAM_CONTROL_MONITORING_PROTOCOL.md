# CDR-Stream control-arm monitoring (H2): GPT exploratory screen, Llama/Gemma validation

Frozen 2026-10-04, before any control-arm metric other than the descriptive
Stage A CQ_ctrl trajectories was computed. This is a separate protocol from
CDR_STREAM_ANALYSIS_PROTOCOL.md and does not change it.

## 1. Why

Stage A showed that dialogue-arm CQ collapses with accumulated history, so the
dialogue arm cannot test task-drift monitoring. The `state_summary` control
arm receives no history, follows the scheduled regimes, and its S0 dialogues
are true nulls with no context drift. It is therefore the stream on which the
proposal's detector layer (H2) can be tested.

## 2. Status of evidence

- **Step 1, exploratory screen.** GPT-4o-mini control arm. Its CQ_ctrl
  trajectories have already been seen in the Stage A figures, so this step is
  exploratory and cannot confirm anything.
- **Step 2, validation.** Llama-3 70B and Gemma 3 12B control arms. These have
  not been acquired, apart from about 8% of Llama's, and that partial Llama
  data has not been inspected. The rules below apply to them unchanged. Step 2
  needs the control calls to be acquired, which in turn needs budget and
  investigator approval.

## 3. Data and metrics

- **Records.** Control-arm primaries (`arm == control`) from the main
  checkpoint. Step 1 uses the pinned Stage A prefix (25,115 lines, `5e1b66cc…`).
- **Per turn:**
  - CQ_c: correctness under the frozen G1 parser; truncated responses are
    missing.
  - LS_c: the same NLI formula as the analysis protocol, applied to the control
    response.
  - ES_c: rqeval efficiency over the trailing 5 valid control primaries.
- CS, SS and RS do not exist in the control arm (K=1, L=0).
- No imputation.

## 4. Monitoring (unchanged machinery)

- **Signals.** `build_window_signal` with W ∈ {10, 20, 30}; W=20 is primary.
- **Detectors.** The reference detectors and grids, and the three strategies:
  - `distribution_based`: CUSUM for every metric;
  - `proposal_mapping`: CQ CUSUM, LS Page-Hinkley, ES EWMA;
  - `adwin_all`.
- **Direction.** Decrease.
- **Null calibration.** Per model, on that model's own S0-E and S0-H control
  streams (12 dialogues). Take the most sensitive grid value whose null
  first-alarm rate is ≤ 0.05, which with 12 nulls means 0 alarms.
- **Held-out FAR.** Leave-one-out over the 12 nulls.
- **Detection.** The first alarm at or after the first degradation change
  point c₁ and no later than c₁ + 33. An alarm before c₁ is a false alarm. The
  change points are S1 34, S2 51 and S3 34, so there are 18 drift dialogues
  per model.
- **Signed delay.** Alarm turn minus c₁.
- **H2 comparison.** For each metric, per-dialogue detection under each
  strategy minus detection under `adwin_all`. Report a paired bootstrap 95%
  interval over dialogues (2,000 draws, seed 20261006), plus median delays and
  held-out FAR.

## 5. Screening criterion ("worth pursuing"), fixed before computing

The GPT screen is **promising** if, at W=20, `distribution_based` or
`proposal_mapping` gives CQ_c all three of:

- detection ≥ 0.50 over the 18 drift dialogues;
- held-out LOO FAR ≤ 1/12;
- median signed delay ≤ 25 turns.

H3-type ordering, meaning whether LS_c or ES_c alarm before CQ_c, is reported
descriptively. It is not part of the screen.

## 6. Validation rule for step 2 (frozen now)

The same code, grids and α are applied, with no re-tuning, to each of
Llama-3 70B and Gemma 3 12B. Each model is calibrated on its own nulls. The
H2 claim is supported on the validation models if the step 1 criterion holds
in both of them. Results are reported for all models whether or not they
pass.
