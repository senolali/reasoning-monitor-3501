# CDR-Stream control-arm completion for Llama-3 70B and Gemma 3 12B (FROZEN 2026-10-04, investigator-approved; call-free)

Written on 2026-10-04, before any of these control responses was inspected or
scored. The investigator approved it on 2026-10-04 and it is now binding. Paid acquisition still needs a separate explicit approval. This
acquisition is a subset of the already committed Stage B
(`CDR_STREAM_ANALYSIS_PROTOCOL.md` §12) and changes nothing in it. It closes
RQ3 of `MERGED_PAPER_CLAIMS.md`.

## 1. Data to acquire

The calls are the `state_summary` control calls of the frozen main manifest
(`5351be89…`), made with the frozen runner, prompt and fingerprint
(`e5a9895b…`):

- Llama-3 70B: 3,000 calls. About 232 were already acquired before the budget
  pause; they have not been inspected and are reused through resume.
- Gemma 3 12B: 3,000 calls.

These calls do not depend on dialogue history, because each control call sees
only the solver state summary and the current turn. Acquiring them now cannot
change any later Stage B call.

Repeats, paraphrases and the remaining Llama turn-100 primary stay in Stage B.

The arm filter also acquires GPT-4o-mini's few missing control calls: 24
calls that were not completed at the pause. These are part of Stage B as well.
They do not enter the H-C1 family.

Cost is estimated from the acquired control calls: about 250 input and 210
output tokens per call. At list prices this is roughly $4–5 for Llama-3 70B and
under $1 for Gemma 3. It is an estimate, not an invoice.

The runner needs one opt-in change: an arm filter (`--arms control`). It
restricts which planned calls are submitted and leaves the fingerprint, seeds
and records unchanged. The change is unit-tested before use.

## 2. Frozen hypotheses

The dialogue-arm CQ of these models is already known from Stage A; their
control-arm CQ is not.

**H-C1 (mitigation; confirmatory).** For each model, the per-dialogue mean of
CQ_ctrl − CQ_dialogue over **all turns 1–100** is greater than 0.

- Valid values only; a missing value in either arm drops that turn.
- Unit: the dialogue (30 per model).
- Test: one-sided exact or Monte-Carlo sign-flip of the mean across dialogues
  (20,000 draws, seed 20261011).
- Multiplicity: Holm correction across the two models.
- Decision: the claim holds for a model if its Holm-adjusted p < 0.05.
- Reported alongside: a dialogue-bootstrap 95% CI (2,000 draws, seed 20261012)
  and the per-scenario means.
- GPT-4o-mini is reported with the same statistic as an existing, already-seen
  result, labelled so, and outside the Holm family.

**H-C1b (late horizon; secondary).** The same statistic over turns 21–100.

This window was suggested by the GPT figures, so it is labelled
data-suggested.

**Task-validity check (descriptive).** Report mean CQ_ctrl in S0-E versus S0-H,
and per regime in S1, S2 and S3, so that the control arm can be seen to follow
the difficulty schedule.

## 3. Exploratory monitoring validation (already frozen elsewhere)

Three frozen documents are applied unchanged to the new control arms:

- `CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md` (step 2 validation rule);
- `CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md` (Part 1 per-dialogue rule);
- Part 2 of the same addendum (6-dialogue aggregated rule).

Their outcomes are reported as validation of rules fixed on GPT data. They are
not tuned.

## 4. What this does not claim

The control arm is a packaged intervention: shorter context, no noise, and a
verified state. H-C1 therefore supports "a verified state summary preserves
accuracy". It does not support "the harm is caused by the model's own replies".

The design does not separate three sources of the harm:

- history length;
- the number of state updates;
- error accumulation.

## 5. Commands (Windows Miniconda Prompt, after approval)

```bat
python scripts\run_cdr_stream.py --config configs\cdr_stream_v1_main_asu.yaml --execute --arms control
python scripts\analyze_cdr_stream_control_completion.py
```
