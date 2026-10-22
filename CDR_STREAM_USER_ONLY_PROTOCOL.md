# CDR-Stream user-only history arm (DRAFT, call-free)

Written 2026-10-04 before any user-only response exists. It closes the
evidence gap identified in the self-review of the merged paper: the reviewer
question "is the Study 2 collapse just the intrinsic difficulty of tracking
state, or does the model's own text add harm?". The protocol becomes binding
on investigator approval. Paid acquisition needs a separate explicit approval.

## 1. Arm definition

The arm uses the same 90 held-out dialogues as the main run: the frozen manifest
`5351be89…`, the same instances, the same 100 turns and the same user texts.

At turn *t* the model receives:

- the same system prompt and response contract as the dialogue arm;
- the same packed-transcript format;
- every previous user turn 1…t−1, followed by turn t;
- **none of its own previous replies**.

The **only** difference from the dialogue arm is that the assistant replies are
absent. This is the long-horizon counterpart of Study 1's fresh prefix. The two
arms also differ in context length and message positions, which is stated.

| Setting | Value |
|---|---|
| Calls per turn | K=1, primary only (no repeats or paraphrases) |
| Temperature | 0.7 |
| Output ceiling | 2,048 tokens |
| Seeds | distinct per call, derived from this protocol's fingerprint |
| Truncation | the frozen main-run policy: kept verbatim, scored as missing; sensitivity analysis scores it incorrect |

No call depends on any model output. All 3,000 calls per model are therefore
independent and can run in parallel. The arm needs its own resumable checkpoint
(`results/cdr_stream_v1_user_only/data/records.jsonl`) and its own fingerprint.
The main-run checkpoint is never touched.

## 2. Models: decided at approval, before any data

| Option | Models | Estimated cost (list prices; not an invoice) |
|---|---|---|
| A (core) | GPT-4o-mini, Gemma 3 12B | about $4 |
| B (full) | A plus Llama-3 70B | about $60 |

The approved option is written into the config **before** acquisition. Its model
set defines the Holm families below and cannot be changed after any user-only
response is seen.

## 3. Frozen hypotheses

Arms:

- U: user-only;
- D: dialogue history (main run, primaries);
- C: verified state summary (control).

All tests use dialogue-level paired means over **turns 1–100**, valid turns only.
A missing value in either arm drops that turn.

The unit is the dialogue: 30 per model. The test is a one-sided sign-flip of the
mean across dialogues (20,000 draws). Each hypothesis family is corrected by
Holm across the approved models. Each mean is reported with a dialogue-bootstrap
95% CI (2,000 draws).

| ID | Statistic | Question | Prediction | Seed |
|---|---|---|---|---|
| **H-U1** | mean(C − U) | State-tracking load: does tracking state from the user history alone lower accuracy relative to verified state? | > 0 | 20261013 |
| **H-U2** | mean(U − D) | Own-reply penalty at long horizon: does adding the model's own replies lower accuracy further? | > 0 | 20261014 |

H-U2 is the long-horizon replication of Study 1.

**Decomposition (descriptive).** The total gap is split as

C − D = (C − U) + (U − D)

and reported per model, per scenario and per turn range (1–20, 21–100).

**Secondary analyses.**

- **H-U3, level-matched contrast inside U.** The same statistic as the Stage A
  level-matched contrast. Does user-only accuracy fall from early to late turns
  at matched difficulty? One-sided sign-flip; it is not in a Holm family.
- **Window 21–100.** H-U1 and H-U2 restricted to turns 21–100, labelled
  data-suggested.

**Dependency.** H-U1 needs the control arm. For Gemma 3 12B and Llama-3 70B it
is evaluated only after the control-arm completion protocol has run. Until then
it is reported as pending, never imputed.

## 4. Interpretation rules (frozen)

| Result | Wording allowed |
|---|---|
| H-U1 supported, H-U2 not | "The collapse reflects state-tracking load; the model's own replies add no detectable harm at long horizon." |
| H-U2 supported, H-U1 not | "The model's own replies, not the user history, drive the long-horizon collapse." |
| Both supported | "Both contribute; the decomposition gives their shares." |
| Neither supported | Report as such. No further arms are added. |

Nothing in this design separates history length from the number of updates
within U. That is not claimed.

## 5. What this protocol does not do

- It does not change the main run, its fingerprint, or Stage A/B.
- It does not add repeats or paraphrases, so CS, SS and RS are not available
  in U.
- It does not re-open any frozen result.
