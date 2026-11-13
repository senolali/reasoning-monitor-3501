# LHE-RQ-v1: per-conversation early warning of accuracy decline (DRAFT)

**Status: DRAFT, call-free. Not frozen. No model call is authorized.**
Investigator decision (2026-11-05): test again, with a corrected design, the
project's core hypothesis. Dataset: Sinha et al. Long-Horizon Execution.
Models: GPT-4o-mini, Gemma 3 12B and, if it passes its gates, Claude 5.5 Haiku.
The Informatica submission of *The Silent Cost of Conversation History in LLM
Reasoning* is on hold until this study reports.

This is a new protocol. It does not reopen, relabel or reanalyse the frozen
LIC-RQ-v2 or CDR-Stream results; their failed gates and negative monitoring
findings stay as recorded.

## 1. Hypothesis (the investigator's, stated operationally)

In independent multi-turn conversations with exact per-turn gold, accuracy
declines over turns. For each conversation separately, at least one
reasoning-quality signal that needs **no gold** detects that decline early:
no later than an accuracy-based detector that is allowed to see the gold,
while keeping false alarms on non-declining conversations under control.

## 2. Why the earlier tests could not answer this (post-hoc diagnosis)

Call-free inspection of CDR-Stream records (2026-11-05, exploratory) found five
design problems relative to this hypothesis:

1. **No healthy phase.** Dialogue-history accuracy started at about 0.6 and the
   per-dialogue collapse began at a median of turn 10 (rolling-5 accuracy
   <= 0.2; 41/90 dialogues within turns 5-10).
2. **Detector could not fire in time.** A W=20 window plus a 20-update warm-up
   puts the earliest alarm near turn 40, after the collapse had happened.
3. **Wrong target.** Detectors were scored against scheduled task-difficulty
   changes (turns 34/51/68), not against each dialogue's own accuracy decline.
4. **Wrong direction.** Turn-level discrimination of current correctness in GPT
   dialogue-history data: CS AUROC 0.65; SS 0.40 and LS 0.46 (both higher when
   wrong). Decrease-only monitors of SS/LS cannot use this signal.
5. **Too few samples for CS** (K=3).

Also: CQ, RS and ES need gold, so they cannot be deployed warnings. "Earlier
than CQ" is stricter than needed; matching a gold-informed detector without
gold is already useful. Both criteria are reported here.

## 3. Data

- Source: `arvindh75/Long-Horizon-Execution` (Hugging Face, Apache-2.0;
  Sinha et al., ICLR 2026). 100-key dictionary of five-letter words with
  integer values in [-99, 99]; 100 independent samples, each a key sequence
  with the exact running sum after every key.
- Stored read-only under `data/external/long_horizon_execution/` with
  SHA-256 hashes recorded before any call.
- Conversations: each sample is one conversation; conversations never share
  context and are analysed independently.
- Split (by sample id, fixed before any model output): calibration 30
  (ids 0-29), test 70 (ids 30-99). The pilot uses calibration ids only.

## 4. Task and prompt

- Turn t: the user gives `k` keys (turn complexity, fixed per model by the
  pilot, Section 7); the model looks up the values, updates the running sum
  and ends with `<answer>N</answer>`.
- Prompt: the authors' chain-of-thought template ("think step by step ...
  respond with the current total enclosed between <answer></answer> tags"),
  with the dictionary in the first message.
- Deviation from the source code (recorded): two of the authors' few-shot
  demonstrations contain wrong answers (`<answer>f10</answer>`; a group example
  answering 10 where 9 is correct). Our demonstrations are corrected.
- History is packed into the request by our code (the ASU gateway dropped native
  message history in CDR-v1); server-side history, search and prompt rewriting
  are disabled. Only the primary reply enters later history.
- Temperature 0.7 for all calls, so that repeated samples can differ.
  Output ceiling 1,024 tokens; truncated replies are kept and scored missing.

## 5. Per-turn measurements

At every turn, from the same packed context:

| Call | Count | Used for |
|---|---|---|
| Primary | 1 | enters history; CQ (target), LS |
| Same-context samples | 4 (K=5 with the primary) | CS, SS, answer entropy |
| Reworded user turn | 2 (L=2) | paraphrase agreement PA |

**Target (gold, evaluation only):**
- `step_correct_t`: the primary's update is correct given its own previous
  answer (Sinha et al.'s step accuracy). Primary target.
- `prefix_correct_t`: the primary's answer equals the true running sum.
  Secondary.

**Gold-free candidate signals (frozen before the main run):**
- CS: RQEval pairwise agreement of the K=5 final answers.
- SS: RQEval BERTScore similarity of the K=5 traces.
- PA: share of the L=2 reworded replies whose answer equals the primary
  answer (a gold-free analogue of RS).
- LS (secondary): RQEval NLI coherence of the primary trace. Short traces may
  make LS degenerate; this is checked in the pilot and reported.

RS (C-subset) and ES are computed for completeness but are not warning
candidates because they need gold. Missing values are never imputed.

## 6. Events, detectors and false alarms

**Healthy start.** A conversation is evaluable if at least 4 of its first 5
turns are step-correct. Others are reported and excluded from detection
analyses.

**Decline onset (gold).** τ = the first turn t > 5 such that at most 3 of the
5 turns t..t+4 are step-correct; τ is set to that window's first error.
Conversations without such a window within the horizon are **non-declining**.

**Monitor.** For each gold-free signal m: a one-sided CUSUM on per-turn values
standardised by the mean and SD of healthy (pre-onset) turns in the
calibration set, with a short per-conversation reference (turns 1-5) and **no
additional warm-up**. Direction (increase or decrease) and the CUSUM threshold
are fixed on calibration conversations, then frozen. A combined rule
(alarm when any of CS, SS, PA alarms) is pre-declared as one extra candidate.

**Oracle comparator.** The same CUSUM on step correctness, with gold, thresholded
on calibration to the same false-alarm rate.

**Scoring per conversation (test set).**
- False alarm: an alarm in a non-declining conversation, or before τ − 10.
- Detection: first alarm in [τ − 10, τ + 5].
- Lead over oracle: oracle alarm turn minus signal alarm turn.

**Calibration target:** false-alarm rate <= 0.10 per conversation on the
calibration set.

## 7. Pilot (paid; separate approval)

Purpose: choose turn complexity `k` per model so that the decline falls inside
the horizon, and check signal health. Calibration ids 0-9 only, primary
replies only for `k` selection, horizon 60 turns, `k` in {1, 2, 4}.
Selection rule (frozen before the pilot): the smallest `k` for which the median
onset τ lies in turns 15-45, at least 70% of conversations have a healthy start,
and at most 50% are non-declining. A short K/L health check on two
conversations confirms that CS, SS and PA are not constant.

## 7a. Pilot-1 outcome and recorded deviation (2026-11-06)

Pilot-1 ran on 2026-11-06 after explicit approval: 3 one-call production
probes (all accepted) and 5,400/5,400 calls (0 failures, 40 min;
`results/lhe_rq_v1_pilot/`, `k_selection.json`). No (model, k) satisfied the
frozen rule. With brief chain of thought, GPT-4o-mini and Claude 5.5 Haiku made
0 step errors in 600 turns at every k in {1, 2, 4}; Gemma 3 12B made 0, 8 and
25. The task was too easy to produce a decline, which is consistent with Sinha
et al., who report the sharp decline mainly without chain of thought.

Deviation (investigator decision, option C): pilot-2 on the same calibration
conversations, horizon 100, primary replies only, with chain of thought at
k in {8, 16} and a **direct-answer mode** (the authors' non-chain-of-thought
template) at k in {4, 8, 16}; GPT-4o-mini and Gemma 3 12B only (Claude 5.5
Haiku left out). Same selection rule, applied per (model, mode). In direct mode
no reasoning is written, so SS and LS are undefined; CS and PA remain the
gold-free candidates. Config: `configs/lhe_rq_v1_pilot2_asu.yaml`.

## 7b. Pilot-2 outcome and model screening (2026-11-06)

Pilot-2 (10,000/10,000 calls, 0 failures; `results/lhe_rq_v1_pilot2/`): direct-
answer mode failed from turn 1 at every k (no healthy start) and is dropped.
With chain of thought, Gemma 3 12B at k=16 had 10/10 healthy starts and 9/10
declines (tau 49-95, median 72); GPT-4o-mini at k=16 declined in 3/10. No unit
met the 15-45 median window, which had been set for a 60-turn horizon.
An exploratory, post-hoc look at single-call signals in the 9 Gemma declines
found no warning: output tokens drop only at tau (contemporaneous), and a
pre-onset LS rise vanished against still-healthy conversations at the same
turns (4/9, p=0.75). CS, SS and PA were not measured (K=1, L=0).

Investigator decision: a single model is insufficient for generalisation; at
least two more models are required. Pilot-3 screens GPT-4o-mini at k=32 and
four new non-thinking candidates from different vendors (GPT-4.1-mini, Nova
Lite, Mistral Large, Gemini Flash 3.1 Lite) at k in {16, 32}, chain of thought,
horizon 100, calibration conversations 0-9. All one-call production probes were
accepted on 2026-11-06. The selection rule is updated **before** pilot-3 for the
100-turn horizon: median tau in 15-85, >= 70% healthy starts, <= 50%
non-declining. The main-run success criterion will require at least three
retained models. Config: `configs/lhe_rq_v1_pilot3_asu.yaml`.

## 7c. Pilot-3 interrupted by the ASU monthly budget (2026-11-06)

Pilot-3 stopped at 4,178/9,000 calls (0 failed calls before the stop) when the
ASU project budget reached 100.44% (HTTP 429; resets at the start of the month).
Records are saved and the run is resumable with the same command. Conversations
were observed to turns 35-51. The pilot analysis now censors unobserved turns
(they are never counted as errors) and no unit may pass the rule while any
conversation is censored. Interim, on observed turns only (not a selection):
GPT-4o-mini k=32: 9/10 healthy, 7 declines (tau 8-23); Mistral Large k=16:
8/10 healthy, 3 declines so far; GPT-4.1-mini and Gemini Flash 3.1 Lite: no
declines; Nova Lite: too few healthy starts. Completion requires budget.

## 8. Hypotheses and analysis (test set; per model; unit = conversation)

- **H1 (detection).** For signal m, the detection rate exceeds that of the same
  monitor run on the signal with turns randomly permuted within each
  conversation (10,000 permutations), one-sided. Holm over the four candidates
  (CS, SS, PA, combined) within model.
- **H2 (no later than the oracle).** Among conversations where both alarm,
  the median lead over the oracle is >= 0 (sign test, one-sided).
- **Secondary.** Turn-level AUROC of each signal for current and next-5-turn
  correctness, within conversation; trajectories aligned at τ (mean ± CI of
  each signal from τ − 20 to τ + 10) in PNG and PDF, with support counts.
- The pre-declared success criterion of the investigator's hypothesis: H1
  holds for at least one candidate in at least two of the three models, with
  test-set false-alarm rate <= 0.15. Anything else is reported as it is.

Models are reported separately; no pooled claim.

## 9. Models and gates

| Model | ASU id | Gate |
|---|---|---|
| GPT-4o-mini | `openai/gpt4o_mini` | 1-call production probe |
| Gemma 3 12B | `aws/gemma3_12b_it` | 1-call production probe |
| Claude 5.5 Haiku | `aws/claude5_5_haiku` | 1-call probe + repeat-diversity gate (>= 0.80, validity >= 0.90, no truncation); fallback `aws/gpt-oss-120b` |

All three appear in the ASU catalogue of 2026-11-05 (free catalogue request).

## 10. Budget (to be computed per model before approval)

Approximate per conversation at k=1 and 60 turns: 420 calls; prompts grow
from about 1k to about 6k tokens. Main run (100 conversations, of which the 30
calibration conversations double as the threshold set): about 42,000 calls per
model. Exact token and dollar estimates are produced by a call-free estimator
from the frozen prompts before any approval request.

## 11. Call-free work before freezing

1. Hash the dataset; build conversation and prompt generator; corrected
   few-shot block.
2. Resumable runner (progress, calls/min, ETA), packed history, K/L calls,
   strict and tolerant `<answer>` parsing.
3. Scorers: step/prefix correctness, CS, SS, PA, LS via RQEval with the real
   NLI/BERTScore preflight; no silent fallback.
4. Monitor and evaluation code with synthetic tests (known onset, known
   precursor) to show that the pipeline can detect a planted early signal.
5. Fixture replay with a simulated client; cost estimator.
