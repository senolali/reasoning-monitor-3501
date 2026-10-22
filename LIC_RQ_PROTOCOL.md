# LIC-RQ-v1: preparation protocol

Status: call-free design, 2026-10-01. Not an execution authorization, independent
validation result, or completed preregistration. Old gates and CDR/RAT holdouts
remain closed and unchanged.

## Question and source

How do six RQEval dimensions evolve during information revelation, compared
with a fresh context containing identical user information? Do pre-outcome
measurements add predictive information beyond turn, information fraction and
context length? Early warning is a question, not an entry condition.

- Paper: https://arxiv.org/abs/2505.06120
- Repository: https://github.com/microsoft/lost_in_conversation
- Data card: https://huggingface.co/datasets/microsoft/lost_in_conversation
- Commit: c865793fe34a929d316119b0451d01bd9183bcfd
- SHA-256: 6667a3ab2b013d51d7808ac913e24b2db843a0288da6a8c5cbddec89d6163f1e

The public filename says 600 but this snapshot contains 627 instructions,
including 103 math tasks and 585 math prefixes. Other tasks are inventoried,
not selected for study. Math is derived from GSM8K. This is only an inventory:
no paid sample, independent holdout designation or statistical power claim.
Audit prior project exposure before allocating exploration/confirmation groups.
All shards, paraphrases and repetitions of a source task stay in one split.

## Adaptation, not direct replication

The paper's simulator chooses shards adaptively and stops on success or
exhaustion of information. Our proposed adaptation makes these changes explicit:

1. Reveal one shard per turn in source ID order, independent of assistant replies.
   This controls information scheduling but sacrifices conversational naturalness.
2. Continue to the last shard regardless of correctness. This avoids selective
   early stopping. Report initial and final success separately; do not claim to
   reproduce the paper's original stopping rule.
3. Compare dialogue_history with fresh_prefix. Only the former carries prior
   primary assistant replies. Both see identical user information: no future
   shards, full original question, solution or gold answer.
4. Use a common packed-user wrapper for ASU. This interface adaptation needs a
   bounded integration check, not a positive-effect gate.

This contrast includes assistant-history length. It is not a pure elapsed-time
effect, natural human-dialogue study, or proof of intrinsic model deterioration.
A true-state oracle is not the matched control.

## Measurement

K=5 includes primary index zero; four repeats are independent same-context
probes. L=2 equivalent paraphrases must be reviewed and frozen before output.
L=2 follows the latest acquisition convention, not the proposal's L=3. Probes
never enter primary history; never select the best/correct repeat to continue.

Allow clarification; do not force numbered reasoning or final numbers on every
turn. Classify response type without gold. Validate extraction from natural
answers. Empty parsed strings must not count as answer agreement.

- Partial prefixes have unreviewed answerability, not proved unanswerability or
  zero quality. CQ/RS/ES are unavailable until a prefix-specific scoring rule is
  independently reviewed. Eligibility cannot depend on observed correctness.
- With full information, answer attempts are scorable. Separately, final task
  success counts no valid answer as noncompletion, not as removal from the
  denominator. Report API/format failures separately.
- CS needs valid extracted answers in K samples; agreement among clarification
  questions is not answer consistency. LS/SS require genuine public reasoning
  steps/traces. No trace means missingness, not zero or perfect quality.
- Real NLI and BERTScore must validate before execution; loading real NLI alone
  does not prove validity on arithmetic explanations.
- RS retains its correct-primary subset. ES retains a causal five-turn token
  window and warmup. Track token availability separately from gold availability;
  the historical integer-only score_cell cannot be reused unchanged.

Show all six panels, per-task views, support counts and missingness. Plot both
turn and revealed-information fraction, with source-task-level uncertainty.
22 tasks have fewer than five turns, so ES warmup limits coverage. Short
trajectories also limit within-dialogue change detection.

RQ6_mean is the equal-weight mean only when all six scores are measured, finite
and in [0,1]. Never zero-fill, forward-fill or change the denominator. It is a
gold-dependent, complete-case-selected descriptive summary (especially due to
RS). Show coverage and never use it as an independent online warning feature.

## Analysis and finite stopping plan

Before model output, choose sample size, grouped split and a small fixed set of
comparisons in an execution manifest. This preparation is not a complete
preregistration. Primary outcomes are paired final task success and observed
six-dimensional profiles. Future-outcome prediction uses earlier predictors
only; separate gold-dependent analyses from label-free ones. Compare against
turn + information fraction + context length. Fit thresholds/normalization on
exploration groups only. Correct the frozen family of metric tests (e.g. Holm),
cluster by source task and do not treat turns/K repeats as independent tasks.

At most one bounded integration check before a frozen campaign. Engineering
repairs preserve original records; no chain of significance pilots or dataset
changes until a desired signal appears. A null conclusion requires uncertainty
and power assessment, not just an insignificant p-value.

## Cost and remaining work

The WHOLE INVENTORY at two models/two conditions/K=5/L=2 would require
585 * 2 * 2 * 7 = 16,380 logical target-model calls, capped at 33,546,240 output
tokens at 2,048 per call. This is NOT an approved campaign or a currency cost.
Input tokens, retries, auxiliary extraction/classification and paraphrase
preparation are extra. A technical check must be much smaller.

Next: semantic/overlap review, frozen paraphrases, sample size/split/budget,
validated response classification/scorers, a primary-only resumable runner
with progress/ETA, then explicit paid-execution approval. Reuse valid six-metric
components without changing historical runners or importing their per-turn
integer-answer assumption. No further scientific-effect pilot is required.

## Input-audit update (2026-10-01)

The next call-free stage is complete; see LIC_RQ_AUDIT.md and the reproducible
JSON artifacts in results/lost_conversation_v1_audit. Exposure/first-pass review
leave 86 provisional candidates. This does not replace independent review or
grant partial-prefix scoring eligibility. Only seven non-quarantined tasks have
candidate sufficiency before the last shard. Dense six-metric curves and their
mean are therefore not guaranteed by this source. Preserve missingness and show
coverage, not fabricated values. No paid sample or execution manifest is frozen.

## Windows Miniconda CMD

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\prepare_lost_conversation.py
python scripts\prepare_lost_conversation.py --write
```

Default mode prints the audit with no writes. --write saves a verified inventory,
prefix review queue and budget report. Changed artifacts are not overwritten.
There is no --execute flag and no model client in this preparation tool.
