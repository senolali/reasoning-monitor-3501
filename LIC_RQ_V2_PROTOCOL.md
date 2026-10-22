# LIC-RQ-v2 frozen study protocol

Status: call-free manifest frozen on 2026-10-01. Model execution remains disabled.

## Dataset and question

The active dataset is the pinned Microsoft Lost in Conversation **math** subset:
103 source tasks and 585 ordered information shards. The study asks whether
carrying prior assistant responses changes final solution success relative to a
fresh prefix containing the same revealed user information, and whether earlier
RQEval measurements predict final failure.

This is a fixed-order adaptation, not an exact replication of the original
adaptive user-simulator experiment. Every task continues through its final shard.

## Frozen design

- Models: GPT-4o-mini, Llama-4 Scout and Mistral-7B through ASU.
- Conditions: `dialogue_history` and matched `fresh_prefix`.
- Sampling: K=5 same-context responses including the primary; L=2 frozen
  paraphrases; temperature 0.7; maximum 2,048 output tokens.
- History: only the prospectively fixed primary response enters later history.
  Repeats and paraphrase responses never enter the conversation.
- Sample: four disjoint technical-integration tasks, followed by one fixed
  48-task science sample. Selection is hash-based and does not use model output.
- Missing values are never converted to zero, carried forward or averaged with a
  changing denominator. `RQ6_mean` exists only for complete six-metric cases.

## Outcomes and metric eligibility

The primary outcome is paired final-turn success under dialogue history versus
fresh prefix. The independent unit is the source task, not a turn, repetition,
condition or model.

Most LIC math prefixes deliberately omit information required for the final
numeric answer. Therefore CQ, RS and ES are normally unavailable before the
gold-eligible final prefix. Their absence is design missingness, not model error.
Pre-final early-warning candidates are CS, LS and SS when their input contracts
are satisfied. Final-turn CQ is the predicted outcome; final-turn RS remains
restricted to correct primary responses. ES keeps its causal five-turn warmup.

All seven panels (six metrics plus `RQ6_mean`) must be produced in PNG and PDF,
with support counts and explicit gaps. The primary early-warning analysis tests
whether pre-final CS/LS/SS add information beyond turn, revealed-information
fraction and context length. Holm correction applies to the frozen metric family.
No unavailable metric may be described as having failed to warn.

## One integration gate, then one study

The four-task integration stage is technical only. It verifies packed history,
cache/resume behavior, answer extraction, real NLI/BERTScore, K/L isolation,
progress/ETA and plots for all three models. Effect direction, significance or
number of errors cannot pass or fail this gate. Repairs must preserve raw records.

Passing integration does not automatically authorize the science run. The
48-task run needs one new explicit approval and cannot be resized after observing
integration effects. There will be no sequence of effect-seeking mini-pilots.

## Frozen budget

| Stage | Tasks | Prefixes per model-condition | Logical calls |
| --- | ---: | ---: | ---: |
| Technical integration | 4 | 26 | 1,092 |
| Science campaign | 48 | 271 | 11,382 |
| Total | 52 | 297 | 12,474 |

The maximum-output-token ceiling is 25,546,752. This is not a currency estimate;
input tokens, retries and local scoring time are additional.

The machine-readable contract is
`results/lost_conversation_v2_manifest/execution_manifest.json`. Reproduce it
without API calls in Windows Miniconda Prompt:

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\freeze_lost_conversation_v2.py
```

## Integration runner readiness

The resumable integration runner is
`scripts/run_lost_conversation_v2_integration.py`. Its call-free fixture replay
passes request isolation, checkpoint reuse, strict parsing, metric missingness
and PNG/PDF plotting. The real local NLI and BERTScore preflight also passes;
reports are saved under `results/lost_conversation_v2_integration/`. The runner
prints completed/total calls, percentage, calls per minute and ETA after every
new record. On 2026-10-01 the investigator explicitly authorized only this
1,092-call technical integration. The 48-task science campaign remains locked.

## Integration result

The integration completed on 2026-10-02 with 1,092/1,092 successful API calls,
zero truncations and equal 364-record model arms. The original strict final-line
parser was too narrow: validity was 36.5% for GPT-4o-mini, 82.4% for Llama-4
Scout and 3.0% for Mistral-7B. A separate call-free semantic-contract parser,
which still requires an explicit final-line marker but accepts observed numbering,
units and nested status variants, recovered validity to 100%, 100% and 94.2%.
Raw records and strict scores remain unchanged.

Final paired success on four technical tasks was 1.00/1.00 for GPT, 0.75/0.75
for Llama and 0.00/0.50 for Mistral (dialogue history/fresh prefix). Mistral's
dialogue arm remained in `NEED_MORE_INFORMATION` at all four final turns despite
the matched fresh-prefix arm attempting answers. This is exploratory integration
evidence, not an effect estimate.

ES and complete-case `RQ6_mean` were structurally unobservable. LS and SS often
remained high during Mistral's unresolved state. An auxiliary, non-RQEval
resolution-readiness gap was therefore defined during integration: fresh-prefix
answer readiness minus dialogue-history readiness at matched information. Mistral
had positive final gaps on 4/4 tasks and pre-final gaps on 3/4, with median lead
two turns. GPT and Llama showed transient gaps but recovered by the final turn.
The persistence/non-recovery of this gap is an exploratory candidate that must be
frozen and tested only in the untouched science sample. The integration is now
locked and the science campaign remains unauthorized.

## Frozen science campaign and CRL

The untouched 48-task science campaign was prepared call-free on 2026-10-02.
Its configuration is `configs/lost_conversation_v2_science_asu.yaml`; the
machine-readable preflight is
`results/lost_conversation_v2_science_preflight/assessment.json`. All source,
manifest, integration-record, acquisition-runner and semantic-parser hashes pass.
The real local NLI and BERTScore scorer preflight also passes. The acquisition
fingerprint is
`9e6c4094f3ff333c4d41bef13713065e5f62e77ba46f1bf8ebd3a2050737c5bc`.

The auxiliary candidate is frozen as **Cumulative Resolution Lag (CRL)**
(`Kumulatif Cozum Gecikmesi`, KCG). At each matched pre-final turn, answer
readiness is 1 for an explicit answer attempt and 0 for an explicit
`NEED_MORE_INFORMATION` response. The readiness gap is fresh-prefix readiness
minus dialogue-history readiness. CRL is the mean of `max(gap, 0)` over
comparable pre-final turns. Invalid pairs remain missing; support count and rate
are reported; there is no fitted threshold or minimum-support exclusion.

The prospectively frozen auxiliary target is a positive final readiness gap:
the fresh-prefix arm attempts an answer while the dialogue-history arm remains
explicitly unresolved. CRL is analyzed continuously, stratified by model and
clustered by source task; AUROC is reported only where both target classes
exist. The primary endpoint remains paired final-turn success. CS, LS and SS
remain the frozen pre-final RQEval candidates with Holm correction. CRL augments
RQEval and is not retroactively called an RQEval dimension.

The resumable runner is `scripts/run_lost_conversation_v2_science.py`. It prints
completed calls, percentage, rate and ETA; reuses the JSONL checkpoint; applies
the frozen semantic parser after acquisition; and writes PNG/PDF trajectories
for all six metrics, the complete-case mean and CRL. Its default invocation is
plan-only, and `--preflight` makes no ASU calls:

```bat
python scripts\run_lost_conversation_v2_science.py
python scripts\run_lost_conversation_v2_science.py --preflight
```

The campaign was execution-ready and deliberately locked during preparation.
On 2026-10-02 the investigator explicitly authorized the fixed 48-task,
11,382-call science campaign. Both authorization switches are now true. This
authorization applies only to the frozen configuration and does not permit
sample resizing, prompt changes or additional exploratory campaigns.

During acquisition, Mistral returned one API-success record with an empty raw
primary response at token count 2,048. The original transport correctly retained
the record but could not place empty assistant content into the next packed
history. The checkpoint was preserved without retry or deletion. The science
resume transport now represents any empty primary response in later history as
the explicit neutral token `[EMPTY_PRIMARY_RESPONSE]`; the original raw record
remains empty and invalid for metric scoring. This deterministic engineering
rule is hash-pinned, reported as a protocol deviation, and does not change the
sample, seed, model, prompt or analysis contract.

## Science result

The science campaign completed and was relocked on 2026-10-02 with
11,382/11,382 records, equal 3,794-record model arms, 100% API success and no
reported truncations. Six empty primary responses were preserved as invalid and
transported under the frozen placeholder rule. Raw record SHA-256 is
`e313f1ce59854d2590efe5602cee4799a917664f6f389426412fe433ea1772ba`.

Dialogue-history versus fresh-prefix final success was 0.688 versus 0.896 for
GPT-4o-mini (paired difference +0.208, Holm-adjusted `p=0.0259`), 0.854 versus
0.958 for Llama-4 Scout (+0.104, adjusted `p=0.125`), and 0.063 versus 0.354
for Mistral-7B (+0.292, adjusted `p=0.00394`). Thus the primary context penalty
was confirmed for GPT and Mistral, while Llama showed the same direction without
confirmatory significance. Excluding pairs with semantically invalid final
format did not remove the GPT or Mistral effects.

The frozen model-stratified, task-clustered CRL analysis yielded AUROC 0.627 and
one-sided permutation `p=0.0219`, with a broad task-bootstrap interval
[0.493, 0.760]. GPT had a nominal model-specific signal (AUROC 0.712, raw
`p=0.0336`) that did not survive three-model Holm adjustment (`p=0.101`). The
pooled stratified result, not the nominal GPT result, supports the auxiliary
warning claim. CS, LS and SS did not predict final dialogue failure after frozen
covariate adjustment and Holm correction; all adjusted p-values were 1.0. ES and
complete-case `RQ6_mean` remained structurally unavailable.

The main scientific interpretation is therefore not that a legacy RQEval
dimension warned reliably. Accumulated dialogue history caused a reproducible
resolution penalty, conventional consistency/coherence/stability trajectories
missed it, and the paired process measure CRL supplied a modest prospective
signal. Full outputs and PNG/PDF figures are under
`results/lost_conversation_v2_science/confirmatory/`.

A post-hoc robustness analysis then replaced the readiness target with the more
independent final accuracy-loss event: fresh prefix correct and dialogue history
failed. The pooled model-stratified result was AUROC 0.571 with clustered
permutation `p=0.0568` and bootstrap interval [0.484, 0.656]. Excluding all six
model-task units affected by empty primary transport produced AUROC 0.569 and
`p=0.0591`. Mistral showed a model-specific association, but GPT and Llama did
not. Therefore CRL is not confirmed as a model-general predictor of final
correctness loss; it is supported for prospective resolution-delay detection,
with only borderline post-hoc evidence for independent accuracy loss.
