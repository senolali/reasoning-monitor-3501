# Deviations from the Project Proposal (TÜBİTAK 3501)

## ADWIN defect and the opt-in corrected detector `adwin_v` (2026-10-04)

**Observation.** `detectors.ADWIN` uses the cut bound
eps = sqrt((2/m)·ln(2n/δ)). This omits the window variance and the
(2/(3m))·ln(2/δ') term of Bifet & Gavaldà (2007). For n ≤ about 100 values in
[0, 1], eps exceeds every attainable mean difference, so the detector can
never alarm. Evidence:

- It never alarmed in the CDR-Stream power simulation, at any drop size or
  number of dialogues aggregated.
- It never alarmed on a noiseless step from 0.8 to 0.2.

The shared-split fast path `detector_development._adwin_first_alarms`
reproduces the same bound.

**Why it matters.** ADWIN is the proposal's generic baseline. It is the
entire `adwin_all` strategy, and `proposal_mapping` assigns it to CS and SS.
Every H2 comparison against `adwin_all`, and every CS/SS monitoring result
under `proposal_mapping`, that used this detector reflects a detector that
could not fire. That covers the LIC-RQ-v2-SR replay, the CDR-Stream Stage A
and control-arm analyses, and possibly earlier sections. A separate audit
task is open for the earlier evidence.

**Change.** A new class, `ADWINVariance` (name `adwin_v`), implements the
paper's practical bound:

- eps = sqrt((2/m)·σ²_W·ln(2/δ')) + (2/(3m))·ln(2/δ');
- m = 1/(1/n0 + 1/n1) and δ' = δ/n.

Everything else is identical to the historical class: split points, minimum
side length, the one-sided decrease direction, warm-up and reset. The
historical `adwin` is unchanged, so frozen results reproduce. Two new
strategies, `proposal_mapping_v` and `adwin_all_v`, are opt-in in
`CORRECTED_STRATEGIES`; `DETECTOR_STRATEGIES` and all frozen analyses are
untouched. Tests are in `tests/test_adwin_variance.py`.

**Cost and caveat.** The corrected ADWIN is faithful but conservative on short
streams. A noiseless 0.6 drop at t=51 is cut at t=77 with δ=0.05 and t=90
with δ=0.01. On the GPT control arm aggregated over 6 dialogues per turn,
with thresholds calibrated on 2,000 permuted nulls, it raised no alarm on any
drift stream, while CUSUM and Page-Hinkley detected all three. ADWIN's
(2/(3m))·ln term needs far more than 100 samples per side for drops of
0.2–0.3. It is therefore an inappropriate generic baseline for 100-turn
streams, and H2 comparisons against it must say so.

**Final report must state.**

- The defect.
- Which results used the defective detector.
- That `adwin_v` replaces it only in protocols frozen after 2026-10-04.
- The sensitivity re-analysis (ADWIN_DEFECT_AUDIT.md §5): CS/SS were re-run with
  `adwin_v` and CUSUM in Sections 39, 49 and 51 and the LIC replay. No early
  warning appeared and no recorded decision changed.
- ADWIN's short-stream limitation.

## CDR-Stream analysis: S0-based null calibration and staged analysis after the budget pause (2026-10-04)

**Observation.**

- *Null calibration.* The design (CDR_STREAM_DESIGN §6, gate G3) calibrated
  detector thresholds on separate calibration-split S0 streams. G3 was never
  run.
- *Budget pause.* The main acquisition stopped at 25,039/54,000 calls (46.4%)
  when the ASU monthly budget ran out. Because primaries are scheduled first,
  all 90 primary chains are complete (one Llama turn-100 primary is missing).
  GPT-4o-mini is almost complete. Llama-3 70B has only about 8% of its
  repeats, paraphrases and control calls, and Gemma 3 12B has none.

**Why it matters.** The proposal's monitoring needs calibrated thresholds. An
interim look whose result decided which metrics are collected would bias the
hypotheses that compare non-CQ metrics with CQ (H3).

**Change.** The investigator approved the following on 2026-10-04:

- Thresholds are calibrated on the main-run S0 dialogues, pooled across models
  (α_cell = 0.05), and FAR is estimated by leave-one-dialogue-out.
- Primary tests pool the models, with sign flips stratified by model.
- The H1 window is 20 turns on either side of each change point.
- Stage A runs now: CQ, LS and ES for all models, and all six metrics plus the
  control arm for GPT.
- Stage B (CS, SS, RS and control for Llama and Gemma) will be completed with
  the same frozen runner, independently of Stage A results.

The protocol is frozen in CDR_STREAM_ANALYSIS_PROTOCOL.md §12.

**Cost.**

- The nulls are 36 dialogues, or 12 for CS/SS/RS in Stage A, rather than an
  independent set. FAR resolution is coarse (about 0.03, or 0.08 for 12
  nulls), and there is no family-wise correction across metrics.
- Until Stage B, the CS/SS/RS, H1b and H3 results rest on GPT alone.

**Final report must state.**

- The missing G3.
- The S0-based calibration.
- The budget pause.
- The Stage A/B split and the pre-results commitment to complete Stage B.
- Which results are Stage A only.

## CDR-Stream Llama replacement and G1 truncation-rule override for Llama-3 70B (2026-10-03)

**Observation.** The CDR-Stream technical pilot failed only on Llama-4 Scout,
for K-repeat raw-text diversity: 0.38 against a gate of >= 0.80. The same
frozen 90-call K-diversity probe gave the following on identical prompts:

| Model | K diversity |
|---|---|
| Scout aws | 0.533 |
| Scout asu-air | 0.50 |
| Maverick aws | 0.633 |
| Maverick asu-air | 0.700 |
| Llama-3 8B | 0.633 |
| Llama-3 70B | **1.000** |

A context-capacity probe showed that both Llama-3 models accept the full
100-turn history (37,139 gateway prompt tokens, oldest message recalled). The
earlier 8k assumption was wrong. Llama-3 70B then ran the frozen G1 (1,440/1,440
calls, 0 failed, contract validity 0.9965), with these results:

- It mapped budget_ledger (E3/M5/H10), inventory_logistics (E5/M9/H10) and
  lab_protocol (E4/M5/H10).
- It was excluded on milestone_chains, where level 1 was 0.67, below the E band.
- No cell needed G1b.
- Exactly one response was truncated at 2,048 tokens: item
  `milestone_chains/L09/SHARE/01`. It was long step-by-step reasoning cut
  mid-formula, not a repetition loop.

The frozen G1 operational rule ("any response is truncated" → model not
retained) therefore set `retained: false`.

**Why it matters.** The proposal commits to a Llama-family model, and CS/SS
require usable repeat diversity. Llama-3 70B is the only ASU Llama that meets
the diversity gate. Keeping Llama-4 Scout instead would make CS/SS degenerate
for the Llama arm.

**Change.** This is a post-hoc override of a frozen rule, decided by the
investigator on 2026-10-03 after the G1 result was seen. The single-truncation
operational failure is overridden for Llama-3 70B only, and Llama-3 70B
replaces Llama-4 Scout as the Llama core model. The rationale:

- 1/1,440 responses (0.07%) were truncated.
- The truncation fell in a template that is excluded for this model on
  accuracy grounds, so it affects no mapped cell.
- It was not a loop.

The override is implemented through `operational_overrides` in
`configs/cdr_stream_v1_g1v2llama3_70b_asu.yaml` and is recorded in the new
mappings file `results/cdr_stream_v1_mappings_llama3.json`. Its effects:

- The original G1 assessment, the records and the original mappings file
  (`9a63512b…`) are unchanged.
- GPT-4o-mini and Gemma 3 mappings are carried over unchanged.
- The rule itself is not changed for any other model or any future run.

The stream-stage technical gate (zero truncation) still applies unchanged to a
fresh Llama-3 70B pilot.

**Cost.**

- The calibration rule was relaxed after its outcome was known, for one model.
  Mistral-7B was dropped under the same rule with 5 truncations, so the two
  cases are treated differently. This is a disclosed inconsistency.
- Llama-3 70B is larger than the proposal's Llama-3-8B, so the core panel
  is no longer uniformly small/medium.
- milestone_chains is not available for the Llama arm.

**Second deviation: pilot gate override and a prospective truncation policy
(2026-10-03).**

*Observation.* The fresh Llama-3 70B technical pilot ran 600/600 calls with 0
failures. It reached validity 0.99/1.00 and K diversity 0.99, with no context
errors, but it failed the frozen §14 zero-truncation gate. Two non-primary
responses reached 2,048 tokens, and both were repetition loops:

- a repeat at turn 26;
- a paraphrase at turn 98.

Records SHA-256 `6f28ae21…`.

*Change.* The investigator chose option 3. It has two parts:

1. **Pilot gate override.** The failed §14 truncation item is overridden for
   Llama-3 70B, and the model proceeds to the main run. The failed pilot gate
   stays recorded as failed.
2. **Main-run truncation policy (`truncated_primary_placeholder_v1`).** It is
   frozen before any main-run output and applies to all three models:
   - Any response that reaches the output ceiling is kept verbatim in the raw
     record. It is never retried and never imputed.
   - It is scored as missing in the primary analysis.
   - A truncated primary enters later packed history as
     `[TRUNCATED_PRIMARY_RESPONSE]` instead of its looping text.
   - A prespecified sensitivity analysis rescores every truncated cell as
     incorrect for CQ, which bounds missing-not-at-random bias.
   - Truncation counts are reported per model, scenario, regime and turn.

The policy is implemented as opt-in, `mask_truncated_primaries`, and is
recorded in `configs/cdr_stream_v1_main_asu.yaml` and its fingerprint. All
earlier configs and fingerprints are unchanged.

*Cost.*

- This is the second post-hoc relaxation for the same model, and the
  fail-closed technical gate is replaced by a missingness policy.
- The pilot rate of about 0.3% implies roughly 60 truncated Llama-3 70B cells
  in the main run. These may concentrate in hard regimes, so missingness can
  correlate with difficulty. The sensitivity analysis and the truncation
  reports address this; they do not remove it.
- If a truncated primary replaces history text with a placeholder, the
  model's own context differs slightly from a free dialogue.

**Final report must state.**

- The original Llama-3-8B choice.
- All six Llama diversity results.
- The context-probe correction.
- The G1 truncation-rule failure and its post-hoc override, with this
  rationale.
- That model retention otherwise followed the frozen rule and no stream
  effect was observed before the decision.
- The 70B pilot truncation failure, its override, the main-run truncation
  policy, the truncation counts and the incorrect-scoring sensitivity result.

## CDR-Stream calibration-driven model and template changes (2026-10-03)

**Observation.** Two calibration rounds (G1 v1, G1 v2; 7,200 calls in total)
found the following:

- Mistral-7B sits at the accuracy floor on every template, with truncations
  and contract-validity problems. This is consistent with Sections 47, 59 and 62
  and with LIC-RQ-v2.
- The schedule (logic) template is too hard at every level for every model.
- GPT-4o-mini and Llama-4 Scout could be calibrated on two templates each.

**Why it matters.** The proposal's core design names GPT-4o-mini, Llama-3-8B
and Mistral-7B, and four reasoning domains including logic.

**Change.** Following its pre-stated rule, the investigator replaced
Mistral-7B. Claude 4.5 Sonnet was tried first. Its technical smoke showed
gateway/classifier truncation on the lab template, so the final third core
model is **Gemma 4 31B IT** (ASU-hosted `asu-air/gemma4_31b_it`, open weights,
a third model family). Gemma 4 31B was then calibrated and hit the ceiling: no hard-band level up to
level 14. The investigator replaced it with **Gemma 3 12B IT**
(`aws/gemma3_12b_it`), using calibration data only. **Claude Sonnet 5**, the
model the proposal names, takes
the proposal's frontier external-validation role, with lab declared excluded
before any accuracy was seen. Llama-4 Scout already substituted for Llama-3-8B
(Section 60). The schedule template was
replaced by the shallower `milestone_chains` logic template. Both changes go
through the unchanged frozen G1 rule before any stream call.

**Cost.**

- Gemma 4 replaces Mistral as the third open-weight core model, preserving the
  proposal's core/frontier split.
- Claude Sonnet 5 is not available for the lab template.

**Final report must state.** The final report must give the original model
list, the reasons (G1 results), the substitutions, and the fact that model and
template retention follows a frozen calibration rule rather than observed
stream effects.

## LIC-RQ-v2-SR session-stream detector replay (frozen 2026-10-02)

**Observation.** LIC-RQ-v2 tested the history penalty dialogue by dialogue and
never used the proposal's streaming layer. Its analysis had no sliding window,
no metric-specific detector, no H2 strategy comparison and no W ablation.
Presented alone, LIC-RQ-v2 would therefore omit the proposal's central
methodological contribution.

**Why it matters.** The proposal's contribution is streaming aggregation, drift
detection and early warning, not new metrics. The final report must show how
the LIC direction relates to that layer.

**Change.** A call-free replay protocol, `LIC_RQ_V2_STREAM_REPLAY_PROTOCOL.md`,
was frozen before any computation. It treats each completed LIC-RQ-v2 session as
a stream item. It constructs sudden (LIC-S) and gradual (LIC-G) streams in which
the regime switches from `fresh_prefix` to `dialogue_history` over the same 48
tasks per model, so the drift is a context-accumulation drift with the task mix
held fixed. It applies the unchanged `WindowMeanMetric`, the existing detectors
and the three strategies, with W in {10, 20, 30} (W=20 primary). Thresholds are
calibrated on order-uninformative permutation nulls and validated on a disjoint
set, with Bonferroni family-wise control (alpha_cell = 0.05/15), following
Sections 36/39. The investigator accepted the recommended decisions D1–D4.

**Cost.** None in model calls. The replay adds no new evidence that history
harms accuracy, because the effect was already known when the protocol was
written. That disclosure, including that the GPT-4o-mini LS/SS trajectory figure
had been seen, is part of the protocol. Detector direction remains the
proposal's "alarm on decrease" for every metric.

**Final report must state.** The replay is a post-hoc method demonstration,
conditional on 288 observed sessions. Its "early" alarms are early in session
order, not within a dialogue. H5 cannot be addressed because ES is
unobservable. The proposal's S2–S6 scenarios and the 5K–10K query scale remain
unaddressed.

**Observed result (2026-10-03).** The replay completed call-free in both value
modes. Under family-wise false-alarm control, which forced CQ-CUSUM to
thresholds 7.5–10 because the default 0.15 alarmed on 90–94% of null orderings,
the monitor almost never detected the drift. At W=20, only Mistral CQ-CUSUM
detected it (LIC-S 0.067, LIC-G 0.012; median delay about +44 sessions). GPT
and Llama windowed detectors were at or near 0.000. No metric was a replay
early-warning candidate. No H3 directional prediction held and no H2 strategy
difference held across models. Descriptively, SS and (for GPT/Mistral) LS rose
under accumulated history while CQ fell, so decrease-alarming detectors cannot
flag them. Final-turn sensitivity agreed. This replicates the Section 36/38/39
stream conclusion with task difficulty held fixed. It concerns detector
sensitivity at 48 sessions per regime and does not weaken the confirmatory
LIC-RQ-v2 history effect. Details and hashes are in Section 13 of the protocol.

## LIC-RQ-v2 frozen direction (2026-10-01)

After the DBQR prompt-only executable-plan probe failed strict scoring at 0/6,
the investigator selected the pinned Microsoft Lost in Conversation math subset
as the active extension. LIC-RQ-v2 freezes three ASU models (GPT-4o-mini,
Llama-4 Scout and Mistral-7B), paired dialogue-history/fresh-prefix conditions,
K=5, L=2, four technical-integration tasks and one 48-task science sample.
Selection is input-only and fixed before LIC model output. The technical
integration was authorized, completed and relocked; see LIC_RQ_V2_PROTOCOL.md and
results/lost_conversation_v2_manifest/.

The accepted proposal anticipated quality trajectories over query streams. This
extension replaces unrelated-query order with coherent progressive information
revelation, while preserving the six RQEval dimensions. It does not claim that
all six are observable at every shard. Most partial prefixes do not determine the
numeric gold answer, so CQ/RS/ES remain missing until eligible; pre-final CS/LS/SS
are the prospectively frozen warning candidates. The primary outcome is paired
final-turn success. Seven plots retain explicit support and missingness, and the
six-metric mean is complete-case descriptive only. One technical integration is
permitted in principle, but no further effect-seeking pilot ladder is planned.

The adapted integration runner subsequently passed two call-free engineering
checks: deterministic fixture replay with full checkpoint reuse and explicit
missingness/plot validation, and loading plus smoke evaluation of the real local
NLI and BERTScore scorers. No ASU call was made. The 1,092-call integration was
explicitly authorized by the investigator on 2026-10-01. This approval is limited
to technical integrity and does not authorize the 48-task science run.

The integration completed on 2026-10-02 with 1,092/1,092 successful calls and
no truncation. Strict formatting substantially understated observability,
especially for Mistral. A marker-anchored, call-free semantic rescore raised
valid-response coverage from 36.5%/82.4%/3.0% to 100%/100%/94.2% for
GPT/Llama/Mistral. This is an engineering correction established on integration
data; both strict and semantic results are retained and the parser is frozen
before science data.

On four integration tasks, paired final success (history/fresh) was 1.00/1.00,
0.75/0.75 and 0.00/0.50 respectively. Mistral's history arm remained explicitly
unresolved on every final task, while LS and SS generally remained high. ES and
the complete-case six-metric mean were structurally unavailable. This motivated
an exploratory auxiliary resolution-readiness gap, not a redefinition of RQEval:
fresh-prefix answer readiness minus history-arm readiness at matched information.
Persistent/non-recovered positive gaps are the candidate warning construct for
prospective validation. Integration N=4 cannot support a scientific effect claim,
and the 48-task science run remains unauthorized.

Before accessing the untouched science sample, this construct was frozen as
Cumulative Resolution Lag (CRL; Turkish KCG): the mean positive
fresh-minus-history answer-readiness gap over comparable pre-final turns. Invalid
pairs are missing, support is explicit, and no empirical threshold or support
cutoff is selected. The target is a positive final readiness gap. CRL is an
auxiliary process measure rather than a new RQEval dimension; paired final-turn
success remains primary and CS/LS/SS remain the multiplicity-controlled pre-final
RQEval family.

The 48-task/11,382-call campaign now passes call-free source, hash, sample,
budget, production-probe, NLI and BERTScore preflight. Its resumable runner also
pre-registers all six metric trajectories, the complete-case mean and CRL plots.
Preparation made no ASU call. On 2026-10-02 the investigator explicitly
authorized this fixed 48-task, 11,382-call campaign and both execution switches
were enabled. This is the final planned campaign, not another effect-seeking
pilot; its sample and analysis contract remain frozen.

The science run stopped after 7,847 checkpointed calls when an ASU-success
Mistral primary contained empty raw text at the 2,048-token ceiling. The raw
record was retained, was not retried, and remains invalid/missing for scoring.
Because the packed-history builder cannot carry an empty assistant message, the
resume transport inserts the explicit neutral marker `[EMPTY_PRIMARY_RESPONSE]`
only into later dialogue context. This is a deterministic transport repair, not
an outcome-based exclusion or response imputation; its occurrence is reported.

The completed 48-task science campaign confirmed a dialogue-history penalty for
GPT-4o-mini and Mistral-7B after paired exact tests and Holm adjustment; Llama-4
Scout showed a smaller non-significant difference in the same direction. The
effect persisted in the format-valid sensitivity analysis. None of the eligible
original pre-final RQEval metrics (CS, LS, SS) predicted final dialogue failure
after task-clustered covariate adjustment and Holm correction. ES and RQ6_mean
were structurally unavailable.

The prospectively frozen auxiliary CRL measure did provide a modest pooled,
model-stratified warning signal (AUROC 0.627; task-clustered permutation
`p=0.0219`), although its bootstrap uncertainty was broad. This changes the
strongest defensible project claim: degradation is detectable through a paired
resolution-process measure, not through any one of the original six RQEval
dimensions. CRL remains explicitly auxiliary rather than being relabeled as an
RQEval dimension.

A subsequent call-free post-hoc robustness analysis tested CRL against a target
independent of its readiness-marker construction: fresh prefix correct while
dialogue history failed. The pooled model-stratified association narrowly missed
the conventional threshold (AUROC 0.571, clustered `p=0.0568`), and exclusion of
all empty-primary transport units was nearly identical (AUROC 0.569, `p=0.0591`).
Only Mistral showed a model-specific association. Accordingly, the project does
not claim that CRL is a confirmed model-general predictor of final correctness;
the supported claim concerns resolution delay, with borderline exploratory
evidence for downstream accuracy loss.

## LIC-RQ-v1 input/observability audit (2026-10-01)

The approved next call-free step now has a reproducible source-input overlap
audit, first-pass semantic review, grouped candidate samples and call budgets.
See LIC_RQ_AUDIT.md. Of 103 public math tasks, 11 match earlier completed inputs
and 8 have provisional fidelity/ambiguity concerns; 2 are in both groups. The
remaining 86 are candidates, not independently validated or contamination-free.
No model outcomes determined exclusions or ranking; no source/gold was repaired.

The first-pass review finds only 7 non-quarantined tasks potentially sufficient
before the final shard. Thus all-six temporal completeness is not a valid promise
for this dataset: CQ/RS/ES eligibility, ES warmup and the original correct-primary
RS subset strongly limit RQ6_mean coverage. Missing values remain missing. These
are data/design findings, not new experimental evidence of degradation. Proposed
24/48/64-task options and four reserved integration tasks are not execution
authorization. Historical CDR/RAT gates and closed splits remain untouched.

## LIC-RQ-v1 preparation decision (2026-10-01)

The investigator approved proceeding with the published Lost in Conversation
tasks as a basis for six-dimensional temporal analysis, including a descriptive
equal-weight mean. See LIC_RQ_PROTOCOL.md for the full preparation contract.
This is a dialogue/information-revelation extension to the proposal, not proof
that the original datasets failed. The source is pinned by commit and SHA-256;
103 math tasks provide 585 prefixes. No model outputs were collected.

Proposed adaptations are fixed source-shard order, complete-shard observation
without correctness-based early stopping, and a paired fresh-prefix control
with identical revealed information. These differ from the paper's adaptive
user simulator and must be disclosed. Models remain GPT-4o-mini and Llama-4
Scout for preparation; this does not silently fulfill the proposal's three-model
commitment or authorize execution. Whole-pool costs are not a selected sample.

Partial-prefix answerability is unreviewed, not automatically incorrect. Natural
clarification is allowed. All metric coverage is explicit; RQ6_mean requires
six measured values and is gold-dependent/complete-case descriptive, not an
online predictor. Prior CDR/RAT gates and splits remain unchanged. Preparation
does not claim statistical power, confirmatory readiness or early warning.

This file documents every intentional divergence between the submitted project
proposal and the implemented framework.  Each entry records **what the proposal
said**, **what the code does**, and **why the deviation is better or necessary**.
These notes are intended for the project report (Section 3.3 "Implementation
Notes" or equivalent).

---

## §1. Metric–Detector Assignment

**Proposal (form table):** Two inconsistent tables appeared in the form.
One stated CQ→CUSUM, CS→ADWIN, RS→DDM, LS→Page-Hinkley, ES→EWMA, SS→ADWIN.
Another stated CQ/RS→CUSUM, CS/SS→ADWIN, LS→Page-Hinkley, ES→EWMA.

**Implementation (v0.1.1+):** Assignment is derived from the *statistical
distribution* of each metric's per-query value:

| Metric | Distribution | Detector | Rationale |
|--------|-------------|----------|-----------|
| CQ     | Bernoulli   | DDM      | DDM is defined on binary error streams |
| CS     | Beta        | CUSUM    | SPRT-derived test for bounded continuous mean shift |
| RS     | Beta        | CUSUM    | Same as CS |
| LS     | Beta        | CUSUM    | Same as CS |
| ES     | Beta        | CUSUM    | Harmonic mean output is bounded [0,1] |
| SS     | Normal      | EWMA     | Shewhart/EWMA for small persistent shifts in window-mean sequence |

ADWIN serves as the distribution-agnostic baseline in the ablation (applied to
every metric via `detector_overrides`) rather than as a primary detector.

**Why better:** Distribution-based assignment is theoretically principled and
directly testable via the ablation (H2). The proposal's ad-hoc tables were
internally inconsistent and not derived from any stated criterion.

**Report note:** State that the theoretical mapping was refined post-submission
to ensure statistical validity; the ablation directly tests whether this
mapping outperforms ADWIN-for-all, which is the H2 hypothesis.

---

## §2. CQ Two-Level Detection (new in v0.1.2)

**Proposal:** Did not distinguish between raw and windowed detection levels.

**Implementation:** CQ has a *secondary* DDM detector that receives the raw
per-query Bernoulli value (0 or 1), in addition to the primary CUSUM detector
on the windowed mean.

**Why necessary:** DDM's Binomial distribution assumptions apply to a binary
stream.  Feeding the window mean (∈ {0, 1/20, 2/20, …, 1}) to DDM violates
those assumptions.  The two-level design preserves the windowed mean pipeline
(needed for composite scoring and the other detectors) while also giving DDM
statistically valid input.

**Report note:** Present two-level detection as a methodological contribution:
it is the first explicit treatment of the raw vs. windowed detection distinction
in the LLM monitoring literature.

---

## §3. Gradual Drift Change Point (t*)

**Proposal:** Defined t* as "the first time expected quality deviates from
baseline by more than ε", which is well-defined for sudden drift but ambiguous
for gradual drift (the quality change is continuous, not instantaneous).

**Implementation:** For gradual scenarios (S1, S4, S5):
```
t* = change_point + transition_width // 2
```
This is the midpoint of the mixing ramp where the post-drift dataset constitutes
50% of the stream.

**Why better:** Using the start of the transition would systematically
under-estimate detection delay (the quality effect is small there); using the
end would over-estimate it (drift is complete by then).  The midpoint is the
most defensible single-number ground truth for a continuous process.

**Report note:** Acknowledge the choice explicitly and report sensitivity
to alternative t* definitions (start, end, ε-crossing) as a robustness check.

---

## §4. ES Formula Correction (v0.1.1 fix, v0.1.0 bug)

**Proposal:** ES = 2·CQ·(1−T) / (CQ + (1−T)), T = n_tokens / t_max

**v0.1.0 implementation (bug):** ES was a log-compressed token ratio with no
CQ term — a pure token economy measure.  A wrong answer using 20 tokens scored
ES ≈ 1.0, violating the proposal's intent.

**v0.1.1+ implementation:** Matches the proposal formula exactly.

**Report note:** Note the v0.1.0 bug was discovered during pre-submission code
review and corrected before any experiments were run.

---

## §5. RS C-Subset Constraint (v0.1.1)

**Proposal:** Did not explicitly state the C-subset constraint.

**Paper (arXiv:2605.24661):** RS is "computed exclusively over originally correct
instances C to prevent trivially high scores for consistently-wrong models."

**Implementation:** RS returns NaN when the primary call is wrong.  The window
metric skips NaN values, so incorrect queries do not enter the window mean.

**Why necessary:** Without the C-subset, a model that consistently gives wrong
answers and consistent paraphrases would score RS = 1.0, making RS meaningless.

**Report note:** Credit the paper formulation and note that the implementation
enforces the C-subset at the metric level (not the monitor level) for modularity.

---

## §6. SS — Aligned with rqeval/arXiv:2605.24661 (v0.1.3-final)

**rqeval definition (arXiv:2605.24661, §3.2):**
```
SS_i = (2 / K(K-1)) * Σ_{k<l} BERTScore(r_i^(k), r_i^(l))
SS(t) = (1/W) * Σ_{i∈W(t)} SS_i
```
Measures pairwise semantic similarity of *reasoning traces* across K runs.
Answers: "Does the model reason the same way when asked the same question twice?"

**v0.1.0–v0.1.2 deviation:** Used variance-of-window-means (temporal stability
of accuracy over the query stream) — a different and arguably complementary
concept, but inconsistent with the published rqeval definition.

**v0.1.3-final fix:** SS now uses pairwise BERTScore across K=5 reasoning
traces per query (k_repeats from config). Falls back to Jaccard token
similarity when bert-score is not installed.

**Distribution:** BETA (bounded [0,1] similarity) → CUSUM detector.

**Report note:** Explicitly state that SS matches the rqeval paper definition.
For publication experiments, use stability_scorer='bertscore'; Jaccard results
are not comparable and must not be pooled with BERTScore results.

---

## §7. Page-Hinkley Implementation (v0.1.2 fix)

**v0.1.0/v0.1.1 bug:** PH had two problems:
1. v0.1.0 used non-standard exponential decay (alpha parameter).
2. v0.1.1 "fixed" it to `stat = max_cum − cum`, which detects *recovery after
   a drop*, not the drop itself — the opposite of what we need.

**v0.1.2 implementation:** One-sided accumulation with max(0,…) reset:
```python
cum_t = max(0, cum_{t-1} + (prev_mean − x_t − δ))
alarm when cum_t > λ
```
This accumulates downward deviations and resets when quality is fine.

**Report note:** Document the bug history and reference the corrected
formulation.  The max(0,…) reset is a CUSUM-style design choice that makes PH
self-resetting (no need for explicit reset after detection).

---

## §8. Scenario Numbering (v0.1.1)

**Proposal S1–S6:** Gradual Difficulty, Sudden Domain, Recurring, Linguistic,
Noisy Prompt, Mixed.

**Implementation S0–S6:** S0 added as the null/stationary scenario needed for
false-alarm rate estimation.  S1–S6 match the proposal.

**Why added:** A null scenario (no drift) is required to estimate the false-
alarm rate of each detector.  Without it, H2 ("metric-specific detectors
outperform ADWIN baseline") cannot be properly evaluated (detection power must
be compared against false-alarm cost, not in isolation).

---

## §9. Dataset-Specific Answer Normalization (v0.1.2)

**Proposal / paper:** Used generic first-number extraction.

**Implementation:** Four dataset-specific extractors:
- StrategyQA: yes/no keyword detection
- ARC-Challenge: single letter A–D
- MATH: \\boxed{} expression (nested braces), then last number
- GSM8K / generic: last number in the string (not first — intermediate steps
  appear before the final answer, so the last number is the answer)

**Why necessary:** Generic first-number regex fails silently on ~15–30% of
items in ARC-Challenge (multiple-choice) and StrategyQA (yes/no).  On MATH,
answers may be fractions, polynomials, or sets — not representable as a single
number.  Incorrect CQ would corrupt ES and SS (which depend on CQ).

---

## §10. K/L Config Propagation Bug Fix (v0.1.2)

**Proposal:** k_repeats=3, l_paraphrases=3 (MonitorConfig).

**v0.1.0/v0.1.1 bug:** Module-level constants `K_RUNS=3` and `L_PARAPHRASES=3`
in dimensions.py shadowed the config values.  Setting `k_repeats=5` in config
had no effect on CS; it always used K=3.

**v0.1.2 fix:** K and L are now constructor arguments of ConsistencyScore and
RobustnessScore, forwarded from MonitorConfig in ReasoningMonitor._build().

**Report note:** Flag this as a critical pre-experiment fix to ensure
reproducibility.

---

## §11. Publication Mode Guard (v0.1.2)

**Proposal:** Did not address the risk of lexical-fallback LS scores being
inadvertently used in published experiments.

**Implementation:** `MonitorConfig(publication_mode=True)` with
`coherence_scorer="lexical"` raises `ValueError` at construction time.

**Why necessary:** Lexical and NLI LS scores are not on the same scale and
must not be pooled.  The guard makes it impossible to accidentally run a
publication experiment with the wrong scorer.

---

## §12. Three-Strategy Experimental Design (v0.1.3)

**Proposal:** Stated a single metric→detector mapping (with two inconsistent tables).

**Implementation:** Three named strategies are provided and compared under H2:

| Strategy | Description |
|----------|-------------|
| `distribution_based` | Theory-derived from metric distribution (Beta→CUSUM etc.) |
| `proposal_mapping` | As stated in proposal: CQ→CUSUM, CS→ADWIN, RS→DDM, LS→PH, ES→EWMA, SS→ADWIN |
| `adwin_all` | ADWIN applied to all metrics (generic H2 baseline) |

**Why better:** The inconsistency between the two proposal tables becomes a research question rather than an error. H2 ("metric-specific detectors outperform generic baseline?") is now tested against two candidate specific-mappings, not one.

**Report note:** §4 ablation table should compare detection delay F1 across all three strategies per scenario type (S1–S6).

---

## §13. CQ Two-Level Detection Fix (v0.1.3 — v0.1.2 bug)

**v0.1.2 bug (flagged by peer review):** DEVIATIONS §2 stated "primary CUSUM for windowed CQ" but the code gave DDM for both primary and raw levels because `THEORETICAL_ASSIGNMENT[Bernoulli]→DDM` was applied to the windowed level too.

**v0.1.3 fix:** `MetricSpec.windowed_distribution=BETA` on CQ causes:
- Primary (windowed) detector: CUSUM — correct for Beta-distributed window mean
- Secondary (raw) detector: DDM — correct for Bernoulli per-query stream

**Report note:** Present this as "the framework distinguishes event-level and aggregate-level drift detection: DDM captures sudden individual errors; CUSUM detects sustained degradation in aggregate quality."

---

## §14. K=5 Default (v0.1.3 — v0.1.2 error in DEVIATIONS doc)

**v0.1.2 DEVIATIONS doc error:** §10 incorrectly stated "Proposal: k_repeats=3". The proposal's experimental protocol page explicitly states K=5.

**v0.1.3 fix:** `MonitorConfig.k_repeats` default changed from 3 to 5.

---

## §15. Fraction Normalization (v0.1.3 — from GPT v0.2.0 review)

**GPT contribution incorporated:** `\frac{a}{b}` and plain `a/b` are now converted to decimal before comparison, enabling cross-format matching (gold=`\frac{1}{2}` matches prediction=`0.5`). Scientific notation support also added.

**Why:** Prevents false negatives where mathematically equal answers are in different formats. Particularly important for MATH dataset which uses LaTeX notation.

---

## §16. t* Sensitivity Analysis (recommended for paper)

**Current implementation (§3):** Gradual t* = change_point + transition_width // 2 (50% mixing point).

**ChatGPT review note:** This is a reasonable choice but not the proposal's formal definition (ε-crossing of expected quality). For robustness, the paper should report detection delay under three t* definitions:
- t_start: beginning of transition band
- t_50%: midpoint (current implementation)
- t_ε: first time estimated quality deviates > ε from baseline

This becomes a sensitivity analysis in §4, strengthening the paper's methodological rigor.


---

## §17. Three-Category Alarm Classification and Signed Delay (v0.1.3-final)

**Problem identified:** For gradual scenarios (S1, S4, S5), alarms in [ts, t*) were
being silently discarded — neither counted as false alarms nor as true detections.

**Example:** ts=100, t*=120, alarm at t=110 → not counted as anything.

This is methodologically wrong because an alarm at t=110 is a genuine early
warning: drift has started (ts=100) and the metric detected it 10 queries
before the quality threshold was crossed (t*=120).

**v0.1.3-final fix — three categories:**

| Alarm time | Category | Comment |
|------------|----------|---------|
| ta < ts | false_alarm | Before drift starts — definitively wrong |
| ts ≤ ta < t* | early_detection | Drift started, quality not yet degraded |
| ta ≥ t* | post_threshold | After quality threshold crossed |

**Signed delay D* = ta - t* (negative = early warning)** is stored alongside
delay_from_transition_start = ta - ts (always ≥ 0 for true detections).

**Report note:** Report D* distribution per metric. CS and LS with mean D* < 0
on gradual scenarios (H3 hypothesis) would be a novel finding.

---

## §18. Event-Based F1 (H2 success criterion from proposal)

**Proposal H2:** "Detection delay and F1 score compared to ADWIN baseline."

**Implementation:** Scenario-level F1 per (metric, detector, strategy):
- TP = drift scenarios (S1-S6) where detected=True (alarm in drift window [ts, end])
- FP = null scenarios (S0) where n_false_alarms > 0
- FN = drift scenarios where detected=False

Multi-event scenarios (S3 recurring, S6 mixed) are treated as single events
in the current implementation. Full multi-event ground truth (DriftEvent list)
is documented as future work.

**Report note:** Table H2 in the paper should show F1 for all three strategies
to directly answer the hypothesis.

---

## §19. H2 Full Matrix (v0.1.3-final)

**Previous:** H2 ran only on S2 (sudden domain shift, mid model).

**v0.1.3-final:** H2 runs on all S0–S6 × all models × 3 strategies.

Design matrix: 3 strategies × 7 scenarios × 3 models × 6 metrics = 378 rows.
S0 contributes false-alarm data; S1–S6 contribute detection+delay+F1 data.

Output files: `h2_strategy_comparison.csv`, `h2_f1_by_strategy.csv`.

---

## §20. rqeval Adapter Layer — Known Deviations (v0.1.3-rqeval)

**Overview:** Metric scoring now delegates to the published rqeval package.
The paper claim is: "We use the six scoring functions from RQEval; our
contribution is the streaming aggregation and drift-detection layer."

Three adaptations were required for streaming compatibility:

### RS: NaN vs. 0 for non-C items

**rqeval:** Returns 0.0 for non-C items (wrong original answers) in
`compute_per_instance()`, then averages only over C items in `compute()`.

**Adapter:** Returns `float("nan")` for non-C items so `WindowMeanMetric`'s
NaN-skip logic achieves the same effect as rqeval's C-subset average.
Result: numerically equivalent to rqeval's batch C-subset RS.

### ES: Window-range vs. Batch min-max normalization

**rqeval:** Normalizes T'_i = (t_i - T_min)/(T_max - T_min) over the full
evaluation batch.  For a single observation (T_min == T_max), T'=0 always,
making ES collapse to CQ — useless for per-query streaming.

**Adapter:** Passes the full current window (W items) to rqeval's
`compute_per_instance()` and returns the window mean.  This approximates
rqeval's batch normalization with a sliding window instead of the full batch.
During warm-up (< W items), the window range shrinks; this is acceptable
because early alarms are suppressed by detector warm-up anyway.

**Note:** This is the only metric where the numerical result may differ from
rqeval's batch computation.  For large batches rqeval's range → global range;
for the window, range → local range over W=20 items.

### LS: NLI model loading in offline/CI environments

**rqeval:** Loads `cross-encoder/nli-deberta-v3-small` from HuggingFace on
first use; falls back to heuristic contradiction detection when unavailable.

**Adapter (v0.2.1):** the fallback is no longer silent.  rqeval swallows the
loading exception and then scores every step pair `neutral`, so LS is a constant
1.0 with no warning.  `LSAdapter.scorer_status()` forces the load and reports
what happened, `ReasoningMonitor.preflight()` refuses the run in
publication_mode, and `MonitorResult.degenerate_metrics()` catches a constant
channel after the fact.  Setting `coherence_scorer="nli"` is now sufficient —
before v0.2.1 that setting never reached the adapter at all (see §24).

---

## §21. rqeval Version Provenance & Pilot Notes (v0.1.3-final)

### rqeval metadata inconsistency

The published rqeval package has a metadata inconsistency:
- `importlib.metadata.version("rqeval")` → 2.0.2 (distribution version)
- `rqeval.__version__` → 1.2.0 (internal string)

This is an rqeval packaging issue, not a reasoning-monitor problem. The
experiment manifest records **both** values plus the import paths under the
`rqeval` key, so any result can be traced to the exact package used.

### Pilot observations (simulated client)

A 40-query pilot on the synthetic corpus confirmed:
- CQ, CS, RS, ES, SS all produce sensible [0,1] variation
- RS NaN rate matches the fraction of wrong primary answers (C-subset working)
- CQ fires on two separate channels: raw DDM (drift_raw) and windowed CUSUM (drift)

**LS = 1.0 constant in simulation** is expected, not a bug:
1. The simulated client emits uniform "compute intermediate value N" steps
   with no semantic contradictions.
2. When HuggingFace is unreachable, rqeval's NLI falls back to a heuristic
   that finds no contradictions in such steps.

LS variation must be re-verified on the first real-model pilot. Run
`scripts/pilot_check.py` and confirm LS shows variation with a real model
and NLI access (coherence_scorer="nli").

### Verification gate

Before real experiments, run in a fresh environment:
```
pip install rqeval
pip install -e ".[dev]"
python scripts/verify_rqeval.py   # → all PASSED
pytest -q                          # → 66 passed, 0 skipped, 0 failed
python scripts/pilot_check.py      # → sensible metric variation
```

---

## §22. E2H-GSM8K Controlled-Composition Redesign (v0.2.0)

**Motivation:** The panel review's central criticism (repeated in both the
Özgün Değer and Yöntem sections) was that mixing datasets of different
difficulty confounds *reasoning degradation* with *questions getting harder*.

**v0.2.0 redesign** replaces the S0–S6 multi-drift-type scenarios with six
controlled-composition streams on a single dataset (GSM8K, split into
Easy/Medium/Hard terciles):

| Stream | Composition | Role |
|--------|-------------|------|
| ST1 | Mixed E/M/H random | Stationary baseline (all difficulties) |
| ST2 | Easy→Medium→Hard | Controlled composition shift (MAIN) |
| ST3 | Easy only | Stationary Easy control |
| ST4 | Medium only | Stationary Medium control |
| ST5 | Hard only | Stationary Hard control |
| ST6 | Hard→Medium→Easy | Reverse / recovery test |

**Key properties:**
- No synthetic questions generated — existing GSM8K items only.
- Within each regime, queries are drawn by random permutation of the
  difficulty pool (never sorted), so within-regime order carries no signal.
- Each stream is regenerated under R independent seeds; the primary analysis
  is the cross-seed reproducibility of the alarm order (e.g. SS→CS→CQ).
- ST3/ST4/ST5 stationary controls give per-difficulty baseline profiles, so
  the ST2 temporal transition can be interpreted against known baselines —
  directly resolving the confound the panel flagged.

**Difficulty proxy:** GSM8K has no official labels; terciles use a reproducible
proxy (operand count, prompt length, gold magnitude), reported openly. This is
a composition-control tool, not a difficulty research claim.

**Scope trade-off:** The linguistic (S4) and noisy-prompt (S5) drift types are
dropped. This is a deliberate depth-vs-breadth trade: the revised design
studies one well-controlled phenomenon reproducibly rather than many synthetic
drift types superficially.

**DDM calibration:** DDM now requires a minimum sample count (30) before
arming and guards against a degenerate s_min, fixing false alarms on short
stationary binary streams observed in the E2H pilot.

## §23. Future direction (recorded, not in current scope)

The original research intuition — whether reasoning quality degrades over a
**single long conversation on one fixed topic** (context/conversation
degradation, distinct from independent-query composition drift) — is a separate
phenomenon requiring a multi-turn conversation stream with accumulating
context. The six metrics and drift-detection layer are reusable for it. This is
noted for a follow-up project, outside the accepted TÜBİTAK 3501 scope.

---

## §24. Configuration Reachability, Preflight and Raw-Record Persistence (v0.2.1)

**Observation.** A pre-pilot audit of v0.2.0 (full source read plus an executed
run in a clean environment) found that several controls the methodology relies
on could not, in fact, be exercised, and that three failure modes were silent:

1. `publication_mode` existed only on `MonitorConfig`; `ExperimentConfig` had no
   such field and `_make_monitor()` never set it, so **no YAML or CLI run could
   reach the guard**. The same held for `k_repeats`, `l_paraphrases` and
   `stability_scorer` — K and L were pinned at the `MonitorConfig` defaults and
   adding them to a YAML raised `TypeError`.
2. `monitor.py` read
   `nli_model = None if cfg.coherence_scorer == "lexical" else None` — `None` in
   both branches — so `coherence_scorer` had **no effect whatsoever** on the
   rqeval path. `stability_scorer` was likewise never passed to `SSAdapter`.
   This is the same shadowing defect as the K/L bug of §10.
3. rqeval's LS silently falls back to a no-contradiction heuristic when the NLI
   pipeline fails to load, and SS to token Jaccard when bert-score is absent.
   Measured on an executed ST2 run: `Var(LS) = 0` exactly, `sd(SS) = 0.011`.
   Nothing warned; the run looked healthy.
4. `MonitorResult.per_query` was never written to disk and model outputs were
   never stored at all, so a completed campaign could not be re-scored offline.
5. The v0.2.0 rename S0–S6 → ST1–ST6 left the old names in `cli.py --quick`,
   `scripts/pilot_check.py` and `configs/live_asu.yaml`; all three died with
   `KeyError`. The CI smoke step calls `--quick`, so **CI was red**, as was
   `ruff check src/ tests/` (64 pre-existing findings).
6. `ESAdapter.per_query_value()` called a private rqeval helper, discarded the
   result and returned `NaN`, so the `ES_raw` column — the one `pilot_check.py`
   reads to judge ES — was NaN for every row.

**Why it matters.** Items 1–3 mean a publication run could have been configured
exactly as the protocol requires and still have measured nothing on two of the
six metrics, with a manifest that recorded the *intended* scorer rather than the
*actual* one. Item 4 means any scoring defect found after a campaign would have
cost the campaign. Item 5 means the pilot's own entry points did not run.

**Change.**

- `ExperimentConfig` gains `k_repeats`, `l_paraphrases`, `stability_scorer`,
  `nli_model`, `bertscore_model`, `publication_mode`; `_make_monitor()`
  propagates all of them, and `_run_strategy_comparison()` now carries the
  scoring configuration into its arms (it previously dropped `use_rqeval`, so a
  `use_rqeval=False` run compared three arms scored by a different
  implementation).
- `coherence_scorer` / `stability_scorer` actually reach the adapters.
- `ReasoningMonitor.preflight()` runs **before the first model call** and checks
  (a) that LS's NLI pipeline is real, (b) that SS's BERTScore is real, and
  (c) that the K repeats are not issued at `temperature=0`, where they are
  identical by construction and CS/SS are 1.0 whatever the model does. The
  report goes into the manifest; in `publication_mode` any problem raises.
- `MonitorResult.degenerate_metrics()` flags any windowed metric that never
  varied; logged always, fatal in `publication_mode`.
- Every model call is written verbatim to `records_<tag>.jsonl` (prompt,
  completion, extracted answer, tokens, latency, `ok` flag) **before** anything
  interprets it, and per-query metric values to `per_query_<tag>.csv`.
  `Sample` gains `raw_text` and `ok`.
- Scenario names corrected in the CLI preset, the pilot script and
  `configs/live_asu.yaml`; `_validate_scenarios()` now fails with the valid
  names instead of a `KeyError` after the datasets have been loaded. The
  headline-figure selector looked for `"S2_"` and therefore drew from ST1, the
  stationary baseline; it now prefers ST2.
- `rqeval` pinned to `==2.0.3` (was `>=2.0.0`). An unpinned dependency that owns
  all six scoring functions is a reproducibility hole: an upgrade silently
  redefines the measurements. Bumping the pin is a protocol change.
- Pre-existing lint debt cleared so `ruff check src/ tests/` passes.

**Cost.** `records_*.jsonl` is the dominant artefact on disk: one JSON object per
call, i.e. `N × (K + L)` objects per run (~1,200 for a 150-query stream at
K=5, L=3), each carrying the full completion text. This is the price of being
able to re-score without re-paying, and it is the cheaper side of the trade.

**Open decision, not taken here.** `configs/live_asu.yaml` shipped with
`temperature: 0.0`, which makes CS and SS degenerate as described above. The
file now carries `0.7` with the reasoning stated inline, but **the temperature is
a protocol choice for the PI, not a patch**: either sample the K repeats
(self-consistency convention, ~0.7) or set `k_repeats: 1` and drop CS/SS from
the run. Whichever is chosen must be declared. Related: this client does not
transmit `seed` in the request body, so repeat-level determinism is not
controllable through the gateway; that too must be stated.

**Open issues identified by the v0.2.1 audit** (resolved in §25-§26):
multi-seed execution and the cross-seed alarm-order statistic are still not
wired into `run_experiment` — `build_multiseed_streams()` is written and tested
but never called, so **H3 currently has no code path**; only the first of ST2's
two change points is evaluated; the H2 strategy comparison re-generates every
completion per arm (~3× the API cost) where one generation replayed through
three detector banks would do; and a NaN from RS suppresses the whole snapshot,
so the detectors see a correctness-conditioned subsample of the stream (measured:
97 snapshots for 150 queries on ST2).

**The final report must declare:** that the v0.2.0 code could not reach its own
publication guard and that `coherence_scorer` / `stability_scorer` were inert on
the rqeval path; that no result predating v0.2.1 can be assumed to have used the
scorer its manifest names; and that from v0.2.1 on, every reported run is
accompanied by its raw generations and its preflight report.

---

## §25. Independent Metric Clocks and Per-Transition Evaluation

**Observation.** In v0.2.1 an undefined RS value suppressed the complete
six-metric snapshot. Because RS is defined only for the correctly answered
C-subset, hard regimes advanced every detector more slowly than easy regimes.
ST2 and ST6 also exposed two regime boundaries, but evaluation used only the
first one.

**Change.** A query now advances every metric whose window value is available;
an undefined RS value skips only RS. The composite score is calculated from
the available metrics after renormalising their active weights. Staged streams
are evaluated in separate adjacent-regime intervals, so E-to-M and M-to-H (and
their reverse counterparts) produce independent change and detection rows.

**Why necessary.** Detection delay must remain on the query-time scale and
must not depend on whether another metric is defined. Treating the two regime
boundaries separately also prevents an alarm attributable to the first shift
from being reused as evidence for the second shift.

**Cost.** Composite values with RS missing are based on five active dimensions
and must therefore be identifiable through the accompanying per-query RS
value. The final report should present per-metric results as primary evidence
and composite results as a secondary summary.

**The final report must declare:** RS is a C-subset metric with its own update
clock; composite weights are renormalised over available dimensions; and each
staged transition is evaluated in its own adjacent-regime interval.

---

## §26. Multi-Seed H3 Analysis and Detector-Bank Replay

**Observation.** The v0.2.1 configuration exposed only one operational seed,
although stream generation already supported multiple seeds. Consequently H3's
cross-seed alarm-order claim had no executable analysis. H2 also regenerated
model responses independently for each detector strategy, tripling API cost and
allowing the strategies to see different stochastic samples.

**Change.** `ExperimentConfig.seeds` now controls independent stream and model
realisations. The default experimental configuration uses five seeds, while the
quick smoke preset remains single-seed. Every result row and artefact tag carries
its seed. For each scenario, model, transition, window and strategy, first-alarm
delays are ranked by seed; Kendall's W, a seeded permutation p-value,
`SS < CQ` rate and `CS < CQ` rate are written to
`cross_seed_alarm_order.csv`, with detailed ranks in
`cross_seed_alarm_ranks.csv`.

H2 now generates and scores each experimental cell once. Saved windowed and raw
metric streams are replayed through fresh detector banks for
`distribution_based`, `proposal_mapping` and `adwin_all`. No model call occurs
during replay.

**Why necessary.** H3 concerns reproducibility of an ordering, not a single
realisation. H2 is a paired detector comparison and therefore requires every
strategy to consume identical observations.

**Cost.** Five seeds multiply the primary generation campaign by five. The
detector comparison itself no longer multiplies generation cost. Missed alarms
are tied after observed alarms in the rank matrix; this convention and the
permutation count must be reported.

**The final report must declare:** the exact seed list, permutation count and
seed, missed-alarm ranking convention, pairwise early-warning rates, and that
H2 detector arms were evaluated by replay over identical metric streams.

---

## §27. Difficulty Proxy Audit and Resumable Live Campaigns

**Observation.** The E2H proxy ranked a lexical tuple whose operand count
dominated prompt length and answer magnitude, while the stronger GSM8K signal
in the reference solution was discarded during gold extraction. The live
client also had no retry or resume path: one transient HTTP failure terminated
the campaign, and already-paid calls could be repeated after restart.

**Change.** `Query.solution_steps` preserves the count of GSM8K `<<...>>`
reference operations. Difficulty terciles now use a declared weighted score led
by solution steps, followed by operand count, prompt length and bounded answer
magnitude. Every run writes `difficulty_proxy_validation.json`. When ST3-ST5
are present, `difficulty_cq_validation.csv` checks `CQ_easy > CQ_medium >
CQ_hard` and requires both adjacent gaps to be at least 0.10.

The live client now distinguishes transient HTTP/network failures from
permanent failures, retries transient failures with exponential backoff, and
returns failed calls as verbatim `ok=False` records rather than converting them
into model inconsistency. Individual call records are flushed to JSONL before
the next call and before scoring. With `resume: true`, observations are
reconstructed from that checkpoint, detector state is rebuilt by replay, and
only missing calls are issued, including inside a partially completed K/L
group. A malformed final line caused by a hard interruption is removed while
valid records are retained.
The manifest records failed-call counts and snapshot/query ratios.

**Why necessary.** A controlled difficulty experiment requires evidence that
its tiers differ on a dataset-backed proxy and on observed model accuracy.
Campaign recovery is necessary both for cost control and for preserving the
identity of stochastic samples across interruptions.

**Cost.** The weighted proxy remains a proxy, not an official GSM8K difficulty
label. Failed queries are excluded from all metrics, so failure counts and
coverage must accompany every result. Retry timing can lengthen a run under
provider instability.

**The final report must declare:** the exact proxy formula and coverage of
reference solution steps; tier-level CQ and the 0.10 acceptance rule; retry and
backoff settings; failed-call exclusions; resume usage; and coverage ratios.

---

## §28. Independent ARC-Challenge Replication and Gateway-Control Probe

**Observation.** `replication_dataset` was present in configuration but did not
affect execution, so the planned external replication could be declared without
ever running. The ASU gateway may also expose controls for history, search and
prompt enhancement, but their names and support have not yet been verified.

**Change.** A completed primary campaign may now launch a separate, smaller
ARC-Challenge campaign. The default replication uses ST2, ST3 and ST5, seeds
42/43/44 and 150 loaded items, writes into `replication_arc_challenge`, disables
recursive replication, strategy replay and window ablation, and links its
dataset source, seeds and scenarios from the primary manifest. Stream
construction now explicitly selects the configured dataset in primary, legacy
strategy and ablation paths.

`scripts/probe_gateway.py` provides five minimal probes: baseline, each of
history and search disabled separately, prompt date/time context neutralised,
and all three controls combined. It is a dry run by default. Network calls require
both `--execute` and the named API-key environment variable; retries are
disabled and completions are capped at eight tokens.

**Why necessary.** External replication must be an executable, traceable
campaign rather than dormant metadata. Gateway-side augmentation can confound
reasoning measurements, while unverified request fields can fail a paid run;
isolated probes distinguish unsupported fields before configuration changes.

**Cost.** ARC-Challenge is a smaller validation campaign and does not replace
the preregistered GSM8K primary analysis. The probe can establish request
acceptance only; it does not prove the provider's internal behaviour. No gateway
control is added to the live configuration until a real endpoint probe accepts
it.

**The final report must declare:** primary and replication datasets separately;
replication scenarios, seeds, item count and source; whether gateway controls
were probed; the probe artefact; and the exact accepted `request_options` used
in each live model configuration.

**Verification (2026-09-18).** The ASU production gateway accepted the combined
controls for both `openai/gpt5_5` and `aws/claude5_sonnet`. Production-shaped
probes also accepted `temperature=0.7` and `max_tokens=512`. The retained
artefacts are `results/probe_gpt5_5_production.json` and
`results/probe_claude5_sonnet_production.json`; both report `accepted=true`.

---

## §29. Staged Real-Model Health Pilot

**Observation.** Running a complete 50-query regime before verifying NLI,
BERTScore, answer extraction, checkpointing and stochastic repeat behaviour
would spend API quota without first establishing that the measurement stack is
healthy.

**Change.** `max_queries_per_scenario` provides an explicitly operational stream
cap. The first live pilot uses ST3, one seed, 12 queries, two models, K=5 and
L=3, for 192 calls. It is labelled as a stationary infrastructure and
metric-health pilot and cannot be reported as drift evidence. Publication-mode
preflight remains strict. `fail_on_degenerate: false` allows the paid pilot to
write its manifest when a metric is constant in this very small sample; the
degenerate metric remains recorded. Full and replication campaigns remove the
cap and retain fatal degeneracy checks.

**Why necessary.** This separates endpoint/scorer validation from hypothesis
testing and limits sunk cost if a scorer, parser or gateway behaviour is wrong.

**Cost.** Twelve Easy questions cannot validate ST2 detection, false-alarm
rates, Easy/Medium/Hard CQ separation or H3 alarm order. Passing this pilot only
authorises a larger methodology pilot; it does not authorise publication claims.

**The final report must declare:** that this health pilot was operational only,
its query and call counts, all preflight and degeneracy findings, and that none
of its metric values entered confirmatory analyses.

**Second-stage methodology pilot.** After the health gate passes, an exploratory
pilot uses 12 queries per regime. ST2 therefore contains 36 queries with scaled
boundaries at 12 and 24; ST3, ST4 and ST5 contain 12 queries each. With two
models, K=5 and L=3 this requires 1,152 calls. The scaled `ScenarioSpec`, rather
than truncation, is passed through evaluation and plotting so pilot alarms are
never compared with the full campaign's 50/100 boundaries. These estimates may
decide whether the E2H design proceeds, but remain separate from confirmatory
effect sizes, delays and false-alarm rates.

**Model-role correction.** The 2026-09-18 health and scaled-stream pilots used
`openai/gpt5_5` and `aws/claude5_sonnet`. These are frontier-model engineering
and external-validity observations, not the accepted proposal's main factorial
models. The accepted main set is GPT-4o-mini, Llama-3-8B and Mistral-7B; the
proposal assigns Llama and Mistral confirmatory inference to the local RTX 4090.
Accordingly, the frontier pilot's failure of the GSM8K CQ-gap gate does not by
itself invalidate the main design. A separate ASU-hosted pilot of the three
named models must precede the local confirmatory campaign. The saved frontier
artefacts remain immutable and are not pooled with main-model analyses.

**Main-model gateway verification (2026-09-18).** Production-shaped requests
were accepted for `openai/gpt4o_mini`, `aws/llama3-8b` and `aws/mistral-7b`
with `temperature=0.7`, `max_tokens=512`, history/search disabled and prompt
date/time context neutralised. The retained probe artefacts are
`results/probe_gpt4o_mini_production.json`,
`results/probe_llama3_8b_production.json` and
`results/probe_mistral_7b_production.json`.

---

## §30. Main-Model Methodology Pilot Gate

**Observation (2026-09-18).** The ASU-hosted pilot of the three accepted main
models completed all 1,728 planned calls with no failed calls. Preflight was
clean and the GSM8K proxy itself was structurally monotonic: mean reference
solution steps were 2.10, 2.85 and 4.36 for Easy, Medium and Hard. The observed
stationary CQ values, however, did not pass the preregistered gate requiring
strict monotonicity and both adjacent gaps to be at least 0.10:

| Model | Easy CQ | Medium CQ | Hard CQ | Gate result |
| --- | ---: | ---: | ---: | --- |
| GPT-4o-mini | 1.000 | 1.000 | 0.750 | fail: Easy = Medium |
| Llama-3-8B | 1.000 | 0.917 | 0.500 | fail: Easy-Medium gap = 0.083 |
| Mistral-7B | 0.750 | 0.500 | 0.417 | fail: Medium-Hard gap = 0.083 |

In ST2, no model detected the Easy-to-Medium transition. All three models
detected Medium-to-Hard with CQ CUSUM delays of 3, 1 and 1 queries respectively
and zero false alarms. No monitored metric alarmed before CQ at that transition,
so the short pilot provides no H3 early-warning support. Several stationary
metric/model cells were degenerate; Mistral CS was constant in all three
stationary tiers and its SS values were effectively one, despite
`temperature=0.7`. A direct record audit found exactly one unique raw text
among the five nominal repeats for every one of the 72 Mistral queries. This is
an ASU-endpoint repeat-sampling failure for the present protocol, not evidence
that Mistral is intrinsically perfectly stable. The local RTX 4090 backend must
demonstrate non-identical sampled repeats before its confirmatory run.

**Decision.** Successful execution is not treated as methodological
acceptance. The confirmatory campaign is not authorised from this pilot, and
the pilot effect sizes, delays and false-alarm rates remain exploratory. The
immutable evidence is under `results/main_models_pilot`; the concise assessment
is `results/main_models_pilot/PILOT_ASSESSMENT.md`.

**Preregistered next gate (recorded before the calibration run).** The candidate
scheme uses only the number of `<<...>>` operations in each GSM8K reference
solution: Easy = 1-2, Medium = 3-4, Hard = 5 or more. Items without a positive
reference-step count are excluded and their count is reported. In the first
300 GSM8K test items this yields 111 Easy, 134 Medium, 50 Hard and 5 excluded;
the corresponding mean step counts are 1.883, 3.455 and 5.560. No model output
is used in this assignment.

The first test is a Llama-3-8B-only, CQ-only stationary calibration: ST3, ST4
and ST5 with 30 questions each, `K=1`, `L=0`, one seed, 90 calls total. It is
defined in `configs/step_band_calibration_llama_asu.yaml` and writes to
`results/step_band_calibration_llama`. The existing acceptance rule is frozen:
`Easy > Medium > Hard` and both adjacent CQ gaps must be at least 0.10. Do not
relax that rule or tune cut points after seeing the calibration. Only a pass
authorises a short ST2/K/L validation; it does not authorise the confirmatory
campaign by itself.

**The final report must declare:** all three tier-level CQ profiles; that the
original tercile gate failed; the absence of Easy-to-Medium alarms and of H3
early warning in this pilot; all degenerate cells; the ASU-hosted rather than
local status of Llama/Mistral; and that these observations were excluded from
confirmatory inference.

---

## §31. Llama Solution-Step CQ Calibration Result

**Observation (2026-09-19).** The preregistered Llama-3-8B calibration
completed 90/90 calls in 292.7 seconds with no failed calls. Each stationary
tier contained 30 unique questions and the observed reference-step ranges
matched the declared rule: Easy 1-2, Medium 3-4 and Hard 5-7. The 300-item pool
contained 111 Easy, 134 Medium, 50 Hard and 5 excluded items.

| Tier | Correct | N | CQ |
| --- | ---: | ---: | ---: |
| Easy | 29 | 30 | 0.967 |
| Medium | 24 | 30 | 0.800 |
| Hard | 21 | 30 | 0.700 |

The profile is strictly monotonic. The adjacent gaps are 0.167 and 0.100, so
the frozen `>= 0.10` CQ gate passes. The Medium-Hard gap is exactly on the
acceptance boundary and is therefore treated as a narrow pass, not evidence of
a large or precisely estimated effect.

**Decision.** The fixed solution-step bands are retained as the candidate
primary GSM8K composition rule. This one-model, one-seed CQ calibration
authorises the already specified short ST2/K/L validation only. It does not
authorise the confirmatory campaign, establish cross-model validity, or supply
confirmatory effect sizes. CS, RS and SS degeneracy in this run is expected
from `K=1` and `L=0` and is not interpreted.

**Evidence.** The immutable artefacts are under
`results/step_band_calibration_llama`; the concise audit is
`results/step_band_calibration_llama/CALIBRATION_ASSESSMENT.md`.

**Preregistered cross-model follow-up (recorded before execution).** Before the
more expensive ST2/K/L validation, the same frozen bands and CQ gate will be
tested on GPT-4o-mini and Mistral-7B. The run uses ST3/ST4/ST5, 30 questions per
tier, `K=1`, `L=0`, one seed and 180 calls. Each model must independently show
`Easy > Medium > Hard` with both adjacent gaps at least 0.10. The configuration
is `configs/step_band_calibration_gpt_mistral_asu.yaml` and its isolated output
directory is `results/step_band_calibration_gpt_mistral`. No cut point or gate
may be changed after these outcomes are observed. Both models passing
authorises the Llama ST2/K/L validation; either model failing stops that
progression for methodological reassessment.

---

## §32. GPT/Mistral Cross-Model CQ Gate Result

**Observation (2026-09-19).** The preregistered cross-model calibration
completed 180/180 calls in 1,054.5 seconds with no failures. Every model-tier
cell contained 30 unique questions assigned to the declared reference-step
band.

| Model | Easy CQ | Medium CQ | Hard CQ | E-M gap | M-H gap | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPT-4o-mini | 1.000 | 0.900 | 0.967 | 0.100 | -0.067 | fail |
| Mistral-7B | 0.600 | 0.433 | 0.267 | 0.167 | 0.167 | pass |

GPT-4o-mini violated monotonicity because Hard accuracy exceeded Medium. A
post-gate, call-free diagnostic grouped the same responses by exact reference
step count: GPT-4o-mini scored 19/20 at 5 steps, 6/6 at 6 steps and 4/4 at 7
steps. This indicates a GSM8K ceiling for GPT-4o-mini rather than a nearby
step threshold that could be defensibly adjusted. The diagnostic is
exploratory and does not alter the frozen gate result.

**Decision.** The cross-model gate fails because both remaining models were
required to pass. Do not start the planned ST2/K/L validation. The fixed
solution-step bands have evidence of separation for Llama-3-8B and Mistral-7B
but are not a common difficulty control for all three proposal models. Cut
points must not be tuned after observing these results. The E2H-GSM8K primary
design now requires methodological reassessment against the accepted proposal
before any further paid campaign.

**Evidence.** Immutable artefacts are under
`results/step_band_calibration_gpt_mistral`; the concise audit is
`results/step_band_calibration_gpt_mistral/CALIBRATION_ASSESSMENT.md`.

---

## §33. Freeze E2H-GSM8K and Restore the Accepted Proposal S1

**Decision before execution (2026-09-19).** The E2H-GSM8K design is retained
with its code path and all existing artefacts, but it is frozen after the
cross-model CQ gate failure in §32. It is not deleted, rewritten or pooled with
the next method. Work resumes on the accepted proposal's multi-dataset design,
beginning with S1 and the two models requested for the first gate:
GPT-4o-mini and Llama-3-8B.

**Change.** A separate `stream_design: proposal_multidataset` catalogue was
added; the existing `e2h_gsm8k` catalogue remains the default for backward
reproducibility. Only proposal S1 is implemented at this gate. It uses six
equal phases whose GSM8K/ARC-Challenge/MATH proportions progress as
`100/0/0`, approximately `70/30/0`, `30/70/0`, `0/70/30`, `0/30/70`, and
`0/0/100`. The construction uses each dataset equally often overall and draws
without replacement under a fixed seed. A stream-plan CSV records every
timestamp, phase, dataset and question id.

ARC-Challenge prompts now include their labelled answer choices. The obsolete
`lighteval/MATH` Hub id was replaced by `EleutherAI/hendrycks_math`; its seven
test configurations are sampled round-robin so a limited load is not dominated
by one mathematical subject. Balanced nested `\boxed{...}` extraction replaces
the previous single-level regular expression. On Windows, the loader uses the
`truststore` package when available so TLS verification uses the system trust
store; certificate verification is not disabled. Scientific configs can set
`allow_synthetic_datasets: false`, making any real-data load failure fatal
before model calls.

**Pilot protocol.** The preregistered configuration is
`configs/proposal_s1_pilot_gpt_llama_asu.yaml`. It uses one seed, 30 unique
questions per dataset (90 per model), W=10, K=1 and L=0. The resulting 180
calls match the cost of the immediately preceding two-model CQ calibration.
This is a CQ and data-path gate; CS, RS and SS are not interpreted.

**Observed result (2026-09-19).** All 180 calls completed in 905.7 seconds with
zero failures. All three 300-item loader pools were real Hugging Face data and
synthetic fallback was disabled. Dataset-level CQ was:

| Model | GSM8K | ARC-Challenge | MATH |
| --- | ---: | ---: | ---: |
| GPT-4o-mini | 1.000 | 0.800 | 0.333 |
| Llama-3-8B | 0.800 | 0.767 | 0.200 |

Both profiles are monotonic at dataset level. Windowed CQ-CUSUM alarmed at
t=34 for GPT-4o-mini and t=46 for Llama-3-8B, against the declared gradual
midpoint t*=45. The six 15-question phase means were not strictly monotonic,
and no proposal-S1 numerical acceptance threshold had been frozen before this
first run. Therefore the result is recorded as evidence that the S1 route is
worth retaining, not as a post-hoc pass of a confirmatory gate. One S1 stream
cannot estimate false-alarm specificity.

**Cost and next gate.** The design intentionally changes dataset, domain and
difficulty together, as stated in the accepted proposal. Raw S1 alarms measure
operational performance drift under that composite input/task shift; they must
not be described as a model internally degrading over time. Before a paid
K=5/L=3 run, add stationary per-dataset controls and a composition-matched
globally shuffled control, then record their acceptance criteria before seeing
outcomes. The full audit is
`results/proposal_s1_pilot_gpt_llama/PILOT_ASSESSMENT.md`.

**The final report must declare:** that E2H-GSM8K was frozen rather than
deleted; the separate proposal stream catalogue; exact six-phase composition;
real dataset ids and hashes; ARC choice rendering and the seven-config MATH
sampling rule; K=1/L=0 metric limits; all CQ values and alarm times; the
non-monotonic phase means; and that the pilot was excluded from confirmatory
inference.

---

## Section 34. Preregistered S1 Temporal-Order Control Gate

**Recorded before control execution (2026-09-19).** The completed S1 pilot's
saved responses will be replayed without any model or network call. The
primary control metric is CQ, matching the declared K=1/L=0 pilot decision
metric. Every arm receives fresh rolling-metric and detector state at W=10
under the source run's `distribution_based` strategy.

Four negative-control arms are fixed. `global_shuffled` uses exactly the same
90 questions and responses, preserving the 30/30/30 dataset totals while
destroying their temporal composition order. `stationary_gsm8k`,
`stationary_arc_challenge` and `stationary_math` each use the exact 30 saved
question-response pairs for that dataset in an independently shuffled fixed
order. The control seed is 10042. Original response files are immutable; all
derived artefacts are written below
`results/proposal_s1_pilot_gpt_llama/order_controls` with source-file SHA-256
hashes.

**Frozen gate.** For each model, a CQ-CUSUM alarm must be reproduced on the
ordered S1 stream, with exact alarm-time parity against the existing result.
The matched global shuffle and all three stationary arms must each produce
zero windowed CQ-CUSUM alarms. Thus any CQ-CUSUM event in a control arm is a
false alarm and fails that model's gate. Raw CQ-DDM is descriptive and is not
part of this gate. Both models must pass before a paid K=5/L=3 S1 validation
is authorised. A failure may be reassessed only under a newly declared control
protocol; the rule above must not be relaxed after observing these outcomes.

**Observed result (2026-09-19).** Call-free replay reproduced every existing
ordered-S1 CQ-CUSUM alarm exactly: GPT-4o-mini at t=34, 55 and 76; Llama-3-8B
at t=46, 67 and 88. All six model-by-stationary-control cells produced zero
CQ-CUSUM alarms. The matched global shuffle nevertheless alarmed for both
models: GPT-4o-mini at t=30, and Llama-3-8B at t=42, 63 and 86. Overall CQ and
the 30/30/30 item composition were identical to ordered S1 in each shuffled
arm, so these are false alarms under the frozen rule. Source response hashes
remained fixed and the control manifest records `no_model_calls: true`.

**Decision.** Both models fail the strict control gate. The clean stationary
arms show that a single homogeneous 30-item block did not trigger CUSUM, but
the matched-shuffle failures show that incidental order in a heterogeneous
mixture is sufficient to trigger it. The ordered S1 alarms therefore cannot
yet be attributed specifically to the designed composition trajectory. Do not
start the paid K=5/L=3 S1 validation. Any permutation-ensemble calibration or
detector retuning must be declared as a new protocol and must not convert this
failed gate into a pass retroactively. The full call-free audit is
`results/proposal_s1_pilot_gpt_llama/order_controls/CONTROL_ASSESSMENT.md`.

---

## Section 35. Preregistered S1 CQ Permutation Reassessment

**Recorded before permutation execution (2026-09-19).** The failed single-
shuffle gate in Section 34 remains failed and will not be reclassified. A new,
call-free protocol will quantify two separate questions using the exact saved
90 CQ outcomes per model: whether ordered S1 carries a temporal decline, and
whether the present W=10 CQ-CUSUM controls stream-level false alarms.

The null ensemble contains 1,000 unique, non-identity permutations generated
with seed 21042. Each permutation preserves all questions, outputs, CQ values
and 30/30/30 dataset totals. The same index permutations are applied to both
models. No model, gateway or network call is permitted. Ordered CUSUM alarm
times must first reproduce the archived values exactly.

**Frozen statistics and gate.** Temporal decline is the negative ordinary
least-squares slope of raw per-query CQ on time normalized to [0,1]; larger
values mean a stronger decline. Its one-sided randomization p-value is
`(1 + count(null >= observed)) / 1001`. Each model must have p <= 0.025,
Bonferroni-correcting family-wise alpha 0.05 across the two models. CUSUM
specificity is the proportion of permuted streams with at least one windowed
CQ-CUSUM alarm. For each model, the upper bound of its two-sided 95% Wilson
interval must be <= 0.05. Both criteria must pass for both models before this
new gate can authorise planning a paid K=5/L=3 validation. Alarm count and
first-alarm extremeness are descriptive only.

If temporal decline passes but specificity fails, the conclusion is limited:
the S1 ordering carries a CQ trend, but the current CUSUM calibration is not
fit for confirmatory use. Threshold selection must not be tuned post hoc on
the ordered outcomes. Any later detector-calibration rule and validation split
must be recorded before execution.

**Observed result (2026-09-19).** Ordered alarm replay again matched the archive
exactly. The raw-CQ decline statistic was 0.9026 for GPT-4o-mini and 0.7890 for
Llama-3-8B. Both one-sided randomization p-values reached the 1,000-permutation
floor, p=0.000999, and therefore passed the corrected p<=0.025 temporal-signal
criterion. In contrast, CQ-CUSUM alarmed in 991/1,000 GPT permutations and
995/1,000 Llama permutations. The false-alarm rates and 95% Wilson intervals
were 0.991 [0.983, 0.995] and 0.995 [0.988, 0.998], respectively. Ordered first
alarms were not extreme under the null: their descriptive earliness p-values
were 0.654 and 0.880.

**Decision.** The new gate fails solely and decisively on detector specificity.
The accepted-proposal S1 ordering carries a reproducible downward CQ trend in
these saved outputs, but the current W=10 CQ-CUSUM threshold is not usable for
confirmatory detection: almost every random ordering alarms. Do not start the
paid K=5/L=3 validation with this detector calibration. The failed Section 34
gate also remains failed. A future calibration must use a predeclared
threshold-selection rule and a separate validation partition or seed set; it
must preserve these results rather than overwrite them. Full evidence is in
`results/proposal_s1_pilot_gpt_llama/order_controls/permutation_test/`.

---

## Section 36. Preregistered Split-Sample CQ-CUSUM Calibration

**Recorded before calibration execution (2026-09-19).** Section 35 supports
the S1 temporal CQ signal but rejects the default CUSUM threshold. This new
call-free gate tests whether a single common threshold can control false alarms
without losing ordered-S1 sensitivity. It does not alter or replace either
earlier failed gate.

Two fresh, matched and mutually disjoint null sets are fixed: 1,000 calibration
permutations with seed 31042 and 1,000 validation permutations with seed 41042.
The same index permutations are used for both models. W=10, drift allowance
0.01 and detector warm-up 20 remain fixed. Only the CUSUM threshold may vary,
over the frozen grid `[0.15, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00,
4.00, 5.00, 7.50, 10.00]`.

**Selection and locked gate.** Select the smallest single threshold whose
calibration permutation-level any-alarm rate has a two-sided 95% Wilson upper
bound <=0.05 for both models. If no grid value qualifies, the gate fails
without validation. Otherwise lock that threshold before reading validation
results. On the fresh validation set, each model must again have a 95% Wilson
upper bound <=0.05. The same locked threshold must also produce at least one
alarm on each ordered S1 stream at t in [15,90]. Both held-out specificity and
ordered sensitivity must pass for both models. The already-passed Section 35
Bonferroni-corrected temporal-trend test is a prerequisite.

No model or network call is allowed. If specificity passes but ordered
sensitivity fails, the present CUSUM form is treated as unable to separate S1
from chance ordering at the required error rate. It must not be rescued by
choosing a different threshold after validation outcomes are seen.

**Observed result (2026-09-19).** Thresholds 0.15 through 5.0 failed the common
calibration specificity rule. Thresholds 7.5 and 10.0 passed; the frozen
smallest-threshold rule therefore selected and locked 7.5. At 7.5, calibration
false-alarm counts were 12/1,000 for GPT-4o-mini and 22/1,000 for Llama-3-8B,
with Wilson upper bounds 0.021 and 0.033.

On the fresh validation permutations, GPT-4o-mini alarmed in 15/1,000 streams,
rate 0.015, 95% CI [0.009, 0.025]; Llama-3-8B alarmed in 33/1,000, rate 0.033,
95% CI [0.024, 0.046]. Both satisfy the locked <=0.05 upper-bound rule. The
same threshold produced one ordered-S1 alarm per model, at t=80 and t=82, so
the preregistered sensitivity condition also passed.

**Decision and limitation.** Section 36 passes and authorises planning, not
executing, the paid K=5/L=3 validation with a locked CQ threshold of 7.5. The
alarms are late: delays are +35 and +37 relative to the declared S1 midpoint
t*=45. No delay limit was part of this gate, so these descriptive delays do not
retroactively change its result; they do show that the calibrated detector is
not an early-warning signal. Before the paid run, its protocol must freeze an
acceptable CQ delay criterion, propagate the explicit threshold through the
experiment configuration, and state that Sections 34 and 35 remain failed.
Evidence is under
`results/proposal_s1_pilot_gpt_llama/order_controls/cusum_calibration/`.

---

## Section 37. Preregistered Paid S1 K/L Metric-Health Validation

**Recorded before execution (2026-09-19).** The paid run is prepared but not
started. Its configuration is
`configs/proposal_s1_kl_validation_gpt_llama_asu.yaml` and its isolated output
directory is `results/proposal_s1_kl_validation_gpt_llama`. It uses the same
real-data S1 construction, seed 42, 30 unique questions per dataset, W=10,
GPT-4o-mini and Llama-3-8B. K=5 and L=3 produce 90 x 8 x 2 = 1,440 resumable
model calls. Gateway history/search/date/time controls remain disabled.

The CQ detector is locked to CUSUM threshold 7.5, drift allowance 0.01 and
warm-up 20 from Section 36. The experiment configuration now propagates
explicit `detector_overrides` into `MonitorConfig` and records them in the
manifest. Combining overrides with the H2 strategy comparison is forbidden,
because an override would mask the strategy-arm difference.

**Prospective CQ delay gate.** The calibrated detector must produce an eligible
ordered-S1 CQ alarm for every model no later than one complete W=10 window
after the effective midpoint t*=45: signed delay <=+10, with zero alarms before
the transition start t=15. This criterion is intentionally stricter than the
descriptive +35/+37 delays observed in Section 36; those old outcomes do not
pass it. The paid run must write `proposal_s1_cq_delay_gate.csv`, and both model
cells must pass before any confirmatory progression.

**Metric-health gate.** Preflight must have no problems; all 1,440 planned
records must exist with zero failed calls; every metric must remain in [0,1]
where defined; RS must be evaluable on at least 20 correct-primary queries per
model; and no windowed metric may be degenerate. At least 80% of questions per
model must have two or more distinct verbatim completions among their five K
runs, which rejects endpoint caching while allowing stable final answers.

This is not an H3 confirmatory run. CQ is the only calibrated detector. CS, RS,
LS, ES and SS values may be used for metric-health assessment, but their alarm
times and ordering remain exploratory until each detector is calibrated with
call-free null permutations from these saved records. Failure of any gate
stops progression; thresholds and criteria must not be relaxed after outcomes
are observed.

The post-run criteria are implemented before execution in
`scripts/assess_s1_kl_validation.py`; it writes `kl_validation_gate.csv`,
`kl_validation_manifest.json` and `KL_VALIDATION_ASSESSMENT.md` without making
model calls.

---

## Section 38. Paid S1 K/L Validation Result

**Observed after the frozen Section 37 protocol (2026-09-19).** The run
completed all 1,440 planned calls: 720 records and 90 queries for each model,
with zero failed calls. Preflight had no problems, no metric was degenerate,
and every non-RS metric was evaluable for all 90 queries per model. RS was
evaluable for 63 GPT-4o-mini and 52 Llama-3-8B queries. All defined metric
values were finite and in [0,1]. Verbatim K-repeat diversity passed at 90/90
(1.000) for GPT-4o-mini and 83/90 (0.922) for Llama-3-8B. Thus every frozen
metric-health criterion passed.

The preregistered CQ-delay criterion failed for both models. With the locked
CUSUM threshold 7.5, GPT-4o-mini first alarmed at t=82, delay +37 from t*=45;
Llama-3-8B first alarmed at t=79, delay +34. Neither model produced an alarm
before the transition start, so the failure is exclusively late detection,
not false alarm. This closely reproduces the late +35/+37 behavior from the
K=1/L=0 pilot.

The descriptive CQ decline is nevertheless substantial: dataset-level CQ was
0.933/0.833/0.333 for GPT-4o-mini and 0.800/0.767/0.167 for Llama-3-8B across
GSM8K/ARC-Challenge/MATH. Windowed pre/post CQ changes were -0.347 and -0.387,
respectively, with nominal Mann-Whitney p-values below 1.2e-8. These values
characterize the composite task shift; they do not rescue the failed
prospective early-warning gate.

**Decision.** The overall Section 37 gate fails, solely because CQ does not
meet the locked maximum delay of +10. Do not relax threshold 7.5 or the delay
criterion after observing this result, and do not treat non-CQ alarm ordering
or the two-cell F1 table as confirmatory. The saved K/L records may be used in
a separately preregistered, call-free detector-development analysis, but any
resulting detector requires a fresh held-out stream before confirmatory use.
Evidence is in `results/proposal_s1_kl_validation_gpt_llama/`, especially
`KL_VALIDATION_ASSESSMENT.md`, `data/kl_validation_gate.csv`, and
`kl_validation_manifest.json`.

---

## Section 39. Preregistered Call-Free Detector Development

**Recorded before execution (2026-09-19).** This analysis is method
development, not a confirmatory H3 test. It uses only the 1,440 saved calls
from Section 38 and makes no model or network calls. The locked CQ-CUSUM
threshold 7.5 is not changed. The five non-CQ metrics use the project
proposal's fixed mapping: CS->ADWIN, RS->DDM, LS->Page-Hinkley, ES->EWMA and
SS->ADWIN. All metrics retain their preregistered higher-is-better direction;
no direction may be flipped after viewing ordered results.

The sensitivity grids, listed from most to least sensitive, are fixed as
follows: ADWIN delta `[0.05, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001]` for
CS and SS; DDM drift level `[2, 2.5, 3, 3.5, 4, 5, 6, 8]` for RS;
Page-Hinkley threshold `[0.1, 0.15, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5,
10]` for LS; and EWMA control limit `[2, 2.5, 3, 3.5, 4, 4.5, 5, 6]` for ES.
Detector warm-ups and all other parameters remain at their code defaults.

Use 1,000 unique matched calibration permutations with seed 51042 and 1,000
disjoint matched validation permutations with seed 61042. The same item-order
permutations apply to every metric and model. For each metric, select using
calibration null streams alone the first, most-sensitive parameter whose
two-sided 95% Wilson upper bound on any-alarm probability is <=0.005 for both
models. The 0.005 cell limit is a Bonferroni control for five metrics by two
models, giving a family-wise bound of 0.05. A selected value must independently
meet the same bound for every model on held-out validation permutations.

Window signals are reconstructed after each permutation. CQ, CS, RS, LS and
SS use the exact W=10 `WindowMeanMetric` behavior, including RS C-subset NaN
skipping. ES is recomputed from saved CQ and primary token counts inside every
permuted W=10 window; its already-saved order-dependent `ES_raw` value is not
permuted. An early-warning development candidate must pass held-out
specificity and first alarm in the frozen interval t=15..44 for both models.
Ordered outcomes are read only after calibration parameters are locked.

Passing candidates still require a fresh held-out model-call stream before
confirmatory use. If none passes, do not reverse metric directions, expand
grids, relax error control or tune against the ordered outcomes. The
implementation is `scripts/develop_s1_detectors.py` and writes under
`results/proposal_s1_kl_validation_gpt_llama/detector_development/`.

**Observed result (2026-09-19).** Calibration selected ADWIN delta 0.05 for
CS and SS and Page-Hinkley threshold 7.5 for LS. Each of these three settings
then produced 0/1,000 held-out null alarms in both models, Wilson upper 0.0038,
but produced no alarm at all on either ordered S1 stream. They therefore pass
specificity and fail early-warning sensitivity.

No common RS or ES parameter qualified in calibration. Even the least
sensitive RS-DDM drift level 8 produced 2/1,000 null alarms for GPT-4o-mini,
Wilson upper 0.0073, although Llama-3-8B had zero. At the least sensitive
ES-EWMA control limit 6, GPT-4o-mini produced 5/1,000 null alarms, upper
0.0117, and Llama-3-8B produced 10/1,000, upper 0.0183. All exceed the frozen
0.005 cell limit.

**Development decision.** No proposal-mapped metric is an early-warning
candidate under the preregistered family-wise specificity control. Together
with the late CQ alarms in Section 38, this means the current proposal-mapped
detector design does not support H3 on this S1 stream. Do not extend grids or
tune against the ordered outcomes. This is a valid negative method result;
any revised detector family must be declared as a new exploratory method and
validated on a fresh held-out model-call stream. Evidence is in
`results/proposal_s1_kl_validation_gpt_llama/detector_development/`.

---

## Section 40. Reopened E2H Llama Solution-Step K/L Pilot

**Recorded before execution (2026-09-19).** Sections 38-39 remain immutable
negative results for the accepted-proposal multi-dataset S1 branch. E2H is
reopened as a separate controlled method-development branch, not as a
replacement or post-hoc rescue of S1. The motivation is that E2H changes
difficulty composition inside GSM8K while holding dataset and domain fixed.
The earlier Llama stationary calibration passed the frozen solution-step CQ
gate at 0.967/0.800/0.700, whereas multi-dataset S1 CQ detection was late.

The new configuration is `configs/e2h_stepband_llama_kl_pilot_asu.yaml` and
writes to `results/e2h_stepband_llama_kl_pilot`. It uses only Llama-3-8B,
new seed 43, ST2, W=10, and the unchanged model-independent reference-solution
bands Easy=1-2, Medium=3-4 and Hard>=5. Each regime contains 30 unique
questions. K=5 and L=3 yield 90 x 8 = 720 resumable calls. Gateway history,
search, date and time augmentation remain disabled. The detector strategy is
the accepted proposal mapping, but its run-time alarms are exploratory.

**Frozen pilot gate.** All 720 records and 90 queries must exist with zero
failed calls; preflight must be clean; every non-RS metric must be evaluable on
all 90 queries and RS on at least 20; values must be finite and in [0,1]; no
metric may be degenerate; and at least 80% of questions must have two distinct
verbatim completions among K=5. The stream must contain exactly 30 Easy, then
30 Medium, then 30 Hard items, all matching the fixed step bands. CQ must be
strictly Easy > Medium > Hard with both adjacent gaps >=0.10. Detector alarms
do not affect this first gate. `scripts/assess_e2h_llama_kl_pilot.py`
implements these criteria before execution.

**Frozen call-free follow-up.** Only a passing pilot gate authorises detector
development on the saved records. Use 1,000 matched calibration permutations
with seed 71043 and 1,000 disjoint validation permutations with seed 81043.
Use the proposal mapping CQ->CUSUM, CS->ADWIN, RS->DDM, LS->Page-Hinkley,
ES->EWMA and SS->ADWIN. CQ uses threshold grid `[0.15, 0.25, 0.5, 0.75, 1,
1.5, 2, 3, 4, 5, 7.5, 10]`; the five non-CQ grids remain exactly those in
Section 39. Select the most sensitive common value from calibration nulls
only. With six metric-model cells, each held-out two-sided 95% Wilson upper
bound must be <=0.05/6. Ordered alarms are inspected only after locking.

An exploratory H3 candidate requires calibrated CQ to alarm no later than ten
queries after the Medium-to-Hard boundary t=60 and a calibrated non-CQ metric
to alarm at t>=60 strictly before CQ, with no alarm before the first boundary
t=30. A pass authorises only a fresh-seed replication; it is not confirmatory.
Failure stops this E2H route without changing bands, grids or alarm direction.

**Observed result (2026-09-19).** The run completed all 720 calls and 90
queries with zero failures. Preflight was clean, no metric was degenerate, all
metric coverage and range checks passed, RS was evaluable on 74 queries, and
79/90 questions had at least two distinct K completions (0.878). The complete
metric-health gate therefore passed.

The frozen CQ band gate failed. Easy was 27/30 = 0.900, Medium was 22/30 =
0.733 and Hard was 25/30 = 0.833. The Easy-Medium gap was +0.167, but the
Medium-Hard gap was -0.100, violating both strict monotonicity and the required
positive adjacent gap. A decision-neutral exact-step audit showed CQ 1.000 at
one step (6 items), 0.875 at two (24), 0.857 at three (21), 0.444 at four (9),
0.813 at five (16), 0.909 at six (11), and 0.667 at seven (3). Thus reference
solution-step count did not monotonically order Llama difficulty in this new
sample. Only 31/90 questions overlapped the seed-42 calibration; primary CQ
changed on 3/31 overlapping questions, so the failed replication is driven
mainly by the new item sample rather than repeat stochasticity alone.

**Decision.** The overall Section 40 pilot gate fails despite healthy K/L
metrics. The planned call-free detector calibration is not authorised and is
not run. Default run-time alarms remain exploratory and must not be used to
rescue the gate. Do not change the 1-2/3-4/5+ bands after observing this
outcome. The fixed solution-step E2H construction is not a reproducible
model-independent difficulty control for this project, so this E2H route stops
here. Evidence is under `results/e2h_stepband_llama_kl_pilot/`, especially
`E2H_KL_PILOT_ASSESSMENT.md`, `data/e2h_kl_pilot_gate.csv` and
`e2h_kl_pilot_manifest.json`.

---

## Section 41. Official Easy2Hard-Bench Calibration

**Recorded before execution (2026-09-19).** Section 40 remains an immutable
negative result for the custom GSM8K reference-solution step bands. That route
is preserved in code and results, but is not reused or relabelled. The new
route uses the published `furonghuang-lab/Easy2Hard-Bench` `E2H-GSM8K`
configuration and its external IRT-derived `rating` and `rating_quantile`
fields, pinned to dataset revision `55bc0d2fb10954151e669d2026b87fa896f2fa26`.
It is implemented as the distinct `e2h_benchmark_gsm8k` stream design;
the frozen legacy `e2h_gsm8k` stream remains available.

Before any model call, item identity fixes a disjoint 40% calibration and 60%
holdout split using SHA-256 of `e2h-v1:<qid>`. This split never depends on run
seed or model outcomes. Published `rating_quantile` defines Easy `(0,1/3]`,
Medium `(1/3,2/3]` and Hard `(2/3,1]`. The calibration configuration is
`configs/e2h_benchmark_calibration_main_models_asu.yaml`: seed 44, GPT-4o-mini,
Llama-3-8B and Mistral-7B, stationary ST3/ST4/ST5, 30 unique calibration items
per band and model, K=1/L=0. The total budget is 270 resumable calls. Synthetic
fallback is forbidden.

**Frozen model-specific gate.** For each model separately, all 90 primary
records must exist, have no call failures, contain the external ratings, and
belong to the calibration partition. CQ is the only decision metric. Fit the
ordinary least-squares slope of binary CQ on continuous Easy2Hard `rating`.
Using 5,000 fixed-seed (`92044`) permutations of CQ over the fixed ratings,
the one-sided probability of a slope at least as negative as observed must be
at most `0.05/3 = 0.0166667`. In addition, mean Easy CQ minus mean Hard CQ must
be at least 0.15. Both conditions are required. There is no adjacent-band
requirement and one model cannot rescue another.

Only models passing their own gate are eligible for a fresh K=5/L=3 run on
the untouched holdout partition. A passing calibration is exploratory method
validation, not confirmation of H3. Failed models stop here; bands, split,
test direction, alpha, permutations and gap threshold must not be changed
after observing results. `scripts/assess_e2h_benchmark_calibration.py` applies
the gate without making model calls.

**Dataset-only preflight (2026-09-19).** The pinned revision loaded 1,319 real
rows with content hash `7007d832a0a0dd00`. The fixed calibration partition
contains 185 Easy, 174 Medium and 187 Hard candidates. Constructed ST3/ST4/ST5
streams each contain 30 unique questions, and the three selected sets are
pairwise disjoint. No model endpoint was called during this preflight.

**Observed result (2026-09-19).** All 270 planned calls completed with zero
failures: 30 unique calibration questions in each band for each of the three
models. GPT-4o-mini CQ was 0.967/0.967/0.867 (Easy/Medium/Hard), Easy-Hard gap
0.100, continuous rating slope -0.2111 and one-sided permutation p=0.04319.
Llama-3-8B CQ was 0.833/0.833/0.733, gap 0.100, slope -0.1465 and p=0.22276.
Both models fail both frozen criteria. Their high, equal Easy/Medium accuracy
is consistent with a ceiling effect; this description does not alter the gate.

Mistral-7B CQ was 0.633/0.467/0.233, gap 0.400, slope -0.8750 and p=0.000200.
It passes both the Bonferroni-adjusted slope test and the minimum gap. Therefore
Mistral-7B alone is eligible for the disjoint holdout K=5/L=3 run. GPT-4o-mini
and Llama-3-8B stop on this branch and must not be carried into that run. This
is model-specific calibration evidence, not evidence that the official labels
separate difficulty for every model. The call-free report and source hashes are
under `results/e2h_benchmark_calibration_main_models/`.

---

## Section 42. Mistral Official-E2H Holdout K/L Validation

**Recorded before execution (2026-09-19).** Section 41 selected Mistral-7B as
the only model eligible for holdout validation. GPT-4o-mini and Llama-3-8B are
excluded. The configuration is
`configs/e2h_benchmark_mistral_holdout_kl_asu.yaml` and writes only to
`results/e2h_benchmark_mistral_holdout_kl`. It uses seed 45, the pinned official
E2H-GSM8K revision and the untouched SHA-256 `holdout` partition. ST2 contains
30 unique Easy, then 30 unique Medium, then 30 unique Hard questions. K=5 and
L=3 produce 90 x 8 = 720 resumable calls. W=10, the proposal detector mapping,
temperature 0.7 and disabled gateway history/search/date/time augmentation are
unchanged. Real NLI and BERTScore are required by publication mode. Synthetic
fallback is forbidden.

**Frozen completion and metric-health gate.** All 720 raw records and 90
per-query rows must exist with exactly five primary and three paraphrase calls
per unique question. All questions must belong to the holdout partition, match
their published quantile band and occur in the declared Easy/Medium/Hard phase
order. Raw, per-query and manifest failure counts must all be zero, and scorer
preflight must have no problems. Every non-RS metric must be finite in [0,1]
for all 90 questions; RS must satisfy the same range and be evaluable for at
least 20. No metric may be degenerate. At least 80% of questions must have two
distinct verbatim completions among K=5.

**Frozen CQ generalisation gate.** Binary primary-run CQ is regressed on the
continuous official `rating`. A fixed 5,000-permutation one-sided test with
seed 93045 must give a negative slope with p<=0.05. Mean Easy CQ minus mean
Hard CQ must independently be at least 0.15. Both criteria and the complete
metric-health gate are required. There is no adjacent-band threshold.

Run-time detector alarms are descriptive only and cannot pass or fail Section
42. A passing run authorises a separately preregistered, call-free detector
analysis on the saved K/L records; it does not itself confirm H3. A failed run
stops this branch without changing the partition, labels, CQ test, thresholds,
metric rules or failure criteria. The frozen assessor is
`scripts/assess_e2h_mistral_holdout.py` and makes no model calls.

**Superseded before execution (2026-09-19).** No Section 42 model call was
made. Running Mistral alone would not first resolve the ceiling-limited
GPT-4o-mini and Llama-3-8B operating ranges required by the proposal's
three-model comparison. The configuration and assessor remain as an immutable
record but must not be run. Section 43 now precedes any K/L holdout campaign.

---

## Section 43. GPT/Llama Extreme-Range Calibration

**Recorded before execution (2026-09-19).** This is transparent method
development prompted by the Section 41 ceiling effect; it does not reinterpret
or replace that negative result. The original SHA-256 split remains unchanged.
The untouched 60% `holdout` pool is deterministically divided, using an
independent SHA-256 key `e2h-v2:<qid>`, into equal `range_calibration` and
`final_holdout` subsets. Original Section 41 calibration items can enter
neither subset.

The active configuration is
`configs/e2h_extreme_calibration_gpt_llama_asu.yaml`. It uses seed 46,
GPT-4o-mini and Llama-3-8B, official E2H-GSM8K, K=1/L=0 and stationary
ST3/ST4/ST5 streams. Difficulty bands are fixed before calls as the bottom 20%
`[0,0.20]`, central 20% `[0.40,0.60]` and top 20% `[0.80,1]` of published
`rating_quantile`. Each model receives 30 unique range-calibration questions
per band, for 180 resumable calls total. Synthetic fallback is forbidden.

**Frozen model-specific gate.** Each model must have 90 unique primary records,
30 in every band, with zero failures, complete external metadata, correct
extreme-band membership and no item outside `range_calibration`. For each model
separately, regress binary CQ on continuous official `rating`. With 5,000
fixed-seed (`94046`) one-sided permutations, the negative-slope p-value must be
at most `0.05/2 = 0.025`. Mean Easy CQ minus mean Hard CQ must independently be
at least 0.15. Both criteria are required; one model cannot rescue the other.

A passing model becomes eligible for a K/L design on `final_holdout`. A failed
model may proceed only to the already declared harder official E2H-AMC fallback;
the GSM8K bands, alpha, gap, split and test direction cannot be tuned after
viewing Section 43. Mistral's next K/L design is also deferred until these
outcomes determine the common comparison plan. The call-free assessor is
`scripts/assess_e2h_extreme_calibration.py`.

**Dataset-only preflight (2026-09-19).** The fixed extreme bands contain
78/79/64 Easy/Medium/Hard candidates in `range_calibration` and 80/85/72 in
`final_holdout`. The configured streams select 90 unique questions, all in
`range_calibration`, with zero overlap against every Section 41 calibration
question. No model endpoint was called during this preflight.

**Observed result (2026-09-19).** All 180 planned calls completed with zero
failures and 90 unique range-calibration questions per model. GPT-4o-mini CQ
was 1.000/1.000/0.867 (Easy/Medium/Hard), with continuous rating slope -0.2355
and one-sided permutation p=0.001600. The slope criterion passes, but the
Easy-Hard gap is 0.133, below the frozen 0.15 threshold; GPT-4o-mini therefore
fails Section 43. The margin corresponds to one answer among 30 Hard items,
but the threshold must not be relaxed after observing that fact.

Llama-3-8B CQ was 1.000/0.833/0.633, gap 0.367, slope -0.5808 and p=0.000200.
It passes both frozen criteria and is eligible for a final-holdout K/L design.
GPT-4o-mini is not eligible for GSM8K K/L and proceeds only to the declared
E2H-AMC fallback. Mistral's common K/L design remains deferred until the GPT
fallback and cross-model comparison plan are fixed. Evidence and source hashes
are under `results/e2h_extreme_calibration_gpt_llama/`.

---

## Section 44. GPT-4o-mini Official E2H-AMC Fallback Calibration

**Recorded before execution (2026-09-19).** Section 43 did not validate a
usable E2H-GSM8K range for GPT-4o-mini under the frozen Easy-Hard gap rule. This
fallback changes the task set, not the observed GSM8K bands or threshold. It
uses the official Easy2Hard-Bench `E2H-AMC` `eval` split at the same pinned
revision `55bc0d2fb10954151e669d2026b87fa896f2fa26`. The 2,975-row parquet has
LFS SHA-256 `2a04f50a80a16bfc5c1ff5667f9403e238184bbe7c325eff99f684d89f8f97fe`
and the framework query-content hash is `b7a9dbd7a037f560`. Synthetic fallback
is forbidden.

The active configuration is
`configs/e2h_amc_calibration_gpt4o_mini_asu.yaml`. It uses seed 47,
GPT-4o-mini, K=1/L=0 and stationary ST3/ST4/ST5 streams. Published
`rating_quantile` terciles define Easy `[0,1/3]`, Medium `(1/3,2/3]` and Hard
`(2/3,1]`. The dataset-specific `e2h-v1:<qid>` SHA-256 split assigns 40% to
`calibration` and leaves 60% as untouched `holdout`. Each stream contains 30
unique calibration questions, for 90 resumable primary calls total. Gateway
history, search, date and time augmentation remain disabled; temperature is
0.7 and maximum output length is 512 tokens.

**Frozen AMC answer scoring.** E2H-AMC gold answers include exact fractions,
radicals, powers and pi expressions. The generic last-number extractor is
therefore prohibited. The dataset-specific scorer extracts a final `Answer:`
or balanced `\\boxed{...}` value and uses exact symbolic equivalence for scalar
LaTeX expressions. Comma-separated/multiple-value answers, `\\pm` answers and
expressions with integer exponents or factorials above 100 use conservative
normalized-surface equality to prevent uncontrolled symbolic expansion. This
rule and its SymPy/ANTLR dependency are frozen before calls.
The rqeval adapters retain rqeval's metric formulas but route E2H-AMC answer
canonicalization through this matcher for CQ and the correctness components of
CS, RS and ES; otherwise `use_rqeval: true` would silently bypass the rule.
The experiment preflight verifies this dependency and parser before any model
client is built, so a missing runtime cannot silently change the scoring rule.

**Frozen model gate.** The run must have 90 unique primary records, exactly 30
per band, zero failed calls, complete external difficulty metadata, correct
published-tercile membership and no item outside `calibration`. Binary CQ is
regressed on continuous official `rating`. A fixed 5,000-permutation one-sided
test with seed 95047 must have a negative slope with `p<=0.05`. Mean Easy CQ
minus mean Hard CQ must independently be at least 0.15. Both criteria are
required; there is no adjacent-band threshold and detector alarms are
descriptive only.

A pass makes GPT-4o-mini eligible only for a separately preregistered E2H-AMC
holdout K/L design. A failure means no validated GPT operating range was found
in the declared E2H-GSM8K/E2H-AMC route; bands, split, alpha, gap threshold and
scoring rules must not be tuned after observing the result. Llama and Mistral
K/L planning remains deferred until this outcome fixes the cross-model
comparison design. The call-free assessor is
`scripts/assess_e2h_amc_calibration.py`.

**Dataset-only preflight (2026-09-19).** Calibration contains 392/406/402
Easy/Medium/Hard candidates and holdout contains 609/589/577. The configured
ST3/ST4/ST5 streams select 30 items each, 90 unique items total, all in
`calibration`. Dataset loading, schema checks, answer-format inspection and
stream construction made no model endpoint calls. Of the 90 selected gold
answers, 86 use symbolic scalar equivalence and four use the preregistered
conservative surface path.

**Observed execution-invalid result (2026-09-20).** All 90 transport calls
returned, and the initial online statistical calculation produced CQ
0.233/0.033/0.033, an Easy-Hard gap of 0.200, slope -0.307812 and one-sided
permutation p=0.005999. This would pass the slope and gap rules in isolation.
It does not establish eligibility: 75/90 completions reached the configured
512-token cap and 81/90 lacked a final `Answer:` marker. Truncation
was strongly band-dependent (19 Easy, 28 Medium and 28 Hard), and every capped
completion was scored incorrect. The apparent difficulty signal therefore
confounds problem difficulty with mechanical output truncation. Section 44 is
retained as an execution-invalid run, not a positive calibration result.

---

## Section 45. E2H-AMC Mechanical Truncation Retry

**Recorded after the Section 44 validity audit and before retry execution
(2026-09-20).** The retry uses the same pinned dataset, seed 47, exact 90
question IDs, calibration partition, difficulty bands, GPT-4o-mini endpoint,
temperature 0.7, K=1/L=0, answer scorer, slope test, permutation seed, alpha and
0.15 Easy-Hard gap. It does not select new questions or alter a statistical
threshold after seeing outcomes. The only generation change is
`max_tokens: 512 -> 2048`, correcting the observed mechanical ceiling. Output
is isolated under `results/e2h_amc_calibration_gpt4o_mini_retry`; Section 44
files remain immutable evidence.

The client now records the provider's `finish_reason` in every raw checkpoint.
Eligibility requires both the unchanged statistical gate and execution
validity: zero records terminated for length/output-cap reasons. For legacy
records without `finish_reason`, reaching the configured token cap is treated
as truncation. Missing `Answer:` markers are reported diagnostically but do not
alone define truncation when the provider explicitly reports a normal stop.

The retry configuration is
`configs/e2h_amc_calibration_gpt4o_mini_retry_asu.yaml`. It contains 90
resumable primary calls. If any retry output is again truncated, the campaign
is execution-invalid and stops for review; no further output limit or prompt
change is automatically authorised.

---

## Section 46. Call-Free AMC Answer-Extraction Correction and Retry Decision

**Recorded after Section 45 execution and before substantive interpretation
(2026-09-20).** The retry produced 90/90 successful records, every provider
`finish_reason` was `stop`, and no response reached the 2,048-token ceiling.
Inspection of the immutable raw answers then found scorer false negatives. In
seven records, the gateway extractor returned the tail of a LaTeX
`\\text{Answer: ...}` wrapper, leaving a leading brace and trailing display-math
delimiter. One additional correct coordinate tuple used algebraically
equivalent component forms. A mixed-number normalization defect was also fixed
and regression-tested, although it did not change the retry score.

This is a scoring implementation correction, not a protocol change. Raw model
outputs, selected question IDs, difficulty bands, calibration partition,
temperature, token limit, slope test, permutation seed, alpha and Easy-Hard gap
remain unchanged. The assessor now recomputes CQ from each raw `answer` and
`gold_answer`; it writes both the original online and corrected values to
`data/e2h_amc_cq_rescore.csv`. Tuple components are compared symbolically in
order, while the preregistered conservative fallbacks remain in force. No model
endpoint was called during correction or reassessment.

For the execution-invalid Section 44 run, one CQ value changes. Its corrected
CQ is 0.267/0.033/0.033, gap 0.233, slope -0.358007 and p=0.003799; the run
remains ineligible because 75/90 responses were truncated. For the clean
Section 45 retry, eight CQ values change from incorrect to correct. Final CQ is
0.600/0.100/0.100, the Easy-Hard gap is 0.500, slope is -0.808150 and the
one-sided permutation p-value is 0.000200. Both frozen statistical criteria and
the zero-truncation execution criterion pass. GPT-4o-mini is therefore eligible
only for a separately preregistered E2H-AMC holdout K/L design; no holdout run
is authorised by this calibration decision alone. Evidence is under
`results/e2h_amc_calibration_gpt4o_mini_retry/`.

---

## Section 47. Three-Track Final-Holdout K/L Protocol

**Recorded before any final-holdout model call (2026-09-20).** Sections 41, 43
and 45 identify a validated model-specific operating range for each project
model, but not one common task range: Mistral-7B passed the official E2H-GSM8K
terciles, Llama-3-8B passed the E2H-GSM8K bottom/central/top 20% bands, and
GPT-4o-mini passed the official E2H-AMC terciles. Section 47 therefore freezes
three separate holdout tracks rather than pooling non-comparable raw accuracy:

- `configs/e2h_final_holdout_llama3_8b_asu.yaml` uses E2H-GSM8K extreme bands
  and the untouched `final_holdout` partition.
- `configs/e2h_final_holdout_gpt4o_mini_asu.yaml` uses E2H-AMC official
  terciles and the untouched broad `holdout` partition.
- `configs/e2h_final_holdout_mistral_7b_asu.yaml` uses E2H-GSM8K official
  terciles and the untouched `final_holdout` partition.

All tracks use seed 48, ST2 Easy-to-Medium-to-Hard order, 30 unique questions
per band, W=10, K=5, L=3, temperature 0.7, rqeval, real NLI, real BERTScore,
proposal detector mapping, disabled gateway history/search/date/time
augmentation, resumable raw checkpoints and no synthetic data. Each track is
90 x 8 = 720 calls; the full family is 2,160 calls. Llama and GPT are scheduled
first for operational reasons, but all three configs and gates are frozen now;
whether Mistral is run must not be selected from the Llama/GPT holdout results.
The superseded Section 42 config remains an immutable record and must not be
used.

**Mechanical completion rule.** Historical 512-token records include 1/90
capped Llama range-calibration calls, 1/90 capped Mistral calibration calls and
13/720 capped calls in the old Llama K/L pilot. Before viewing Section 47
outcomes, all three tracks therefore use a common 2,048-token ceiling and record
provider `finish_reason`. The post-run gate requires zero provider- or
token-cap-detected truncations. This is a mechanical validity control, not a
difficulty-band or statistical threshold change.

**Frozen post-run gate.** Each track must contain exactly 720 successful raw
records and 90 per-query rows, with five primary and three paraphrase records
per question, correct partition and band metadata, exact phase order, complete
metric ranges, at least 20 evaluable RS values, no degenerate metric and K-repeat
verbatim diversity of at least 80%. CQ is recomputed call-free from the first
primary raw completion and audited against online CQ. Binary CQ is regressed on
continuous official rating with 5,000 fixed-seed permutations (96048 Llama,
97048 GPT, 98048 Mistral). The one-sided negative-slope p-value must be at most
`0.05/3 = 0.0166667`, and mean Easy CQ minus mean Hard CQ must independently be
at least 0.15. Detector alarms cannot pass or fail this gate. A passing track
may proceed only to a separately preregistered call-free normalized detector
analysis; a failing track cannot be retuned against holdout outcomes.

**Call-free preflight (2026-09-20).** The official pinned Hugging Face datasets
loaded with the established content hashes (`7007d832a0a0dd00` for E2H-GSM8K,
`b7a9dbd7a037f560` for E2H-AMC). Llama pools contain 80/85/72 available
Easy/Medium/Hard items, GPT pools 609/589/577, and Mistral pools 127/130/136.
Every track selected 30/30/30 unique questions with zero overlap against prior
calibration items. Llama and Mistral naturally share 23 selected GSM8K question
IDs; their labels agree, but this does not authorise pooling raw CQ or absolute
alarm times. The selection, config hashes and query hash are stored under
`results/e2h_final_holdout_preflight/`. No model client was constructed and no
model endpoint was called.

**Observed Llama track (2026-09-20).** The Llama-3-8B arm completed all 720 raw
calls and 90 per-query rows with zero failures, zero detected truncations,
correct metadata and phase order, adequate RS coverage, finite in-range metrics
and no degenerate metric. Call-free CQ recomputation changed no online values.
The CQ holdout gate passed: Easy/Medium/Hard CQ was 0.900/0.867/0.633, the
Easy-Hard gap was 0.267, slope was -0.536946 and the fixed-seed one-sided
permutation p-value was 0.000200.

The independently frozen K-repeat diversity rule failed. Only 60/90 questions
had at least two distinct verbatim completions among five primary runs, for a
rate of 0.667 versus the 0.80 minimum. Fourteen Easy, eleven Medium and six Hard
questions had five identical primary completions, so the failure is not confined
to one difficulty phase. Section 47 therefore records the Llama arm as an
overall failure despite valid execution and successful CQ generalisation. The
diversity threshold, sampling temperature and repeat rule must not be changed
against this holdout result. Evidence is under
`results/e2h_final_holdout_llama3_8b/`. The frozen GPT-4o-mini arm remains
unexecuted and is next in the operational sequence.

**Observed GPT-4o-mini track (2026-09-20).** The E2H-AMC arm completed all 720
raw calls and 90 per-query rows with zero failures and zero detected
truncations. All 90 questions had at least two distinct verbatim K=5
completions, giving diversity 1.000; 32 questions had evaluable RS, all metric
coverage and range checks passed, metadata and phase order were correct, and no
metric was degenerate. Call-free CQ recomputation changed none of the online
values.

The frozen CQ holdout gate also passed. Easy/Medium/Hard CQ was
0.767/0.233/0.067, the Easy-Hard gap was 0.700, the continuous-rating slope was
-1.155497 and the 5,000-permutation one-sided p-value with seed 97048 was
0.000200, below the family-adjusted 0.0166667 threshold. The GPT-4o-mini track
therefore passes overall and its saved records are eligible only for a
separately preregistered call-free normalized detector analysis. That analysis
is deferred until the already frozen Mistral track is resolved. Evidence is
under `results/e2h_final_holdout_gpt4o_mini/`.

**Observed Mistral-7B track (2026-09-20).** The Mistral arm completed all 720
raw calls and 90 per-query rows with zero failures and zero detected
truncations. Metadata, phase order, metric coverage and ranges, RS coverage (44
questions) and nondegeneracy checks passed. Corrected Easy/Medium/Hard CQ was
0.767/0.433/0.267, the Easy-Hard gap was 0.500, slope was -0.949548 and the
fixed-seed one-sided permutation p-value was 0.000200. The CQ holdout gate
therefore passed.

The overall Mistral track nevertheless failed the frozen metric-health gate.
For every one of the 90 questions, all five primary completions were verbatim
identical; K-repeat diversity was 0.000 versus the 0.80 minimum. This behavior
occurred in all three difficulty phases and is consistent with the endpoint not
expressing seed-level sampling variation despite temperature 0.7. The project
must not change temperature, repeat definitions or the diversity threshold
against these holdout outcomes. Evidence is under
`results/e2h_final_holdout_mistral_7b/`.

---

## Section 48. Call-Free Thousands-Separator Parser Correction

**Recorded during the Mistral raw-record audit (2026-09-20).** The first
call-free rescore appeared to change one hard-item CQ from incorrect to correct.
Manual inspection showed that the model answered `4,000` while the gold answer
was `14,000`, so the change was a false positive. The generic number parser had
removed commas before applying a regex limited to one-to-three leading digits;
both four- and five-digit integers could consequently be reduced to their last
partial match, in this case `0`.

The parser was corrected to distinguish comma-grouped integers from
arbitrary-length plain integers and to remove commas only after selecting the
complete numeric token. Regression tests now require `14,000 == 14000` and
`4,000 != 14000`. This is an implementation bug fix applied to immutable raw
records, not a threshold or protocol change, and issued no model calls. All
three Section 47 assessors were rerun. Llama was unchanged, GPT uses the
separate AMC symbolic matcher, and Mistral now has zero CQ changes relative to
online scoring. Its corrected CQ is 0.767/0.433/0.267 rather than the transient
0.767/0.433/0.300; both versions pass the CQ gate, and both leave the overall
decision as failure because K-repeat diversity is zero.

**Section 47 family decision.** GPT-4o-mini is the only model passing execution,
CQ generalisation and the complete K/L metric-health gate. Llama-3-8B and
Mistral-7B remain valid negative results for repeat diversity. Only the GPT
saved records may proceed to a separately preregistered call-free normalized
detector analysis. No additional paid holdout calls, endpoint-option changes or
post-hoc threshold changes are authorised by these results.

---

## Section 49. GPT-Only Call-Free Normalized Detector Timing

**Recorded after the Section 47 family decision and before inspecting saved
alarm values (2026-09-20).** GPT-4o-mini is the only track eligible for detector
analysis. Section 49 uses the primary `level=drift` alarms already produced by
the frozen `proposal_mapping` during the paid run. It does not refit, replay or
replace any detector and issues no model calls. The exact contract is stored in
`configs/e2h_gpt_detector_analysis.yaml`.

The Easy-to-Medium and Medium-to-Hard transitions are evaluated separately at
change points 30 and 60. For each metric, delay is the first primary alarm after
the boundary and within the following 30-question phase, minus the change point;
normalized delay divides this value by 30. CQ is the reference. CS, LS, ES and
SS are preregistered early-warning candidates. RS is descriptive only because
its Section 47 per-query coverage is sparse.

The early-warning gate passes only if at least one candidate metric has no
primary alarm in the initial Easy phase, alarms in both post-change intervals,
CQ also alarms in both intervals, and the candidate precedes CQ by at least one
query at both transitions. CQ itself must also have a clean Easy baseline. All
conditions are conjunctive; a favorable single transition cannot rescue the
other, and thresholds or detector assignments must not be changed after the
alarm file is read. Phase means and losses relative to the Easy mean are
reported descriptively. A pass is single-model method evidence, not cross-model
confirmation of the proposal hypothesis.

**Observed result (2026-09-20).** The call-free analysis read the immutable
Section 47 GPT alarm trace only after the contract above had been frozen. CQ had
no Easy-phase false alarm and detected both transitions: t=37 for
Easy-to-Medium (delay 7/30 = 0.233) and t=79 for Medium-to-Hard (delay 19/30 =
0.633). Every candidate also had a clean Easy baseline. LS was the only
candidate to precede CQ anywhere, alarming at t=36 in the first transition for
a one-query lead, but it produced no primary drift alarm in the second
transition. ES alarmed at t=39 in the first transition, two queries after CQ,
and did not alarm in the second. CS and SS did not alarm in either post-change
interval. RS did not alarm and remains descriptive because only 23/7/2 values
were evaluable across Easy/Medium/Hard.

No candidate therefore satisfies the conjunctive two-transition early-warning
gate. This is a valid negative result for the frozen detector mapping on the
single eligible GPT-4o-mini track: it does not support a claim that the
non-CQ metrics reliably anticipate CQ degradation. Detector thresholds and
assignments remain unchanged, and no model calls were issued. Evidence is in
`results/e2h_final_holdout_gpt4o_mini/SECTION49_GPT_DETECTOR_ANALYSIS.md` and
the accompanying `data/section49_*.csv` files.

---

## Section 50. Call-Free Answer and Reasoning-Trace Parsing Audit

**Recorded after the Section 49 negative result (2026-09-20).** The possibility
that model outputs were being collapsed by parsing was tested directly on all
immutable Section 47 raw records. The audit separates final-answer extraction,
which affects CQ/CS/RS, from reasoning-trace segmentation, which affects LS/SS.
It also confirms that the frozen K-repeat diversity gate uses verbatim raw text
and is independent of both parsers.

One AMC raw-rescore discrepancy exposed a nested-label bug: `Final answer:
Answer: 36` was correctly reduced to `36` by the live client but only the outer
label was removed when the raw completion was rescored. The raw matcher now
peels repeated answer labels. This call-free correction changes no primary CQ,
no per-query CS and no Section 47 or Section 49 decision. After correction,
Llama and GPT have zero stored-versus-raw CQ changes and zero CS changes.
Mistral has five representation-path mismatches from one question, but both
paths remain incorrect and its CQ/CS values are unchanged.

The reasoning-trace audit found a separate material limitation. The live client
splits completions on every period and retains only the first 12 fragments.
All 450 GPT K-run traces hit this cap; their median retained character fraction
is 0.263 and none retains the final answer marker. Llama has 147/450 capped
traces, with the cap rate increasing from 0.147 on Easy to 0.247 on Medium and
0.587 on Hard. Mistral has 240/450 capped traces, although its verbatim K runs
were already identical before parsing.

Therefore final-answer parsing does not explain the observed CQ separation,
GPT CS decline, or Llama/Mistral diversity failures. However, LS and SS were
computed from incomplete reasoning prefixes and must not be treated as fully
validated evidence until a separately specified, structure-aware call-free
reparse sensitivity analysis is completed. The frozen historical results are
retained rather than silently overwritten. Evidence is in
`results/answer_parsing_audit/ANSWER_PARSING_AUDIT.md`.

---

## Section 51. Preregistered GPT Trace-Reparse Sensitivity

**Recorded after the Section 50 truncation diagnosis and before computing any
corrected LS/SS value (2026-09-20).** This sensitivity analysis uses only the
immutable Section 47 GPT-4o-mini raw records. It issues no model calls, does not
alter CQ/CS/RS/ES, and preserves the historical online results. The exact
contract is stored in `configs/e2h_gpt_trace_reparse_sensitivity.yaml`.

The replacement parser `structured_paragraphs_v1` uses blank-line-delimited
paragraphs as reasoning steps, merges standalone display-math blocks into the
preceding explanation, and uses punctuation-plus-whitespace sentence boundaries
only when a provider returned one unstructured paragraph. Decimal points are
not boundaries. The final answer segment is retained and there is no maximum
segment count. These rules are fixed before the corrected metric values are
viewed.

LS and SS are recomputed with the same rqeval implementations and cached NLI
and BERTScore models used by Section 47. The analysis retains W=10, the
unchanged `proposal_mapping` assignments (LS Page-Hinkley, SS ADWIN), detector
defaults, stream order and transition boundaries 30 and 60. Detectors are
replayed from a fresh state but are neither refit nor retuned. Old and corrected
per-query values, phase means, rank/linear association and primary alarms are
reported side by side.

The Section 49 timing rule is applied descriptively to corrected LS/SS: a
candidate must have a clean Easy baseline, alarm in both following 30-query
intervals and precede the unchanged CQ alarms in both. A changed outcome is a
post-diagnostic sensitivity result, not a new confirmatory test. It determines
whether the Section 49 scientific interpretation must be revised before any
new dataset experiment.

**Observed result (2026-09-20).** All 90 GPT queries were rescored with real
NLI and BERTScore; no model endpoint was called. Corrected LS changed
substantially: old-versus-corrected Pearson/Spearman associations were
0.286/0.272, mean absolute change was 0.186 and maximum absolute change was
0.565. Easy/Medium/Hard LS changed from 0.594/0.585/0.694 to
0.672/0.734/0.729. The old LS alarm at t=36 disappeared and corrected LS
produced no primary drift alarm, so the apparent one-query LS lead in Section
49 was a prefix-parsing artifact rather than supporting evidence.

SS was much less sensitive to reparsing. Pearson/Spearman associations were
0.861/0.868, mean absolute change was 0.010 and its phase means changed from
0.920/0.915/0.911 to 0.924/0.917/0.906. Neither old nor corrected SS alarmed.
Corrected LS and SS therefore both fail the original two-transition timing
rule. The parser correction materially improves trace validity but does not
rescue the early-warning hypothesis. Section 49 remains a negative result, now
with the old LS t=36 alarm explicitly withdrawn as an artifact. Evidence is in
`results/e2h_final_holdout_gpt4o_mini/SECTION51_TRACE_REPARSE_SENSITIVITY.md`.

---

## Section 52. SuperGPQA Multiple-Choice Exploratory Track

**Recorded before any SuperGPQA target-model response was generated
(2026-09-20).** This track tests the alternative explanation that free-form
answer extraction or insufficiently specialized questions obscured separation.
It does not replace or erase the frozen E2H-GSM8K design and does not revise any
Section 47-51 result. The source is pinned to `m-a-p/SuperGPQA` revision
`4430d4458112c7d4497fdcf94d7cc223313d6acf`, SHA-256
`28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910`.

The call-free structural audit examined all 26,529 rows. It excluded 75 rows
containing duplicate option texts and all four rows belonging to two duplicate
normalized-question groups. To hold the guessing baseline fixed, only clean
ten-option questions are eligible. The selected subfield is determined without
model outputs: maximize the minimum official Easy/Middle/Hard count, then total
count, then lexical hierarchy. This rule selects Science > Chemistry >
Analytical Chemistry with 45/107/163 eligible questions.

Seed 52 freezes two disjoint UUID-only pools, each containing 20 Easy, 20
Medium and 20 Hard questions. Pilot gold positions are exactly balanced with
two instances of every letter A-J per tier. Prompts require a final line of
`FINAL_CHOICE: X`, and the client/parser now prioritizes that anchored marker.
This directly audits CQ/CS answer extraction while retaining complete raw text
and the Section 51 reasoning parser for LS/SS.

The design has an explicit interpretation boundary. SuperGPQA difficulty was
formed partly from expert judgment and LLM response accuracy, and the authors'
filtering used GPT-4o among other models. It is not a wholly model-independent
difficulty treatment. Within the selected clean subfield, calculation rates
are 0.133/0.636/0.975 for Easy/Middle/Hard, so any detected change is a
transparent difficulty-and-calculation composition shift rather than an
isolated causal difficulty effect. The composite ODC-BY dataset also requires
upstream attribution/license compliance.

The call-free integration preflight passes with 60 unique questions and change
points 20 and 40. The GPT-4o-mini screening config uses K=1 and L=0, so its
ceiling is 60 calls and 30,720 output tokens. It remains marked
`HOLD FOR EXPLICIT EXECUTION`; neither this record nor the successful preflight
authorizes a paid run. Evidence and frozen UUIDs are under
`results/supergpqa_pool_audit/`.

Before viewing any pilot output, the screening gate was also frozen. It requires
60 unique successful, non-truncated calls; 100% valid extracted A-J choices;
at least 95% anchored `FINAL_CHOICE` compliance; non-increasing Easy/Medium/Hard
CQ; an Easy-Hard gap of at least 0.20; and each tier accuracy within 0.10-0.90
inclusive to avoid floor/ceiling degeneracy. No significance threshold is used
at 20 questions per tier. A pass permits consideration of the already frozen
validation pool but is not confirmatory evidence.

**Observed Section 52 result (2026-09-20).** All 60 calls returned HTTP-level
success, but only 44 produced an A-J choice and anchored `FINAL_CHOICE` marker.
Every one of the 16 invalid outputs reported exactly 512 completion tokens and
ended mid-reasoning, even though the gateway labelled `finish_reason` as
`stop`. The original assessor's stop-reason-only truncation check therefore
missed provider-side token-ceiling censoring. The check now also treats usage at
the configured token ceiling as truncation; this is a diagnostic correction,
not a changed gate. Censoring was differential: 0/20 Easy, 4/20 Medium and
12/20 Hard.

Conservative CQ (censored outputs counted wrong) was 0.45/0.15/0.05, but its
missing-outcome bounds were [0.45,0.45], [0.15,0.35] and [0.05,0.65]. The Hard
interval overlaps Easy, so the apparent 0.40 Easy-Hard gap cannot be attributed
cleanly to difficulty. Section 52 fails its frozen execution, parsing and
nondegeneracy gates and does not authorise the validation pool.

---

## Section 53. Homogeneous Answer-First SuperGPQA Retry

**Recorded after diagnosing Section 52 and before generating any retry output
(2026-09-20).** Selectively retrying only the 16 censored questions would couple
a new prompt/cap to Medium and Hard items. Section 53 instead reruns all 60
frozen pilot UUIDs as a separate, homogeneous experiment; Section 52 remains
immutable and the two runs are never pooled.

The question content, options, gold labels, order, model, temperature, seed,
gateway controls, K=1/L=0 and W=5 are unchanged. The output contract changes in
two preregistered ways: `FINAL_CHOICE: X` must be the first output line, followed
by at most four short reasoning sentences, and the output ceiling increases
from 512 to 1024 tokens. This makes CQ observable before any possible reasoning
truncation while preserving a short trace for descriptive LS/ES. The complete
60-call rerun has a 61,440-token ceiling.

The Section 52 gate is retained, with token-ceiling detection and one added
format requirement: at least 95% of responses must put the anchored choice on
the first line. The retry remains `HOLD FOR EXPLICIT EXECUTION`; no call is
authorised by this preregistration.

**Observed Section 53 result (2026-09-20).** All execution and parsing checks
passed: 60/60 successful calls, valid choices and first-line markers, with zero
token-ceiling hits. Easy/Medium/Hard CQ was 0.35/0.25/0.10 and the Easy-Hard gap
was 0.25. Every frozen screening condition passed. The one-sided ordered trend
test was z=-1.869, p=0.0308; the descriptive Easy-versus-Hard one-sided Fisher
test was p=0.0637. The pilot therefore supports advancing to the already frozen
validation pool but is not itself confirmatory evidence.

---

## Section 54. Disjoint SuperGPQA GPT Validation

**Recorded after Section 53 passed and before any validation output was
generated (2026-09-20).** Section 54 uses the 60 validation UUIDs frozen in
Section 52, with no overlap with the pilot. It retains the Section 53 model,
answer-first prompt, 1024-token ceiling, temperature, K=1/L=0, twenty questions
per official tier and all gateway controls. Seed 53 changes only stream order;
the provider does not receive a seed parameter.

The validation gate requires all execution/parser checks, CQ within 0.10-0.90
for each tier, non-increasing Easy/Medium/Hard CQ, an Easy-Hard gap of at least
0.20, and a one-sided Cochran-Armitage ordered-trend p-value no greater than
0.05. Easy-versus-Hard Fisher and Wilson intervals are descriptive. Conditions
are conjunctive and will not be relaxed after viewing the validation output.
This is still a single-model, single-subfield validation and cannot establish
cross-model generality. The 60-call run remains on hold for explicit execution
approval.

**Observed Section 54 result (2026-09-20).** All 60 calls succeeded, all choices
and first-line markers were parseable, and no response reached the token
ceiling. Easy/Medium/Hard CQ was 0.25/0.30/0.05. The Easy-Hard gap equalled the
0.20 minimum, but three conjunctive conditions failed: Medium exceeded Easy by
one correct response, Hard was below the 0.10 nondegeneracy floor, and the
one-sided ordered-trend result was z=-1.581, p=0.0569 rather than p<=0.05. The
descriptive Easy-versus-Hard Fisher p-value was 0.0909. Section 54 therefore
fails its frozen validation gate. The thresholds are not relaxed and this
result does not authorise automatic expansion to Llama.

As a post-validation descriptive check only, pooling the disjoint Section 53
and Section 54 sets gives CQ=0.300/0.275/0.075 (12/40, 11/40 and 3/40), an
Easy-Hard gap of 0.225, ordered-trend p=0.0073 and Easy-versus-Hard Fisher
p=0.0099. This pooled result shows that the directional contrast is not absent,
but it is exploratory because the validation decision had already been made;
it cannot rescue or relabel the failed Section 54 confirmatory gate.

---

## Section 55. Accepted-Proposal S1 Answer Observability Audit

**Recorded after Sections 37-39 and without model calls (2026-09-20).** The
immutable S1 K=5/L=3 records were re-read to test whether final-answer parsing
or output censoring contributed to the negative early-warning result. This
audit does not change or pool any historical observation. It distinguishes an
explicit final answer from a completion that reached the configured 512-token
ceiling before producing one.

The dominant defect is differential MATH censoring. GPT-4o-mini reached exactly
512 tokens on 94/150 MATH K-runs and all 94 lacked a defensible final answer;
Llama-3-8B reached the ceiling on 72/150 and 71 lacked one. In the primary K
call, only 12/30 GPT and 14/30 Llama MATH answers were strictly observable.
Their complete-case CQ values were 0.667 and 0.500, while missing-outcome CQ
bounds were [0.267,0.867] and [0.233,0.767]. The original online values were
0.333 and 0.167. These intervals show that the magnitude and timing of the MATH
CQ decline cannot be cleanly inferred from the original answer-last run.

There is also a genuine ARC extraction defect. Llama often returned a boxed
choice inside full prose without the client's exact `Answer:` delimiter. The
legacy ARC normalizer could then select an earlier standalone A-D token.
Strict final-choice extraction changed CS on 9/29 complete Llama ARC queries;
its primary ARC complete-case CQ was 0.862 with bounds [0.833,0.867], compared
with online CQ 0.767. GPT ARC did not show this correction pattern.

**Decision.** The Section 38-39 failure remains an immutable historical result,
but it is not a clean test of H3 because the answer stream, and therefore CQ,
CS, RS and ES, was not fully observable. Detector retuning on these records is
not authorised. The minimal corrective design is a homogeneous rerun of every
S1 cell with a typed first-line answer, bounded short reasoning and a larger
output ceiling. Converting all tasks to multiple choice is not necessary and
would alter the accepted-proposal datasets; answer-first formatting provides
the required observability while preserving GSM8K, ARC-Challenge and MATH.
Evidence is in `results/proposal_s1_parsing_audit/`.

---

## Section 56. Parser-Controlled Accepted-Proposal S1 Pilot

**Recorded after the Section 55 diagnosis and before generating any Section 56
target-model response (2026-09-20).** This pilot tests whether the accepted-
proposal S1 trajectory becomes identifiable when final answers are observed
before reasoning can be truncated. It does not erase, modify or pool the
Section 37-39 records, and it does not change GSM8K, ARC-Challenge or MATH into
a multiple-choice surrogate.

The frozen stream contains the same 90 question IDs in the same order as the
historical S1 run: 30 GSM8K, 30 ARC-Challenge and 30 MATH questions under seed
42 and W=10. The call-free preflight loaded real cached benchmark data and
confirmed sequence SHA-256
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`.
GPT-4o-mini and Llama-3-8B retain temperature 0.7 and the same disabled gateway
history/search/prompt-context controls. K=1 and L=0 isolate answer
observability; CS, RS and SS are not interpreted in this pilot. Rqeval is
disabled for this operational gate because the frozen assessor recomputes CQ
directly from raw first-line answers; loading its unused local Transformer
scorers would change neither the gate nor its evidence. The complete design has
180 calls and a maximum-output ceiling of 184,320 tokens.

ARC responses must begin with `FINAL_CHOICE: X`, where X is A-D. GSM8K and
MATH responses must begin with `FINAL_ANSWER: <answer>` on one line. At most
eight short numbered reasoning steps may follow, and the first-line answer may
not be revised. The completion ceiling is 1024 tokens. The client prioritises
these anchored markers, and the first-line answer declaration is excluded from
the reasoning-step trace so a choice letter or answer value cannot inflate
future LS/SS comparisons.

The model-specific gate is conjunctive and was frozen prospectively. Each
model must produce 90 unique successful calls, 30 per dataset, in the frozen
order; 100% must contain a valid typed answer on the first line; stored-client
answers must agree with the strict first-line parser on 100%; and no output may
reach the token ceiling or report length truncation. For scientific dynamic
range, each model must also have GSM8K CQ minus MATH CQ at least 0.15 and a
one-sided ordered GSM8K-to-ARC-to-MATH trend p-value no greater than 0.05.
Wilson intervals are reported. Failure of either model fails the overall gate;
thresholds will not be relaxed and selective reruns are not permitted.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass establishes
parser observability and sufficient CQ range for planning a separate K=5/L=3
metric-health run. It is not itself evidence that a non-CQ metric provides
early warning, and it does not automatically authorise that larger paid run.

**Observed Section 56 result (2026-09-20).** Execution and typed-answer
observability succeeded: both models returned 90/90 successful calls, every
response began with the required marker, stored answers agreed with the strict
parser on every item, and the stream hash matched the frozen S1 sequence. GPT
had no token-ceiling output. Llama reached 1024 tokens on seven traces (two
GSM8K and five MATH), so its trace-completeness condition failed even though
its first-line CQ answer remained observed.

The scientific range conditions failed decisively. GPT CQ was
0.200/0.867/0.367 and Llama CQ was 0.200/0.733/0.267 for
GSM8K/ARC-Challenge/MATH. Their GSM8K-MATH gaps were -0.167 and -0.067, and
one-sided ordered-trend p-values were 0.9019 and 0.7009. These values do not
support the preregistered decline and do not authorise a K=5/L=3 run.

A call-free prompt-order diagnosis shows that this failure is not residual
answer parsing. On the identical GSM8K questions, historical answer-last CQ
was 28/30 for GPT and 24/30 for Llama, compared with 6/30 for each answer-first
run. In the new GPT traces, the last numeric result in the subsequent reasoning
was correct on 20/30 questions: fourteen initially wrong first-line answers
were followed by the correct number, while no initially correct answer became
wrong. For Llama, first and later numeric values differed on 21/30 questions;
six initially wrong answers were followed by the gold value and four initially
correct answers were followed by a wrong value. Examples include GPT emitting
120 before later deriving 84, and Llama emitting 40 before later deriving 30.

The answer-first contract therefore changes model solving behaviour and turns
the visible reasoning into post-answer justification rather than a trace that
leads to the measured answer. Section 56 is retained as a failed operational
design, not as evidence against the S1 hypothesis. Any next calibration must
restore bounded reasoning before a typed final-answer line and first verify
that this ordering preserves answer quality and avoids censoring. No further
paid run is authorised by this diagnosis.

---

## Section 57. Bounded Answer-Last Prompt-Order Calibration

**Recorded after the Section 56 prompt-order diagnosis and before generating
any Section 57 target-model response (2026-09-21).** Section 57 tests one narrow
question: does restoring reasoning before a typed final answer preserve the
high GSM8K answer quality seen in the historical answer-last run while keeping
the answer stream fully observable? It is an operational calibration, not an
S1 early-warning experiment, and its records will not be pooled with Sections
37 or 56.

The frozen stream contains the first 10 GSM8K and first 10 MATH question IDs
from the immutable Section 56 sequence, in that order. ARC-Challenge is omitted
because the diagnosed intervention effect concerns open-ended numeric/symbolic
answers. The normalized dataset/QID sequence SHA-256 is
`99f3b47eb250b15defee6086a76adb2c777e824ff3308f35aa4ebc463d024e20`.
The call-free integration preflight loaded real cached benchmark records,
confirmed 20 unique questions and reproduced this hash exactly.

GPT-4o-mini and Llama-3-8B retain temperature 0.7 and the same disabled gateway
history, search and prompt-context controls. K=1 and L=0 isolate prompt order;
rqeval is disabled because this calibration assesses CQ and output integrity
directly from saved raw responses. Each model receives the same 20 questions.
The complete ceiling is 40 calls and 81,920 maximum output tokens.

Each response must first solve the problem in at most six short numbered
reasoning steps, then end with exactly one line of the form
`FINAL_ANSWER: <answer>`, with nothing after it. The completion ceiling is 2048
tokens. The generic client extracts the last anchored typed answer and excludes
typed answer lines from future reasoning traces. The frozen assessor is stricter:
it requires exactly one typed answer line and requires it to be the final line.

The model-specific gate is conjunctive. Each model must produce 20 unique
successful primary calls, 10 per dataset, in the frozen order; final-line
answer observability and strict-parser/stored-answer agreement must both be
100%; and no output may report length truncation or reach the 2048-token
ceiling. To reject another severe prompt-induced GSM8K collapse, GSM8K CQ must
be at least 0.60 for each model. MATH CQ and Wilson intervals are reported only
descriptively because ten questions do not support a stable difficulty-trend
test. Failure of either model fails the overall gate, and no threshold,
question, prompt or ceiling may be changed after output is observed.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass permits
planning a new homogeneous accepted-proposal S1 run with bounded answer-last
formatting; it does not authorise that larger run and is not evidence that a
non-CQ metric leads CQ.

**Observed Section 57 result (2026-09-21).** Both model gates passed. All 40
primary calls succeeded, the frozen stream hash matched, every response had
exactly one final-line typed answer, stored answers agreed with the strict
parser, and neither model produced a length or token-ceiling truncation.
GPT-4o-mini answered 10/10 GSM8K and 7/10 MATH questions correctly; Llama-3-8B
answered 8/10 and 4/10. Thus both exceeded the prospectively frozen GSM8K CQ
floor of 0.60. The corresponding Wilson intervals remain wide because each
dataset has only ten observations; MATH differences are descriptive.

A call-free, post-gate paired comparison used the same selected QIDs in the
historical answer-last and Section 56 answer-first records. On GSM8K, GPT
correctness was 10/10 historically, fell to 2/10 under answer-first, and
returned to 10/10 under bounded answer-last. Llama was 8/10, 1/10 and 8/10.
Relative to Section 56, eight GPT and seven Llama GSM8K items changed from
wrong to correct, while none changed from correct to wrong. On MATH, the three
corresponding counts were 5/10, 3/10 and 7/10 for GPT, and 3/10, 3/10 and 4/10
for Llama.

This small non-randomized comparison supports, but does not by itself prove,
the prompt-order diagnosis because the bounded step instruction and token
ceiling also differ across runs. It does show that strict answer observability
can coexist with the historical GSM8K quality level when reasoning precedes
the typed answer. Section 57 therefore authorises preregistration of a full
90-question K=1/L=0 S1 validation using the bounded answer-last contract. It
does not authorise a K=5/L=3 campaign or establish early warning.

---

## Section 58. Full Bounded Answer-Last S1 Validation

**Recorded after Section 57 passed and before generating any Section 58
target-model response (2026-09-21).** Section 58 tests whether the bounded
answer-last contract remains operationally valid and yields the required CQ
dynamic range on the complete accepted-proposal S1 stream. It is a homogeneous
new run; Sections 37, 56 and 57 remain immutable and are not pooled into its
confirmatory gate.

The stream is the exact 90-question sequence used in the historical S1 run:
30 GSM8K, 30 ARC-Challenge and 30 MATH items, seed 42 and W=10. Its normalized
dataset/QID sequence SHA-256 is
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`.
The call-free preflight loaded all three real cached benchmark sources,
confirmed 90 unique questions, reproduced the hash and matched every QID and
position to the historical stream.

GPT-4o-mini and Llama-3-8B retain temperature 0.7 and disabled gateway history,
search and prompt-context controls. K=1 and L=0 isolate CQ and format health;
rqeval is disabled because no K/L metric is interpreted. Each completion may
use at most six short numbered reasoning steps and has a 2048-token ceiling.
GSM8K and MATH must end with exactly one `FINAL_ANSWER: <answer>` line.
ARC-Challenge must end with exactly one `FINAL_CHOICE: X` line, where X is A-D.
Nothing may follow the typed line. The complete ceiling is 180 calls and
368,640 maximum output tokens; checkpoint/resume remains enabled.

The model-specific gate is conjunctive and frozen prospectively. Each model
must produce 90 unique successful primary calls, 30 per dataset, in the frozen
order. Strict final-line observability and strict-parser/stored-answer agreement
must both be 100%, and no completion may report length truncation or reach the
token ceiling. Scientific dynamic range requires GSM8K CQ minus MATH CQ at
least 0.15 and a one-sided ordered GSM8K-to-ARC-to-MATH trend p-value no greater
than 0.05. Wilson intervals are reported. Either model failing any condition
fails the overall gate; thresholds, prompts, questions and token ceiling will
not be changed after output is observed.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass permits
preregistration of a separate K=5/L=3 metric-health campaign under the same
answer-last contract. It does not automatically authorise that campaign and is
not itself evidence that a non-CQ metric leads CQ.

**Observed Section 58 result (2026-09-21).** The overall frozen gate failed.
All 180 calls returned, both models had 90 unique records in the exact frozen
stream order, and no HTTP-level execution failed. Both scientific range checks
passed strongly. GPT-4o-mini CQ was 0.967/0.900/0.567 for
GSM8K/ARC-Challenge/MATH, yielding a 0.400 GSM8K-MATH gap and one-sided ordered
trend p=0.0000378. Following the call-free delimiter correction documented
below, Llama-3-8B CQ is 0.833/0.867/0.300, with gap 0.533 and
p=0.00000589.

Strict answer observability nevertheless failed prospectively frozen 100%
requirements. GPT had one MATH format violation (`math-precalculus-00035`):
the response stopped normally at 507 tokens and derived the correct
`\frac{5}{13}`, but placed `FINAL_ANSWER` inside a LaTeX `\text{...}` display
rather than on a standalone final line. It is semantically recoverable but
remains a strict gate failure. Llama had three genuine MATH censoring events
(`math-precalculus-00037`, `math-intermediate_algebra-00000`, and
`math-intermediate_algebra-00026`). Each entered a repetitive reasoning loop,
reached exactly 2048 tokens despite the gateway reporting `stop`, and emitted
no typed answer. Thus GPT final-line observability/parser agreement was 89/90;
Llama was 87/90 with three token-ceiling truncations.

A call-free post-gate sensitivity analysis kept the formal decision unchanged.
Recovering GPT's visibly correct answer raises MATH CQ to 0.600 while retaining
gap 0.367 and trend p=0.000102. For Llama, the strict result is the lower CQ
bound; even treating all three unresolved MATH answers as correct gives MATH
CQ 0.400, gap 0.433 and trend p=0.000125. The ordered CQ decline is therefore
robust to every unresolved outcome, but operational trace completeness is not.

Section 58 remains a failed validation under its conjunctive gate. Its
thresholds will not be relaxed, its failed items will not be selectively
rerun, and it does not authorise a K=5/L=3 campaign. Any subsequent development
must be separately preregistered and address both plain-text marker placement
and Llama's high-difficulty runaway traces before another homogeneous
validation.

---

## Section 59. Enriched Answer-Format Stress Pilot

**Recorded after the Section 58 failure diagnosis and before generating any
Section 59 target-model response (2026-09-21).** This is a development stress
test, not a selective confirmatory rerun. It cannot modify, pool with or pass
Section 58. Its purpose is only to test whether a stricter output contract
prevents the two observed operational failures: a LaTeX-embedded marker and
runaway reasoning that never reaches the final marker.

The frozen MATH selection contains all four unique Section 58 failure QIDs:
`math-precalculus-00037`, `math-intermediate_algebra-00000`,
`math-precalculus-00035`, and `math-intermediate_algebra-00026`. Six controls
are selected mechanically as the first six remaining MATH QIDs in historical
S1 order. The ten items retain their historical relative order. Selected
sequence SHA-256 is
`1a71ba588cd589bb0a07a1801d9a16c518c31e6066d44a0371e439e789213a37`.
The call-free preflight reconstructed the failures from both Section 58 raw
record files, verified the control rule, loaded the real cached MATH source and
reproduced this hash.

GPT-4o-mini and Llama-3-8B retain temperature 0.7, K=1/L=0, disabled gateway
history/search/prompt context and a 2048-token ceiling. The revised prompt
allows no more than six numbered reasoning lines and exactly one short sentence
per line. After line six or sooner the model must immediately provide its best
answer even if uncertain. The final `FINAL_ANSWER: <answer>` line must be plain
text, outside LaTeX, code blocks, brackets and list items, with nothing after
it. The complete ceiling is 20 calls and 40,960 output tokens; resume is on.

The gate is operational and conjunctive. Each model must return ten unique
successful calls in the frozen order, with 100% strict final-line observability,
100% strict-parser/stored-answer agreement, zero token-ceiling or length
truncations and zero violations of the six-numbered-line budget. CQ is reported
by failure/control subgroup but is not a gate because the sample was enriched
using observed Section 58 outcomes. Either model failing any operational check
fails the pilot. Thresholds, questions and prompt will not be altered after
output is observed.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass permits only
planning a new homogeneous 90-question validation with this prompt. It does
not alter Section 58 and does not authorise K=5/L=3.

---

## Section 60. Llama-4 Scout Substitution Feasibility Arm

**Recorded after Section 59 and after an accepted minimal gateway probe, but
before generating any Section 60 stress response (2026-09-21).** Llama-3-8B
failed the same high-difficulty formatting items under both Sections 58 and 59.
Section 60 therefore evaluates Llama-4 Scout as a post-hoc candidate model.
This is explicitly a model-substitution feasibility arm: it does not erase,
replace, pool with or relabel any Llama-3 result.

The ASU model catalogue contains `aws/llama4_scout-17b`. A saved one-call probe
using temperature 0.7 and the frozen `disable_all` gateway controls returned
`accepted: true`. Section 60 changes only the provider model. It reuses the
exact Section 59 set of four recurrent/problematic MATH QIDs and six mechanical
controls in the same order, selected-sequence SHA-256
`1a71ba588cd589bb0a07a1801d9a16c518c31e6066d44a0371e439e789213a37`.

The model receives the identical strict stress prompt: no more than six
numbered reasoning lines, one short sentence per line, an immediate best
answer by line six, and exactly one plain-text final `FINAL_ANSWER` line outside
LaTeX and code blocks. Temperature remains 0.7, K=1/L=0, the output ceiling is
2048 tokens and resume is enabled. The complete ceiling is 10 calls and 20,480
output tokens. The call-free preflight verified the real MATH source, source
failure/control derivation, QID order, provider model and accepted probe.

The operational gate is unchanged from Section 59: all ten calls must be
successful and unique, strict final-line observability and parser agreement
must be 100%, no output may reach the token ceiling, and no output may exceed
the six-numbered-line budget. CQ is descriptive because the selection is
outcome-enriched. Thresholds and prompt will not be changed after output is
observed.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass permits only
planning a homogeneous full 90-question Llama-4 Scout K=1/L=0 validation. It
does not authorise K=5/L=3 and does not convert Llama-3 failures into passes.

**Observed Section 59 result (2026-09-21).** The two-model conjunctive gate
failed. GPT-4o-mini passed every operational condition: all ten calls were
successful and unique, strict final-line observability and parser agreement
were 10/10, no output reached the ceiling and no output violated the six-line
budget. Descriptive CQ was 1/4 on the outcome-enriched Section 58 failure set
and 4/6 on controls; these values are not inferential and were not gated.

Llama-3-8B failed with strict observability/parser agreement of 8/10, one
token-ceiling truncation and two six-line-contract violations. Both failures
were recurrent Section 58 items. On `math-intermediate_algebra-00000`, Llama
ignored the hard line budget, reached step 13, entered a repetitive expression
loop and exhausted 2048 tokens without a marker. On
`math-intermediate_algebra-00026`, it stopped normally at 257 tokens but wrote
an incorrect `FINAL_ANSWER` as numbered line 7, violating both the line budget
and plain standalone marker contract. The other two historical failure QIDs
became observable. All six controls were observable. After the call-free
delimiter correction documented below, descriptive Llama CQ is 0/4 on prior
failures and 3/6 on controls.

The stress intervention is therefore sufficient for GPT formatting but not
for Llama's high-difficulty instruction following. Section 59 remains a failed
overall pilot and cannot change Section 58. Repeated Llama prompt tuning is
stopped to avoid an outcome-adaptive sequence of retries. Llama is not eligible
for K=5/L=3 under this protocol. Any GPT-only metric-health campaign must be
separately preregistered, retain the Section 58 full-stream prompt that already
preserved CQ, and treat GPT's single LaTeX-wrapped answer as a parser robustness
case rather than changing answer order again.

**Post-run MATH delimiter correction (2026-09-21; no model calls).** Inspection
of the immutable Section 60 records found two answers whose mathematical
content exactly matched gold but whose complete expressions were wrapped in
`$...$`. The numeric fallback could not full-match the wrapped LaTeX fraction
and incorrectly selected its denominator as the last number. The correction is
narrow: for MATH only, one complete outer pair among `$...$`, `$$...$$`,
`\(...\)` or `\[...\]` is removed before the existing normalizer runs. It does
not alter raw responses, answer extraction, QID selection, hashes, prompts or
gate thresholds. Regression tests freeze all four accepted delimiter forms.

A call-free audit of Sections 58-60 found four previously false-negative
comparisons: one Section 58 Llama answer, one Section 59 Llama control and two
Section 60 Llama-4 controls. Section 58 Llama MATH CQ therefore changes from
8/30 to 9/30, while its three truncations and failed operational gate remain.
Section 59 Llama control CQ changes from 2/6 to 3/6, while its two marker/line
failures and failed gate remain. No GPT score changes.

**Observed Section 60 result (2026-09-21).** Llama-4 Scout passed the complete
frozen operational gate. All ten calls were successful and unique, all ten had
a strict standalone final marker agreeing with the stored answer, no output
was truncated, and no output violated the six-numbered-line contract. The raw
record SHA-256 is
`555862b6365c05749d14ef5903c8297d5f9bc22702cf890fc5ecfc08fd979bdb`.
Corrected descriptive CQ is 3/4 on the outcome-enriched Section 58 failure set
and 5/6 on controls, or 8/10 overall; it is not an inferential comparison.

This pass makes Llama-4 Scout operationally eligible for preregistration of a
homogeneous 90-question K=1/L=0 validation using the frozen S1 stream and the
validated strict prompt. It does not erase the Llama-3 results, authorise
K=5/L=3, or establish production-campaign readiness by itself.

---

## Section 61. Full Llama-4 Scout S1 Validation

**Recorded after the Section 60 pass and before generating any Section 61
target-model response (2026-09-21).** Section 61 is the prospectively frozen
full validation authorised by Section 60. It remains a post-hoc model-
substitution study and does not replace, pool with or relabel the Llama-3
results from Sections 58-59.

The stream is the exact historical accepted-proposal S1 sequence used in
Section 58: 30 GSM8K, 30 ARC-Challenge and 30 MATH questions in the same order,
with sequence SHA-256
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`.
The sole model is `aws/llama4_scout-17b`, previously accepted by the frozen
`disable_all` gateway probe. Temperature remains 0.7, K=1/L=0, W=10, resume is
enabled and the output ceiling is 2048 tokens. The complete ceiling is 90
calls and 184,320 output tokens.

The Section 60 output contract is retained. Each response may contain at most
six numbered reasoning lines with one short sentence per line and must then
emit its best answer even if uncertain. GSM8K and MATH require exactly one
plain final `FINAL_ANSWER` line; ARC-Challenge requires exactly one plain final
`FINAL_CHOICE` line. The marker must remain outside LaTeX, code blocks,
brackets and list items, with no following text. The corrected MATH outer-
delimiter normalizer is frozen before this run.

The call-free preflight loaded the real cached datasets, reproduced all 90
historical QIDs in order, matched the stream hash, verified the provider model
and confirmed the accepted gateway probe. The conjunctive gate requires all 90
calls to execute successfully and uniquely, 100% strict final-line
observability and stored/parser agreement, zero token-ceiling or length
truncations, and zero six-line-contract violations. Scientific dynamic range
requires GSM8K CQ minus MATH CQ at least 0.15 and a one-sided ordered-trend
p-value no greater than 0.05. Thresholds, prompts, questions and parser rules
will not be changed after output is observed.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. A pass permits only
preregistration of a separate Llama-4 Scout K=5/L=3 metric-health pilot; it
does not start that pilot or establish early-warning performance.

**Observed Section 61 result (2026-09-21).** The full validation passed its
prospectively frozen conjunctive gate. All 90 calls returned successfully and
uniquely in the historical stream order. Strict final-line observability and
stored/parser agreement were 90/90, with zero truncations, zero code-fence
failures and zero six-line-contract violations. The stream SHA-256 remained
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca` and
the raw record SHA-256 is
`35e2e561f702e305f810cd5486825870634ba3af2ba9b1147efb61aecfa8860b`.

Llama-4 Scout CQ was 29/30 (0.967) on GSM8K, 27/30 (0.900) on
ARC-Challenge and 17/30 (0.567) on MATH. The GSM8K-MATH gap was 0.400 and the
one-sided ordered-trend test gave z=-3.958 and p=0.0000378. Both frozen
scientific range criteria passed, and inference was fully observable.

The aggregate counts happen to equal Section 58 GPT-4o-mini. A post-run,
call-free contamination audit verified the exact model configuration as
`aws/llama4_scout-17b`, found zero identical raw completions across the 90
paired QIDs, and found 12 item-level correctness disagreements: six correct
only for Llama-4 and six correct only for GPT. Thus the aggregate equality is
genuine and not evidence of cache reuse or record misrouting. This paired
comparison is diagnostic and post-hoc, not a new confirmatory model-comparison
test.

Section 61 authorises only preregistration of a separate Llama-4 Scout
K=5/L=3 metric-health pilot. It does not itself establish repeat diversity,
RQEval metric coverage, nondegeneracy, detector calibration or early warning.

---

## Section 62. ASU Mistral-7B S1 Validation

**Recorded after the Section 61 result and before generating any Section 62
target-model response (2026-09-21).** The accepted proposal named Mistral-7B
as one of the three primary models and planned its confirmatory execution on a
local RTX 4090. That GPU is unavailable. Section 62 therefore moves only the
execution venue to the existing ASU OpenAI-compatible gateway and freezes the
exact provider model as `aws/mistral-7b`. The saved production probe was
executed and accepted with history, search, and time/date prompt enhancement
disabled.

This is a homogeneous K=1/L=0 model-validation arm, not a retry of the Section
47 E2H experiment and not a metric-health run. It uses the exact historical S1
sequence shared by Sections 58 and 61: 30 GSM8K, 30 ARC-Challenge and 30 MATH
questions in the same order, with sequence SHA-256
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`.
Temperature is 0.7, W=10, resume is enabled, and the output ceiling is 2048
tokens. The maximum budget is 90 calls and 184,320 output tokens.

The Section 61 response contract and corrected parser are retained unchanged.
Each response may contain at most six numbered one-sentence reasoning lines,
followed by exactly one plain final `FINAL_ANSWER` or `FINAL_CHOICE` line with
no following text. The conjunctive gate requires complete and unique
execution, 100% strict final-line observability and stored/parser agreement,
zero token-ceiling or length truncations, zero code fences and zero six-line
contract violations. Scientific dynamic range requires GSM8K CQ minus MATH CQ
at least 0.15 and a one-sided ordered-trend p-value no greater than 0.05.
Prompts, questions, parser rules and thresholds are frozen before output is
observed.

The call-free real-data preflight passed before target responses were
generated. It reproduced the 30/30/30 dataset counts, all 90 unique historical
QIDs and the frozen sequence hash, and verified both the exact ASU provider
model and the accepted `disable_all` probe. No network model call was made by
the preflight.

The prior Section 47 ASU Mistral records remain immutable evidence: all 720
calls completed and corrected CQ separated the regimes, but every K=5 set was
verbatim-identical and repeat diversity was 0.000. Since Section 62 uses K=1,
it cannot clear or contradict that limitation. A Section 62 pass authorises
only preregistration of a small ASU K=5 repeat-diversity capability check. It
does not authorise direct execution of K=5/L=3, establish metric
nondegeneracy, or establish early-warning performance. The configuration
remains `HOLD FOR EXPLICIT EXECUTION`.

**Observed Section 62 result (2026-09-21).** The ASU run completed all 90
calls and unique QIDs in the frozen order, with no truncations. The source
record SHA-256 is
`e28b727e0befed5273a6214c244ccb702b972b528abbab8fca7d577e128a4429`.
Only 61/90 responses satisfied the strict final-line contract, stored/parser
agreement was 61/90 and three responses violated the numbered-step contract.
The strict CQ values were 11/30 (0.367) on GSM8K, 6/30 (0.200) on
ARC-Challenge and 3/30 (0.100) on MATH. The GSM8K-MATH gap was 0.267 and the
one-sided ordered-trend test gave z=-2.484 and p=0.00649.

The two dynamic-range checks passed numerically, but the frozen gate failed
because final-answer observability, parser agreement and step compliance did
not pass. Inference is invalid. Formatting failures were highly
dataset-dependent, so the strict CQ values, especially ARC-Challenge, must not
be interpreted as clean model-accuracy estimates. Section 62 does not
authorise the ASU K=5 repeat-diversity capability check.

---

## Section 63. Mistral Format Sensitivity Audit

**Recorded after Section 62 and performed without model calls (2026-09-21).**
This post-hoc diagnostic preserves the immutable Section 62 source file and
uses only explicit `FINAL_ANSWER` or `FINAL_CHOICE` markers. It relaxes the
marker's position and, for ARC-Challenge, ignores text following the first
choice letter. It does not use free-form answer inference or alter stored raw
responses.

All 29 strict format failures were deterministically recoverable. Eleven
markers appeared inside numbered reasoning steps, 16 ARC-Challenge markers
included explanatory text after the choice letter, and two typed markers were
not the strict final line. Explicit-marker observability therefore rose from
61/90 to 90/90. Sensitivity CQ was 12/30 (0.400) on GSM8K, 25/30 (0.833) on
ARC-Challenge and 3/30 (0.100) on MATH. The GSM8K-MATH gap was 0.300 and the
linear ordered-trend test gave z=-2.338 and p=0.00969. The three cell means are
not monotonic because ARC-Challenge is substantially higher than GSM8K; the
significant linear score test alone must not be described as monotonicity.

This audit demonstrates that the strict Section 62 ARC result was dominated by
format noncompliance rather than answer correctness. It does not retroactively
change the prospectively frozen Section 62 gate, and it does not establish
that a relaxed parser generalises out of sample. Consequently, the small ASU
K=5 repeat-diversity capability check and any K=5/L=3 campaign remain
unauthorised. A future corrective experiment, if approved, must freeze its
parser and use disjoint questions before output is observed.

---

## Section 64. Disjoint GPT-4o-mini and Llama-4 Scout Metric-Health Pilot

**Recorded after Section 63 and before generating any Section 64 target-model
response (2026-09-21).** Section 64 is a development metric-health pilot, not
a confirmatory drift or early-warning experiment. It includes GPT-4o-mini and
the post-hoc Llama-4 Scout substitute. Mistral is excluded because its Section
47 ASU K=5 repeat diversity was 0.000 and its Section 62 operational gate
failed; this exclusion does not erase either negative result.

The selection contains five questions from each accepted-proposal dataset.
For each dataset, it uses seed-42 shuffled ranks 31-35, which follow the 30
items used in Sections 58, 61 and 62. The 15 QIDs are interleaved in
GSM8K/ARC-Challenge/MATH round-robin order and are completely disjoint from the
historical 90-question stream. The frozen sequence SHA-256 is
`9ef33e895ac902ea42383e30f4812231c76297635f468c8c91a1fb4d5333d1a5`.

Each model receives K=5 original-prompt calls and L=3 deterministic paraphrase
calls per question. Thus 15 questions x 8 calls x 2 models produce a maximum
of 240 resumable calls and 491,520 output tokens. Temperature is 0.7, the
per-call ceiling is 2048 tokens, W=5, and history, search and time/date prompt
enhancement remain disabled. The Section 61 strict six-line answer-last
contract is applied to original and paraphrased prompts. RQEval, real NLI,
real BERTScore and publication mode are mandatory; scorer fallback is not
allowed.

The call-free real-data preflight passed. Both frozen provider probes were
accepted, the selection was reproduced exactly with zero overlap, and the
scorer preflight reported NLI and BERTScore ready with no problems or warnings.
The conjunctive per-model gate requires 120/120 records and 15/15 query rows,
five run plus three paraphrase records per QID, 100% successful calls, strict
typed markers and stored/parser agreement, zero truncations, zero six-line
violations and K-repeat diversity at least 0.80. All non-RS metrics must be
finite and in range for all 15 queries; RS must be evaluable for at least six
queries overall and at least one query in each dataset. Every metric must have
at least two distinct finite values. Both models must pass.

The configuration remains `HOLD FOR EXPLICIT EXECUTION`. Accuracy and detector
alarms are descriptive in this small, composition-balanced selection. A pass
permits only preregistration of a separate full 90-question GPT-4o-mini plus
Llama-4 Scout K=5/L=3 campaign; it does not start that campaign or establish
early warning.

**Observed Section 64 result (2026-09-21).** All 240 calls completed. Each
model produced 120/120 records and 15/15 per-query rows on the exact frozen
selection. Both models had 100% strict typed markers and stored/parser
agreement, zero truncations, zero six-line violations, complete finite metric
coverage and a clean real-NLI/BERTScore preflight. Thus infrastructure,
formatting and scorer fallback did not cause the failure.

GPT-4o-mini had K-repeat raw-completion diversity 15/15 = 1.000. Llama-4 Scout
had diversity 9/15 = 0.600, below the frozen 0.80 threshold. Llama-4's six
raw-identical groups comprised four GSM8K, one ARC-Challenge and one MATH
question. RS was evaluable for 12 GPT and 11 Llama-4 questions, satisfying
coverage, but every evaluable value was exactly 1.0. Each model therefore had
only one distinct RS value and failed the all-metric nondegeneracy requirement.
The overall Section 64 gate failed, and the full 90-question campaign is not
authorised.

---

## Section 65. Section 64 Failure Diagnostic

**Recorded after Section 64 and performed without model calls (2026-09-21).**
All source manifest, record and per-query hashes match the immutable Section
64 assessment. The audit separates the RS perturbation design from K-repeat
sampling diversity.

For every QID, the three so-called paraphrase prompts are the complete original
prompt plus one deterministic wrapper: a fixed prefix, a fixed suffix, or a
second fixed prefix. Consequently, all 90 paraphrase calls retain the original
question and output-format instructions verbatim. This implementation does not
perform a substantive paraphrase. It measures sensitivity to superficial
wrapping and is too weak to establish robustness to meaning-preserving
rephrasing.

RS evaluability exactly equalled the rqeval C-subset: 12 CQ-correct primaries
for GPT and 11 for Llama-4. All corresponding paraphrase answers remained
correct, producing RS range [1.0, 1.0] in both models. This is not an answer
parser failure. It is a genuine degeneracy under the current weak perturbation
operator.

GPT had at least two distinct raw K completions on all 15 questions, but only
5/15 questions had more than one distinct extracted answer. Llama-4 had raw
diversity on 9/15 and answer diversity on 3/15. The Llama-4 sampling failure is
therefore independent of RS and remains a second blocker.

Section 65 does not alter the frozen Section 64 decision. Thresholds will not
be relaxed, and the same paid protocol will not be repeated. Any future
development protocol must preregister a frozen bank of genuine,
meaning-preserving paraphrases with semantic-equivalence checks. It must also
address Llama-4 repeat diversity independently before a full campaign can be
considered.

---

## Section 66. Independent RS and Llama-4 Sampling Repairs

**Preregistered before new target-model output (2026-09-21).** Section 65
identified two independent failures, so Section 66 does not combine their
repairs in one gate.

The RS arm freezes three substantive question-body rewrites for each of the 15
Section 64 QIDs. The original body is stored beside every rewrite, ARC answer
choices are preserved verbatim, and the complete source output contract is
attached mechanically at stream construction. The bank SHA-256 is
`6a3e426f713b98e2c02d43640154b9ec7848a6f427e6d1f819dbaa795ae30a7f`.
The authoring audit is complete; independent human review remains required
before confirmatory use.

The development pilot uses the same 5 GSM8K, 5 ARC-Challenge and 5 MATH QIDs,
but K=1/L=3 for GPT-4o-mini and Llama-4 Scout. This produces at most 120 model
calls and isolates RS from K-repeat sampling. The frozen conjunctive gate
requires 60/60 records per model, complete strict-marker/parser agreement,
zero truncations and six-line violations, at least six evaluable RS values
overall, at least one in each dataset, and at least two distinct RS values.
A pass permits only planning of a fresh holdout metric-health protocol.

The call-free preflight passed: all 45 rewrites were loaded from the exact
frozen bank, every output contract and ARC choice block was preserved, both
production gateway probes matched, and the real NLI/BERTScore scorers were
ready. This paid arm remains `HOLD FOR EXPLICIT EXECUTION`.

Llama-4 repeat diversity is tested separately. `OpenAICompatibleClient` now
has an opt-in `send_seed` setting that forwards the monitor's existing
per-call seed while preserving identical user prompts. The default is false,
so no historical protocol changes. A five-call ASU capability probe is frozen
on `gsm8k-00131`, a Section 64 QID whose five Llama-4 raw completions were
identical. The probe completed with all five seed values accepted, but all five
raw completions and extracted answers were identical. Its frozen capability
gate therefore failed. The result SHA-256 is
`0d278e0f485829d329851e6bc22c90f9833022cb4412f30cf2bb64e07684ab5a`.
This observation does not establish that the provider ignores every seed on
every prompt, but it does block the preregistration of a larger seeded K=5
experiment under this branch. No threshold relaxation or selective retry is
authorised. The independent RS arm remains `HOLD FOR EXPLICIT EXECUTION`.

**Observed Section 66 RS result (2026-09-21).** All 120 calls completed. Each
model produced 60/60 records on the exact frozen sequence, with 100% successful
calls, strict typed markers and stored/parser agreement, zero truncations and
zero six-line violations. GPT RS was evaluable on 12 questions and Llama-4 RS
on 11, with coverage in all three datasets. Both models had raw RS range
[2/3, 1] and two distinct values. The frozen formal gate therefore passed for
both models. This pass permits planning only and does not authorise a fresh
holdout or the full campaign without the post-run semantic audit below.

---

## Section 67. Section 66 Semantic-Equivalence Audit

**Recorded after Section 66 and performed without model calls (2026-09-22).**
The immutable Section 66 record and per-query hashes match its formal
assessment. The audit examined the two QIDs responsible for raw RS=2/3.

For GPT-4o-mini on `gsm8k-00174`, the primary answer was `95 minutes` and the
third paraphrase answer was `5700 seconds`. The reasoning correctly converted
1.5 minutes to 90 seconds, added 5 seconds per potato and multiplied 95 seconds
by 60. These answers are mathematically identical. Explicit time-unit
normalisation changes that query's RS from 2/3 to 1, leaving all 12 evaluable
GPT RS values at 1.0. Its apparent nondegeneracy was therefore an answer-
equivalence scoring artifact, not paraphrase fragility.

For Llama-4 Scout on `math-counting_and_probability-00009`, the affected
paraphrase derived the correct expression `6! - 5! * 2!` but reported `384`
instead of `480`. This is a genuine arithmetic error. Llama-4's corrected RS
range remains [2/3, 1] across its 11 evaluable questions.

The formal Section 66 gate remains passed because preregistered decisions are
immutable. Its conjunctive scientific interpretation does not pass: only one
of two models retains genuine RS variation after semantic equivalence. No
fresh holdout or full campaign is authorised. Any successor protocol must
freeze a semantic answer-equivalence policy before target outputs are observed
and must not be tuned selectively to these two cases.

---

## Section 68. Prospectively Frozen Semantic Answer Policy

**Frozen before any successor target-model output (2026-09-22).** Section 67
showed that surface numeric matching can both miss equivalent quantities and
accept equal numbers carrying incompatible units. `semantic-v1` is therefore
an explicit, opt-in CS/RS answer-equivalence policy. The default remains
`None`; historical configs, records and decisions are unchanged.

The rqeval metric formulas are retained. CS still computes pairwise agreement,
but an enabled policy supplies the pairwise equivalence relation. RS still
uses the rqeval C-subset: the primary answer must pass the existing gold
correctness check. Each paraphrase then counts as correct if it passes the
existing gold check or is semantically equivalent to the accepted primary
answer.

For GSM8K, both compared answers must state explicit recognized units. Only
units in the same frozen dimension may be converted. The registry covers time,
length, mass, metric volume and temperature. The policy never infers a missing
unit from the prompt or unitless gold, never compares different dimensions and
never performs currency conversion. ARC-Challenge retains exact answer-choice
identity. MATH uses the already frozen symbolic scalar and ordered-sequence
matcher.

The policy SHA-256 is
`d823546b24df250c0f20fd7577e5ec04f94897616472a716dde2b5577e57db58`.
Its independently stored 20-case validation table has SHA-256
`90e27a6a5446cc7e66d8bd88389ada776b6678b78875a7d4ca2a6f16f8b11c37`.
The call-free preflight passes 20/20 cases, including positive conversions and
negative same-number/wrong-unit, cross-dimension, unitless and currency cases.
The historical policy default is verified as `None`.

Section 68 authorises only preregistration of a fresh, disjoint semantic-RS
development holdout. It does not authorise paid calls, retrospective relabeling
of Section 66, or the full 90-question campaign.

---

## Section 69. Terminal Hash-Verified Evidence Synthesis

**Recorded after Section 68 and performed without model calls (2026-09-22).**
This section replaces the previously possible successor holdout with a final
evidence synthesis. It introduces no new threshold, model response, rescore or
retrospective gate. A frozen manifest records the paths and SHA-256 hashes of
24 evidence sources spanning Sections 47-68. The synthesis checks all hashes,
re-evaluates 19 cross-section consistency conditions and writes a structured
JSON assessment, a hypothesis decision table and a narrative report.

All 24 hashes and all 19 consistency checks pass. The final scientific
classifications are:

| Claim | Final classification |
| --- | --- |
| Non-CQ metrics reliably precede CQ | Not supported in the only eligible frozen detector-timing track |
| CQ tracks difficulty change | Supported across the frozen E2H holdouts and clean Llama-4 S1 validation |
| Framework identifies measurement failures | Supported as an engineering/methodological contribution |
| CS/SS cross-model generalisability | Not evaluable on the current ASU endpoints |
| RS paraphrase robustness | Mixed and inconclusive |

The negative early-warning decision is not a claim of universal impossibility.
In the eligible GPT track, CQ detected both transitions while no non-CQ metric
led CQ in both intervals; correcting the reasoning-trace parser did not rescue
LS or SS. Llama-3, Llama-4 and Mistral did not provide sufficient repeat
diversity for a valid cross-model CS/SS test. The genuine-paraphrase arm found
one unit-equivalence artifact for GPT and one genuine arithmetic failure for
Llama-4, which is insufficient to establish a stable cross-model RS signal.

No full K=5/L=3 campaign or additional paid pilot is authorised. This chain
ends at Section 69. Section 70 may be opened only if a newly demonstrated
software defect invalidates immutable evidence used by this synthesis;
scientific disappointment, a new dataset or a desire for more power is not
sufficient. Any future confirmatory work is a new protocol that begins with
the prospectively frozen `semantic-v1` policy, genuine paraphrases and a
backend that demonstrates usable stochastic repeat diversity before campaign
execution.

Reproduction command:

```cmd
python scripts\synthesize_final_evidence.py
```

Primary output:
`results/final_evidence_synthesis/FINAL_EVIDENCE_SYNTHESIS.md`.

---

## Independent Protocol RAT-v1. Risk-Adjusted Controlled Trajectories

**Prospectively frozen after the Section 69 chain closed (2026-09-26).** This is
not Section 70, does not relabel any historical gate and does not change the
terminal Section 69 conclusions. It is a separate confirmatory route intended
to test a narrower and more identifiable question: whether reasoning-quality
signals at one controlled difficulty level forecast the first CQ failure at
the next level beyond what level, task family and model already predict.

The benchmark contains three exact-solver templates: ordered arithmetic,
weighted shortest path and precedence-constrained scheduling. Each template
has 20 independent families and every family has six nested levels, giving 60
families and 360 items. Family assignment is frozen before any model output:
24 calibration families (144 items), 18 development families (108 items) and
18 untouched holdout families (108 items). Each item has three substantive,
meaning-preserving prompt surfaces and a bounded plain-integer answer
contract. Gold answers are regenerated by the template solver.

The primary outcome is first CQ failure at level `t+1` among trajectories that
remain correct through level `t`. The baseline discrete-time hazard model uses
trajectory level, task template and model. The full model adds prospectively
measured RS, LS and ES. K=1/L=3 is the core design; CS and SS may enter only
for provider-model cells that pass a separately frozen repeat-diversity gate.
Families, not individual prompts, are the grouping and resampling unit. The
primary confirmatory score is family-clustered holdout log-loss improvement;
the early-warning claim requires its bootstrap 95% interval to be above zero
in at least two preregistered models after multiplicity correction.

The deterministic bank SHA-256 is
`8404ed2743e2bdb61b004a6cb1f88fb2acfec60e3305270d183d857225016c31`;
the protocol SHA-256 is
`2e79ad402c4918632a53e3d76a25cada3b2323791978cddb711ceac045fb906f`.
The call-free structural preflight passes all checks: counts, family-level
split isolation, six-level nesting, exact-solver answers, prompt contracts,
paraphrase uniqueness, monitor conversion and deterministic regeneration.
Paid calls remain explicitly unauthorised. This pass permits only preparation
and call-free validation of a calibration-split execution configuration.

That frozen calibration configuration has now passed its separate call-free
operational preflight. It contains only the 24 calibration families (144
items/model), K=1/L=3, GPT-4o-mini, Llama-4 Scout and Mistral-7B, a 768-token
per-call ceiling and mandatory resume. All three saved ASU production probes
match, history/search/context enhancement are disabled, and the real NLI and
BERTScore scorers load without fallback. The ceiling is 576 calls/model, 1,728
calls total and 1,327,104 output tokens. The config SHA-256 is
`3b6ffe15e12056d8ddd0878cad5756b6fa96dd9815e316a38285cdf09ce98345`.
The gate remains non-executing and requires explicit investigator approval
before any paid call.

Before any calibration output, the executable inferential path and post-run
transition gate were also frozen. The risk set includes level `t` only while
CQ remains correct through that level and uses RS/LS/ES at `t` to predict the
first failure at `t+1`. Ridge `C` is selected from {0.01, 0.1, 1, 10} with
family-grouped folds. Confirmatory holdout inference uses 2,000 family-cluster
bootstrap samples, seed 20260926, one-sided alpha 0.05 and Holm correction.

The calibration gate is operational rather than confirmatory. Per model it
requires at least 16 level-1 entries, six first failures, four censored
families, events in two task templates, complete finite nondegenerate RS/LS/ES
and the frozen record/format contract. At least two preregistered models must
pass before development may be prepared. Synthetic tests demonstrate that the
locked analysis recovers a planted prospective signal. Structural preflight
hash-checks the analysis source, calibration assessor and threshold file, so
post-output edits invalidate the gate rather than silently changing the test.

**Calibration result and frozen call-free signal diagnostic (2026-09-27).**
The approved calibration completed with 576 records per model, complete API
execution and zero token truncations. The original operational gate remains a
failure. GPT-4o-mini and Llama-4 Scout met answer observability, event-density
and predictor-health requirements but produced respectively 47 and 51 traces
above the strict eight-numbered-line limit. Mistral-7B also had only 0.398
strict final-marker observability, 12 level-one entries and zero censored
families. These thresholds were not relaxed after observing output.

Before inspecting predictor performance, a separate call-free diagnostic was
frozen for GPT-4o-mini and Llama-4 Scout. It compared the preregistered baseline
against baseline plus RS/LS/ES using four-outer/three-inner family-grouped
nested validation. The combined mean log-loss improvement was -0.021522 with
one of four positive folds. GPT-4o-mini alone was +0.034123 (3/4 positive),
whereas Llama-4 Scout was -0.024805 (2/4 positive). All frozen continuation
checks failed. This is calibration-stage model-heterogeneity evidence, not a
holdout claim, and it does not authorise development or holdout model calls.

**Finite metric-subset termination audit (2026-09-27).** After the primary
screen failed, one final call-free gate prospectively enumerated all seven
non-empty subsets of RS, LS and ES under the same grouped nested validation.
No subset qualified. RS improved GPT-4o-mini by +0.073802 with 4/4 positive
folds but was effectively null for Llama-4 Scout (+0.000021; 2/4). LS improved
Llama by +0.014473 with 3/4 folds but degraded GPT by -0.011417 with 1/4. This
opposing metric preference is retained as evidence of model-specific
measurement behavior, not a basis for post-hoc model selection. The frozen
termination rule closes RAT-v1 at calibration; development and holdout remain
untouched and no additional paid RAT campaign is authorised.

---

## Independent Protocol CDR-v1. Controlled Longitudinal Dialogue Reasoning

**Prospectively designed after RAT-v1 terminated (2026-09-27).** The
investigator identified a construct mismatch in the prior designs: independent
GSM8K-like questions do not create a correlated conversation state, so their
order cannot directly test degradation caused by accumulated dialogue context.
CDR-v1 is a separate protocol, not Section 70 and not a reinterpretation of
Section 69 or RAT-v1.

CDR-v1 uses exact-solver 12-turn cases in three domains. Every later turn
depends on a shared evolving state. Whole dialogue families are assigned to
calibration, development, and holdout before model output. Three matched arms
receive the same current user turn: actual model history (`self_history`),
canonical correct assistant history (`oracle_history`), or only the verified
current state (`state_summary`). These contrasts distinguish total context
penalty, propagated model error, and long-history load.

The first feasibility pilot is intentionally narrow: one calibration dialogue
per template, GPT-4o-mini and Llama-4 Scout, K=1/L=0, temperature 0, and 216
resumable calls. Stage 1 measures CQ and operational observability only. A
context penalty of at least 0.10 plus self-history failures in at least two
dialogues for at least one model is required before RS/LS/ES metric work may be
prepared. This gate prevents an early-warning campaign unless the target
within-dialogue degradation phenomenon is first demonstrated.

Implementation and call-free preflight are authorised. Paid pilot execution,
development, holdout, and scientific early-warning claims are not authorised
by this entry. The complete design is in `CONTROLLED_DIALOGUE_PROTOCOL.md` and
`configs/controlled_dialogue_v1.yaml`.

### CDR-v1 integration invalidation and CDR-v1.1 transport correction

**Recorded after the completed CDR-v1 pilot (2026-09-28).** GPT-4o-mini and
Llama-4 Scout each completed all 108 requested records without truncation.
The original gate failed answer observability. More importantly, a frozen
call-free audit found the same deterministic pattern in both models: accuracy
was 1.0 at turn 1 and 0.0 after turn 1 for both native `self_history` and
`oracle_history`, their CQ sequences were identical, and the two-message
`state_summary` arm had accuracy 1.0. Responses explicitly requested earlier
facts even though the local request records contained those facts in correctly
ordered role messages.

This shared signature is treated as a gateway history-transport failure, not a
reasoning decline. The original records, hashes, and failed gate remain
immutable. They cannot support a context-degradation or early-warning claim.

CDR-v1.1 addresses only transport. It retains `enable_history: false` to avoid
hidden server state, but serializes prior role-labelled turns into the system
message and sends the current user turn separately. Before any replacement
pilot, a frozen 12-call probe tests turn 2 for three topics, two history arms,
and both models. The probe requires 100% execution, answer accuracy, strict
marker validity, and zero truncation. A pass authorizes call-free preparation
only; the larger pilot still requires a separate decision and approval.

**Transport probe resolution and fresh pilot freeze.** Eleven of the 12 packed
transport calls passed. The sole failure was the Llama self-history cohort
cell, whose response recovered every prior fact and applied the requested
update correctly but stopped at exactly the 256-token ceiling before emitting
its final answer. The frozen original probe gate remains failed. A separate
one-cell retry reused the exact messages with a 512-token ceiling and passed:
answer 122, valid final marker, 298 output tokens, and no truncation or ceiling
hit. This establishes packed transport as operationally feasible for both
models; it is not a scientific degradation result.

A fresh CDR-v1.1 calibration-only feasibility pilot is now frozen and passes
call-free preflight. It uses three 12-turn dialogues, all three matched history
conditions, GPT-4o-mini and Llama-4 Scout, K=1/L=0, and a 512-token ceiling,
for 216 resumable calls and at most 110,592 output tokens. Turn 1 is excluded
from context-effect contrasts because no history exists at baseline. The gate
requires complete transport, no token-ceiling hits, state-summary accuracy at
least 0.95, oracle-history accuracy at least 0.85, a context penalty at least
0.10, and self-history failures in at least two dialogues for at least one
operationally valid model. This freeze permits explicit investigator
consideration of the paid run; it does not itself authorize execution or any
scientific claim.

**Fresh pilot outcome and call-free diagnostic.** The 216-call CDR-v1.1 pilot
completed with 100% request success, strict marker validity and stored-answer
agreement in both models, with no truncation or token-ceiling hit. Its frozen
gate remains failed because post-baseline oracle-history accuracy was 0.485 for
GPT-4o-mini and 0.576 for Llama-4 Scout, below the preregistered 0.85 floor.

A separately frozen descriptive diagnostic did not alter that decision. It
found state-summary accuracy 1.0 versus self-history accuracy 0.636 for GPT and
0.818 for Llama. State summary was uniquely correct in 12/33 paired GPT cells
(one-sided exact p=0.000244) and 6/33 paired Llama cells (p=0.015625), with no
reverse discordance. Both models shared self-history failure in the budget
dialogue from turn 7 through turn 12; those tails remained wrong after the
first failure. This is exploratory cross-model evidence of accumulated-context
load and persistent error propagation, not a confirmatory claim, because only
three dialogue clusters were observed.

The oracle arm cannot cleanly identify error propagation: its context was much
shorter than self history and it was less accurate in both models. A successor
design may therefore be prepared with information-format-matched
`self_history`, `user_only_history`, and `state_summary` arms. The comparison
of summary versus user-only history targets raw-history integration load; the
comparison of user-only versus self history targets the incremental effect of
prior model prose. This diagnostic authorizes design work only, not paid calls,
development or holdout access, or a scientific claim.

### CDR-v1.2 frozen user-only-history mechanism probe

**Recorded prospectively on 2026-09-28, before any CDR-v1.2 output.** A new
mechanism arm replaces the information-format-mismatched oracle control with
`user_only_history`. It packs every prior user turn into the system context,
omits all prior assistant responses, sends the current user turn separately,
and retains disabled server-side history. This isolates accumulated raw-user
history from the additional effect of prior model prose more directly than the
CDR-v1.1 oracle arm.

The probe adds one condition for the same three 12-turn calibration dialogues
and the same two models: 36 calls per model, 72 calls total, and a maximum of
36,864 output tokens. It reuses, without rerunning, 144 frozen CDR-v1.1
`self_history` and `state_summary` records. Turn 1 is excluded from mechanism
contrasts. The preregistered primary contrast is `state_summary` versus
`user_only_history`; `user_only_history` versus `self_history` is the secondary
mechanism description.

For each model, the mechanism gate requires a primary accuracy difference of
at least 0.10, a one-sided paired exact p-value no greater than 0.05, and
user-only failures in at least two dialogues. Both models must qualify, in
addition to operational integrity checks. The call-free preflight passes and
locks the data, code, analysis, and gate hashes. This entry authorizes only
investigator consideration of the 72-call probe. It does not itself authorize
paid execution, development or holdout access, or a scientific claim.

**CDR-v1.2 outcome and post-hoc capacity diagnostic.** The authorized 72-call
arm completed, but its frozen formal gate failed. GPT produced three and Llama
seven responses ending exactly at the 512-token output ceiling; these 10 cells
accounted for all strict-marker and stored-answer-agreement failures. The
original records and failed decision remain immutable.

A source-hashed, call-free sensitivity diagnostic separated observable wrong
answers from capacity-censored answers. GPT was correct on 24/30 uncensored
post-baseline cells. Even under the optimistic assignment that all three
censored GPT cells were correct, the state-summary advantage is 0.182, with
6-0 complete-case discordance, p=0.015625, and failures in three dialogues.
That model's raw-history-load signal is robust to censoring. Llama was correct
on 24/26 uncensored cells. If all seven censored cells were correct, the gap is
only 0.061, with 2-0 discordance, p=0.25, and failures in one dialogue. Its
mechanism result is therefore unresolved and capacity-dependent.

This diagnostic does not rescue or relabel the frozen gate and cannot support
a scientific claim. An exact-message retry of only the 10 censored cells is
now prospectively frozen and passes call-free preflight. It preserves the
messages and original call seeds, changes only the output ceiling from 512 to
1,024 tokens, and requires 3 GPT plus 7 Llama calls. The retry is resumable and
has a maximum of 10,240 output tokens. Its analysis substitutes only these
cells for a capacity-adjusted sensitivity classification and cannot alter the
original failed gate. The retry and all later validation calls require
separate explicit approval.

**CDR-v1.2a retry outcome.** The 10 authorized exact-message calls completed
with 100% request success, marker validity, stored-answer agreement, message
identity, and freedom from truncation or the 1,024-token ceiling. GPT answered
3/3 retried cells correctly and Llama 4/7. Substituting only those completed
outputs yields user-only-history accuracy 0.818 for GPT and 0.848 for Llama,
versus 1.0 state-summary accuracy. The paired state-summary advantages are
0.182 (6-0, p=0.015625, three dialogues) and 0.152 (5-0, p=0.03125, two
dialogues), so both models satisfy the prospectively frozen mechanism
thresholds in the capacity-adjusted sensitivity analysis.

The secondary user-only versus self-history contrast is significant for GPT
(+0.182, 7-1, p=0.035156) but not Llama (+0.030, 2-1, p=0.5). Accordingly, the
cross-model primary mechanism is accumulated raw-history integration load and
its mitigation by a verified structured state summary. Error propagation from
prior model prose remains model-dependent and secondary.

This result does not relabel the failed original CDR-v1.2 gate. It is strong
exploratory evidence, not a scientific claim, because the capacity correction
was diagnostic and only three dialogue clusters were observed. It authorizes
call-free preparation of a prospective validation on the nine unused
calibration dialogues. No validation, development, or holdout calls are
authorized by this entry.

### CDR-v1.3 cluster-aware primary calibration validation

**Recorded prospectively on 2026-09-29, before any CDR-v1.3 output.** The
capacity-adjusted cross-model mechanism result authorizes preparation of a
larger calibration validation. To control cost and keep the primary question
focused, Stage 1 uses only `state_summary` and `user_only_history` on all nine
unused calibration dialogues. The three feasibility dialogues are excluded.
With 12 turns and two models, the stage requires 432 resumable calls at a
1,024-token ceiling and a maximum of 442,368 output tokens. The 216-call
`self_history` arm is deferred until after the primary result.

The independent unit is the dialogue cluster, not an individual turn. Each
model's primary statistic is the equal-weight mean of nine within-dialogue
state-summary minus user-only accuracy differences over turns 2-12. Its
one-sided p-value uses the complete exact sign-flip distribution over the nine
cluster differences. Both models must show a mean difference of at least 0.10,
p<=0.05, positive differences in at least seven dialogues, and state-summary
accuracy of at least 0.95.

The longitudinal prerequisite is separately frozen. Early history covers
turns 2-6 and late history covers turns 7-12. The per-dialogue interaction is
the late condition gap minus the early condition gap. At least one model must
show mean interaction >=0.05, exact p<=0.05, and positive interactions in at
least six dialogues; the other model must have a nonnegative mean. This guards
against interpreting a static condition difference as time-dependent decline.

The complete call-free preflight passes. This entry authorizes only explicit
investigator consideration of the 432-call Stage 1 run. It does not authorize
paid execution, the conditional self-history arm, development or holdout
access, or a scientific claim.

**Outcome recorded after execution on 2026-09-29.** The frozen formal gate
failed and remains failed. GPT-4o-mini passed all operational checks and showed
a mean dialogue-level state-summary advantage of 0.282828 (one-sided exact
cluster sign-flip p=0.005859; positive in 8/9 dialogues). Its late-minus-early
interaction was 0.192593 (p=0.023438; positive in 7/9). However, its
state-summary accuracy was 94/99 = 0.949495, one correct answer below the
95/99 minimum implied by the frozen 0.95 threshold. All five wrong
state-summary cells were fully observed, had strict final markers agreeing
with stored answers, and were below the token ceiling. They are arithmetic
errors and provide no evidence of parsing or capacity failure.

Llama-4 Scout showed a mean primary gap of 0.181818 (p=0.003906; positive in
8/9 dialogues) and a longitudinal interaction of 0.211111 (p=0.03125;
positive in 6/9). It failed the operational gate because one user-only response
reached the 1,024-token ceiling. A hash-frozen, call-free post-hoc sensitivity
analysis preserves the failed decision and treats the censored answer as
correct in the most favorable bound. The Llama primary gap remains 0.171717
with p=0.003906 and 8/9 positive dialogue gaps. Its individual longitudinal p
becomes 0.054688, while GPT continues to satisfy the frozen requirement that
at least one model show a significant positive interaction and neither model
show a negative mean interaction.

The assessment SHA-256 is
`617e38629156f3f6f97dd929b02152be47326a3b1e1d425c6ace810728a2d145`; the
sensitivity report SHA-256 is
`6c86f8902fa449b4393f27a9216704eebfb6b627db970d3aab7700d4f4da5548`.
These findings are strong calibration evidence that raw-history integration
load grows over dialogue time and is mitigated by verified structured state.
They do not authorize threshold relaxation, relabelling the gate, the deferred
self-history arm, additional paid calls, development/holdout access, or a
scientific claim. The next permitted activity is call-free preparation of a
fresh independent replication design.

### CDR-v1.4 outcome and call-free warning diagnostic

**Content follow-up authorized by the investigator:** Saved CDR-v1.4 outputs
were checked for explicit numeric equality errors and template-specific
transition violations. Transition checks use the previous model answer and
the current user update, never gold state. An incorrect prior answer can
persist consistently or a correct recovery can violate this relation, so this
is not equivalent to accuracy. Indeed, on these additive exact tasks it cannot
precede the first wrong answer if all preceding answers are correct.

The conservative arithmetic parser covers 103/108 GPT and 99/108 Llama
user-history responses, checking 495 and 378 numeric equalities. Nine GPT and
four Llama responses contain a detected arithmetic error. Unsupported prose or
symbolic expressions remain unchecked; absence of a detected error is not a
certificate of correct reasoning. Transition checks cover all 99 post-baseline
responses per model and flag 52 GPT and 37 Llama responses. Neither direct
content flag appears one turn before any of the nine GPT or eight Llama first
failures. This is a diagnostic limitation, not a negative result for all
possible reasoning metrics.

With whole template families held out, combining content predictors with turn
and context length reduces overall next-turn Brier loss from 0.15098 to 0.14695
for GPT and from 0.22684 to 0.20415 for Llama. This includes post-failure turns
and therefore does not establish early warning before initial failure. The
analysis remains post-hoc and does not separate turn, operation schedule,
history length or response-format constraints causally. No fresh model calls
or development/holdout records were used.

Reproduction: `python scripts\analyse_dialogue_content.py`. Reports, per-turn
metrics, audited equations, out-of-group predictions and PNG/PDF figures are
under `results/controlled_dialogue_v1_4_content_diagnostic`. Temporal plots
include per-turn accuracy, metric coverage, content flags, token lengths, and
individual dialogue error timelines. The investigator's preference to include
time/turn figures in future result presentations is recorded in AGENTS.md.

The investigator authorized the replication and subsequently authorized
call-free early-warning exploration. The formal replication gate remains
failed: GPT has three noncanonical final markers, all trailing periods on
correct first-turn budget answers. None affects the scored post-baseline
contrast; all post-baseline markers pass for both models. Llama passes all
frozen checks. Source record hashes are verified by the diagnostic script,
and the original assessment is reproduced before analysis.

Exploratory next-turn prediction uses CDR-v1.4 user-only-history records alone
(99 prediction opportunities per model). It does not pool the differently
prompted CDR-v1.3 data. Four fixed feature sets compare turn, turn plus context
length, addition of tokens and token change, and addition of latency. Logistic
regression uses C=1, training-only standardization, no hyperparameter search,
and threshold 0.5. Each dialogue, and separately each entire template family,
is held out in turn. Future response information and gold values are excluded
from predictors. Gold labels identify outcomes and retrospectively define
first-failure opportunities, not an online eligibility filter.

Adding tokens improves leave-dialogue-out Brier loss over turn plus context
by 0.00629 for GPT and 0.00439 for Llama, but worsens leave-template-out loss
by 0.00915 and 0.01594 respectively. At threshold 0.5, the token variant warns
one turn before 3/9 GPT and 2/8 Llama first failures in leave-dialogue-out
evaluation, with false alarms at 2/40 and 7/53 preceding negative opportunities.
These are per-opportunity false-alarm rates, not per-dialogue alarm rates.
Latency has no consistent incremental benefit. No transferable early-warning
metric is established by this screen. All results are post-hoc calibration
diagnostics, without significance or publication-validity claims.

The three shared template families, turn-specific update schedule, and the
eight-line response instruction limit interpretation: history length, turn
position, and operation type are not causally separated. Good aggregate
next-error prediction is not itself evidence of warning before the first
error. Further work should test content-based signals against these baselines
on saved data before considering additional paid runs.

### CDR-v1.4 fresh calibration replication

**Recorded prospectively on 2026-10-01, before any CDR-v1.4 model output.** A
fresh replication is prepared without opening the existing development or
holdout splits. The deterministic seed `20261001` creates nine new dialogues
in a separate `cdr-v14-*` namespace across the same budget, incident, and
cohort template families. The frozen bank SHA-256 is
`f72067c5903b4cd4b1ae2cb790665e4344e1f9330f7ffaaf835b1dd3c7439cec`.

The replication retains the two models, `state_summary` versus
`user_only_history` contrast, 12 turns, nine independent dialogue clusters,
mean gap >=0.10, one-sided exact sign-flip p<=0.05, positive gaps in at least
7/9 dialogues, and the separately frozen early-versus-late interaction. The
state-summary absolute accuracy floor is prospectively revised to 0.90 as an
assay-validity condition. This revision is explicit and occurs before new
outputs: the former 0.95 cutoff maps discontinuously to 95/99 = 0.9596 and
made one fully observed arithmetic answer the sole reason GPT failed despite
a large, cluster-significant effect. The original CDR-v1.3 gate remains failed
and unchanged.

To address the independent capacity failure observed in Llama, CDR-v1.4 uses a
2,048-token ceiling and requests no more than eight short reasoning lines. The
run is 216 calls per model, 432 total, checkpointed and resumable. Its runner
emits completed/total calls, percentage, calls per minute, and ETA after every
new checkpoint. The complete call-free preflight passes. This entry does not
authorize paid execution, further calls, development or holdout access, or a
scientific claim.

### Six-metric dialogue acquisition amendment (2026-10-01)

Limited pilot approved after the offline sampling review: do not open the full
campaign. Fix budget-002 by sequence (next budget dialogue after the probe),
not by selecting metric outcomes. Both models, both conditions, 12 turns,
K=5/L=2 produce 336 logical calls and a 688,128 output-token ceiling before
retry overhead. All six plots, coverage and descriptive pre-first-error values
are mandatory. No learned threshold, significance claim, automatic expansion,
or holdout access is authorized. No error in the pilot means first-error
prediction was not testable. This dialogue has prior calibration exposure;
it is explicitly not independent confirmation. The mini runner verifies the
frozen sampling audit SHA and underlying probe hashes; the old runner remains
blocked by its original rule. Remote execution is left to the user's command.

Subsequent offline probe audit: the original per-cell diversity gate failed
because Llama returned one unique raw response at turn 2, despite four at turn
11. Preserve that gate and source files. `sampling-readiness-v2-posthoc` reports
an alternative acquisition criterion: every cell operationally valid (including
strict final marker/extracted-answer agreement), and at least one diverse cell
per model. This rule was revised after observing results and must not be called
independent validation or evidence of provider seed forwarding. A constant cell
can represent genuine agreement; final-answer diversity is separate from raw
text diversity. The runner's original execution gate is not silently relaxed.
The diagnostic covers only two turns from one dialogue, with no paraphrases;
RS and temporal ES cannot be measured. No trend, warning, or sensitivity claim
is authorized. Raw records and the failed original report are hash-checked
before and after offline scoring; new artifacts use a separate directory.

The user requires all six trajectories. Previous K=1,L=0 records cannot
estimate CS/RS/SS. A separate plan collects K=5 (primary included), L=2 frozen
paraphrases, temperature 0.7, with the same two CDR-v1.4 conditions and token
cap. L=2 reflects the existing bank, not the proposal's L=3. Old temperature-zero
records are not pooled into this new condition. Cost: 3,024 calls plus a separate
20-call diversity probe. No paid calls were made during preparation.

Real cached NLI and BERTScore must load before acquisition. The conservative
diversity probe does not prove provider parameter forwarding. Checkpoints
distinguish repeats even with identical prompts. Failed calls remain visible,
are excluded from scores and are not automatically replaced. Scorer failures
stop scoring, never silently create constants. Missing values are NaN, not zero.

RS retains its correct-primary subset; ES uses correctness and causal five-turn
token normalization, with warmup flagged. Neither alone supports gold-free
deployment. LS uses rqeval consecutive-step hard NLI labels; SS uses pairwise
BERTScore with distilbert-base-uncased. Graphs show each model and dialogue.
Prior gates remain immutable. Early-warning tests must use pre-error features,
dialogue/template-separated evaluation and multiplicity control across six
metrics (e.g. Holm on frozen tests or independent confirmation).

## DBQR-RQ-v1 dataset selection (2026-10-01; call-free only)

The investigator selected DBQR-QA's supplied-table conversational reasoning
task as the next candidate. This is a protocol adaptation, not evidence that
reasoning degrades. LIC is frozen and preserved; historical gates are unchanged.
The practice archive is hash-pinned and structurally audited, without loading
pickles or executing reference programs. Semantic review, independent label
recalculation, input-size measurement and typed evaluation remain pending.
Training/test are unopened. No paid run or sample size is authorized.
See DBQR_RQ_AUDIT.md for observed counts and specific first-pass review flags.

### DBQR representation correction

The Training audit rejected direct raw-table prompting: only 9/20 dialogues are
below the descriptive per-turn 8,192 token proxy throughout, and table size is
strongly section-dependent. DBQR-RQ-v1 therefore targets conversational table-
program synthesis with compact schemas and a secure declarative JSON DSL. Typed
answers are produced by local execution. This is an explicit method adaptation;
it preserves objective correctness but changes the task from direct table reading.
No paid execution is authorized. See DBQR_RQ_PROTOCOL.md.

### DBQR restricted reference execution gate

The independently written allowlisted interpreter passed the pinned Practice
gate without model calls: 50/50 plans executed and 50/50 typed labels matched,
with 10/10 in each of the five sections. The report is
`results/dbqr_v1_audit/reference_plan_assessment.json`. This validates the
objective execution/scoring path only; it does not resolve the recorded
natural-language/program review flags and is not evidence of degradation,
early warning, or model quality. Ordinal answer words are canonicalized in two
predeclared equivalence classes because the corpus programs emit
`greater`/`lower` while labels may use `higher`/`more`/`less`. No paid run is
authorized. The subsequent compact-schema gate also passed all 50 Practice
turns: deterministic, maximum 1,455 rendered characters, no private table keys,
program targets or resolved text variables. The remaining call-free gates were
subsequently completed as recorded below.

### DBQR frozen budget and framing interpretation

The call-free budget/manifest/plot contract passes without authorizing
execution. It freezes GPT-4o-mini, Llama-4-Scout and Mistral-7B; two history
conditions; K=5; L=2; temperature 0.7; and seven trajectory panels. The single
Practice transport dialogue is selected by median compact-prompt character
load, before model outcomes. Its 420 calls include an 84-call review stop.
Training requires 8,400 calls and locked Test 6,300, for 14,700 calls and a
30,105,600 maximum-output-token ceiling across Training+Test.

The L variants retain the original question/schema verbatim and vary only a
semantically invariant instruction frame. RS is therefore prompt-framing
robustness, not independently authored paraphrase robustness. Label-blind
Practice/Training prompt manifests and the seven-panel render passed. The
causal oracle broker and resumable acquisition runner then passed a 420-record
call-free preflight, including the frozen 84-call review boundary. The locked
real runner also passed preflight with the frozen provider probes and real local
NLI/BERTScore scorers. This closes gates 1-7 without making an ASU model call;
the full-execution flags remain false. On 2026-10-01 the investigator separately
authorized only the frozen 84-call Practice review; the remaining 336 Practice
calls, Training and Test remain locked.

The review completed with 84/84 responses but 0/84 valid executable plans and
failed its frozen operational gate. The immutable call-free diagnostic found
62 missing-final-line errors, 14 invalid envelopes, seven reasoning-format
errors and one JSON decoding error. Although 61 responses mentioned the marker
and all 84 contained numbered reasoning, none mentioned `DBQR-PLAN-v1` or a
`steps` key. The model-facing prompt had omitted the required envelope,
reference syntax, operation allowlist and valid example. Therefore this result
is an operational prompt-contract failure, not evidence about reasoning-plan
quality or early warning. No further paid call is authorized. A corrected v1.1
prompt contract is prepared call-free and requires a new prospective gate.

The investigator then authorized only the six-call DBQR-RQ-v1.1 operational
contract probe: three models by two invariant instruction framings on one
Practice turn. Its frozen gate requires 2/2 valid locally executable plans per
model and zero truncations. This authorization does not extend to a fresh
84-call review, remaining Practice calls, Training or Test.

The six-call probe completed but failed. The original operational assessor
reported 1/2 executable plans for GPT-4o-mini, 2/2 for Llama-4-Scout and 0/2
for Mistral-7B. A stricter call-free assessment then re-executed every raw plan
and applied the frozen typed comparator: 0/6 produced the gold answer `2020`.
GPT's only scalar result was `440`; Llama's plan results could not satisfy the
scalar answer contract; Mistral used unsupported filter arguments. Thus better
format compliance did not establish correct plan synthesis. The single-shot
prompt-only DBQR path is terminated, the fresh 84-call review is not authorized,
and no further prompt-repair pilot should be run.

The subsequent strict JSON DSL and call-free fixture replay also passed. All
five Practice dialogues were replayed through `self_history` and
`oracle_answer_history`; 100/100 translated plans executed to the typed label,
branches remained isolated, and resume reused every per-turn checkpoint. No
model call occurred. This closes call-free protocol gates 3 and 5 but supplies
no model-quality or early-warning evidence. Budget/plot and stage-access gates
subsequently passed call-free; paid execution remains separately locked.
