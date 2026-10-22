# reasoning-monitor

Real-time multi-dimensional reasoning quality monitoring for LLMs under concept drift.

**TÜBİTAK 3501 project framework — v0.2.1**

---

## Overview

`reasoning-monitor` implements the RDMS (Reasoning Drift Monitoring System) proposed in the TÜBİTAK 3501 project. It measures six complementary dimensions of LLM reasoning quality over a sliding window and raises drift alarms when quality degrades.

Six dimensions (CQ, CS, RS, LS, ES, SS) are tracked simultaneously. Each feeds a statistical drift detector. Three detector assignment strategies are compared under hypothesis H2.

---

## Six Metrics

| Metric | Full Name | Distribution | Primary Detector |
|--------|-----------|-------------|-----------------|
| CQ | Correctness Quality | Bernoulli (raw) / Beta (windowed) | DDM (raw) + CUSUM (windowed) |
| CS | Consistency Score | Beta (working approx.) | CUSUM |
| RS | Robustness Score (C-subset) | Beta (working approx.) | CUSUM |
| LS | Logical Coherence Score | Beta (working approx.) | CUSUM |
| ES | Efficiency Score | Beta (harmonic mean) | CUSUM |
| SS | Stability Score | Beta (pairwise trace similarity) | CUSUM |

**Note:** "Beta" is a working approximation for bounded [0,1] signals. See `DEVIATIONS_FROM_PROPOSAL.md §8` for the distributional character framing. SS moved from window-mean dispersion (Normal/EWMA) to per-query pairwise trace similarity in v0.1.3 — see §6.

---

## Stream Scenarios

Two scenario catalogues are kept separate so archived results remain reproducible.

### Frozen E2H-GSM8K catalogue (v0.2.0)

Controlled difficulty-composition streams on a single dataset (GSM8K split into
Easy/Medium/Hard terciles). This isolates reasoning-quality drift from the
confound of switching datasets.

| ID | Composition | Role |
|----|-------------|------|
| ST1 | Mixed E/M/H, random | Stationary baseline (all difficulties) |
| ST2 | Easy → Medium → Hard | Controlled composition shift (MAIN) |
| ST3 | Easy only, random | Stationary Easy control |
| ST4 | Medium only, random | Stationary Medium control |
| ST5 | Hard only, random | Stationary Hard control |
| ST6 | Hard → Medium → Easy | Reverse / recovery test |

Within each regime, queries are drawn by **random permutation** of the
difficulty pool. Each stream is regenerated under **R independent seeds**; the
key analysis is the cross-seed reproducibility of the alarm order (e.g.
SS→CS→CQ), which tests whether some metrics give early warning before CQ.
The default experiment uses five seeds and writes Kendall's W, a permutation
p-value, and `SS < CQ` / `CS < CQ` rates to
`cross_seed_alarm_order.csv`. H2 replays the same saved metric streams through
all detector banks, so the strategy arms issue no additional model calls.

This catalogue remains available as `stream_design: e2h_gsm8k`, but it is frozen
after its cross-model CQ calibration gate failed. Existing E2H code and results
are retained and are not pooled with the proposal scenarios.

A preregistered Llama-only reopening then completed 720/720 K=5/L=3 calls with
healthy metrics, but the fixed solution-step CQ profile was
`0.900/0.733/0.833`: Hard exceeded Medium and the frozen monotonic gate failed.
The authorised call-free detector follow-up was therefore not run. This second
failure shows that the `1-2/3-4/5+` bands are not a reproducible difficulty
control across item samples; see Section 40 of `DEVIATIONS_FROM_PROPOSAL.md`.

**Change points:** staged streams (ST2, ST6) have two regime boundaries
(t=50, t=100); `spec.change_points` returns both, `effective_change_point()`
returns the first for backward compatibility; the experiment evaluator reports
both boundaries separately. Stationary streams (ST1, ST3–ST5) return -1.

### Accepted-proposal catalogue

`stream_design: proposal_multidataset` restores the accepted proposal's
multi-dataset scenarios without changing E2H. The first implemented scenario is
S1: a six-phase gradual composition shift from GSM8K through ARC-Challenge to
MATH. Its exact timestamp-to-dataset plan is written to `stream_S1_seed*.csv`.
Live scientific configs set `allow_synthetic_datasets: false`, so a dataset
load failure stops before any model call.

The completed K=1/L=0 S1 pilot is retained under
`results/proposal_s1_pilot_gpt_llama`. Its call-free temporal-order controls
reuse the exact saved responses. All stationary dataset arms were clean, but
the composition-matched global shuffle raised CQ-CUSUM alarms for both models,
so the preregistered control gate failed and a paid K=5/L=3 run is not yet
authorised. See `order_controls/CONTROL_ASSESSMENT.md` in that result directory.
The subsequent preregistered 1,000-permutation analysis found a significant
ordered CQ decline for both models (`p=0.001` each), while the current W=10
CUSUM alarmed in 99.1% and 99.5% of null permutations. This supports the S1
trajectory but rejects the present detector calibration.
A subsequent preregistered split-sample calibration selected the smallest
common threshold, 7.5. On 1,000 fresh validation permutations per model, false
alarm rates were 1.5% and 3.3%, and ordered S1 alarmed at t=80 and t=82. The
gate passed, but detections were 35 and 37 queries after the declared midpoint;
the result authorises protocol planning rather than starting paid calls.
The paid K=5/L=3 metric-health protocol in
`configs/proposal_s1_kl_validation_gpt_llama_asu.yaml` completed all 1,440
calls. All metric-health checks passed, but the frozen prospective CQ-delay
gate failed: delays were +37 for GPT-4o-mini and +34 for Llama-3-8B against the
maximum +10 criterion. Non-CQ alarm ordering remains exploratory; see Sections
37-38 of `DEVIATIONS_FROM_PROPOSAL.md` and the frozen assessment under
`results/proposal_s1_kl_validation_gpt_llama/`.
The subsequent preregistered, call-free proposal-mapped detector development
also produced no candidate: CS/LS/SS controlled held-out false alarms but did
not alarm on either ordered stream, while RS/ES could not meet the common
calibration specificity bound. This is a negative development result, not a
reason to retune on the observed S1 order.

A subsequent call-free S1 answer-observability audit found that this negative
result is not a clean test of the scientific hypothesis. Of 150 MATH K-runs per
model, 94 GPT-4o-mini and 72 Llama-3-8B outputs reached the exact 512-token
ceiling; 94 and 71 respectively had no defensible final answer. In the primary
K run, only 12/30 GPT and 14/30 Llama MATH answers were strictly observable,
giving CQ bounds of [0.267,0.867] and [0.233,0.767]. Boxed ARC choices embedded
in Llama prose also exposed answer-extraction errors and changed CS on 9/29
complete ARC queries. Historical S1 results remain immutable, but a new S1
test must use a typed answer-first contract and a larger output ceiling before
the early-warning hypothesis is judged. See
`results/proposal_s1_parsing_audit/SECTION55_S1_PARSING_AUDIT.md`.

Section 56 freezes the minimal corrective pilot before any new response is
generated. It reruns the same 90-question S1 sequence for GPT-4o-mini and
Llama-3-8B with K=1/L=0, typed first-line answers, at most eight short numbered
reasoning steps and a 1024-token ceiling. The call-free preflight confirms 30
questions per dataset, real cached benchmark sources and an exact question/order
match to the historical S1 stream (sequence SHA-256 `791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`).
Unused rqeval scorers are disabled because the frozen assessor recomputes CQ
from raw first-line answers. The run ceiling is 180 calls / 184,320 output tokens. Both model gates require
complete execution and parser observability, zero truncations, a GSM8K-MATH CQ
gap of at least 0.15 and one-sided ordered-trend p<=0.05. The paid run remains
on hold for explicit execution; a pass would permit planning, not starting, a
separate K=5/L=3 metric-health run.

The completed Section 56 run passed execution and parsing but failed its
scientific gate. Both models had 100% first-line marker/parser agreement; GPT
had zero truncations and Llama had seven. Nevertheless, GSM8K CQ collapsed to
0.20 for both models, producing negative GSM8K-MATH gaps. A call-free diagnosis
found direct prompt-order interference: GPT gave a wrong first-line answer but
later derived the correct GSM8K number on 14/30 questions. Historical
answer-last GSM8K CQ on the same items was 28/30 for GPT and 24/30 for Llama,
versus 6/30 each under answer-first. The intervention is therefore frozen as a
failed operational design; it is not evidence against S1 and does not authorise
K=5/L=3 expansion.

Section 57 freezes the smallest direct test of that prompt-order diagnosis.
For each of GPT-4o-mini and Llama-3-8B it uses the first frozen 10 GSM8K and
10 MATH items from the Section 56 sequence, with K=1/L=0. Models may produce
at most six short numbered reasoning steps and must put exactly one
`FINAL_ANSWER: <answer>` declaration on the final line. The 2048-token ceiling
tests trace completion without reintroducing answer-first interference. The
20-item sequence SHA-256 is
`99f3b47eb250b15defee6086a76adb2c777e824ff3308f35aa4ebc463d024e20`.
The call-free preflight passed with real cached benchmark sources. The frozen
budget is 40 calls / 81,920 maximum output tokens. Each model requires complete
execution, 100% final-line observability and parser agreement, zero truncations,
and GSM8K CQ at least 0.60; MATH CQ is descriptive. The paid calibration remains
on hold for explicit execution and cannot itself establish early warning.

The completed Section 57 calibration passed both frozen model gates. All 40
calls succeeded, every response ended in one strictly parseable typed answer,
stored answers agreed with the strict parser, and no completion was truncated.
GPT-4o-mini CQ was 1.00 on GSM8K and 0.70 on MATH; Llama-3-8B CQ was 0.80 and
0.40. On these same ten GSM8K items, historical answer-last, Section 56
answer-first and Section 57 bounded-answer-last correctness was respectively
10/10, 2/10 and 10/10 for GPT, and 8/10, 1/10 and 8/10 for Llama. The result
supports the prompt-order diagnosis and authorises planning only. Before the
K=5/L=3 metric run, the bounded answer-last contract should be validated on the
complete 90-question S1 stream at K=1/L=0, including ARC-Challenge.

Section 58 preregisters that complete validation before any new target-model
response is generated. It uses the exact historical 90-question S1 sequence,
GPT-4o-mini and Llama-3-8B, K=1/L=0, temperature 0.7 and a 2048-token ceiling.
GSM8K and MATH require a final-line `FINAL_ANSWER`; ARC-Challenge requires a
final-line `FINAL_CHOICE`, always after at most six short reasoning steps. The
call-free real-data preflight passed with 30 questions per dataset and sequence
SHA-256 `791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`.
The frozen ceiling is 180 calls / 368,640 output tokens. For each model the
conjunctive gate requires complete execution, 100% strict final-line
observability and stored/parser agreement, zero truncations, GSM8K-MATH CQ gap
at least 0.15 and a one-sided ordered-trend p-value no greater than 0.05. The
paid run remains on hold; a pass would permit preregistration, not automatic
execution, of a separate K=5/L=3 metric-health campaign.

The completed Section 58 validation failed its frozen conjunctive gate despite
clear CQ separation. GPT-4o-mini CQ was 0.967/0.900/0.567 for
GSM8K/ARC-Challenge/MATH (gap 0.400, one-sided trend p=0.0000378). One
untruncated MATH response embedded the correct `5/13` declaration inside a
LaTeX `\text{...}` block rather than emitting the required standalone line, so
strict observability was 89/90. After a call-free outer-LaTeX-delimiter parser
correction, Llama CQ is 0.833/0.867/0.300 (gap 0.533,
p=0.00000589); three MATH responses entered repetitive loops, reached the
2048-token ceiling and never emitted a final marker, giving 87/90 strict
observability. All other calls succeeded and the stream hash matched.

The CQ conclusion is robust to missing outcomes: recovering GPT's visibly
correct answer gives MATH CQ 0.600, gap 0.367 and p=0.000102; even counting all
three unresolved Llama answers as correct gives MATH CQ 0.400, gap 0.433 and
p=0.000125. These are post-gate sensitivity results, not a relaxed pass.
Section 58 remains an operational failure and does not authorise K=5/L=3 or
selective reruns.

Section 59 preregisters a small post-Section-58 development stress test. It is
not a selective confirmatory rerun and cannot change Section 58's failed
decision. The frozen ten-question MATH set contains all four Section 58
format/truncation failures plus the first six remaining MATH questions in the
historical S1 order. Both models use K=1/L=0 and the same 2048-token ceiling,
but the prompt now permits no more than six numbered lines, one short sentence
per line, requires an immediate best answer after line six or sooner, and
requires the final marker as plain text outside LaTeX and code blocks.

The call-free preflight reconstructed the four failures from raw Section 58
records, verified the deterministic controls and reproduced selected-sequence
SHA-256 `1a71ba588cd589bb0a07a1801d9a16c518c31e6066d44a0371e439e789213a37`.
The frozen ceiling is 20 calls / 40,960 output tokens. For each model the gate
requires all ten calls to succeed, strict final-line observability and parser
agreement of 100%, zero truncations and zero six-line-contract violations.
Accuracy is descriptive because the sample is outcome-enriched. A pass permits
planning a new homogeneous validation but does not alter Section 58.

The completed Section 59 stress pilot failed its two-model gate. GPT-4o-mini
passed every operational check: 10/10 calls succeeded, all answers used a
strict plain final line, no trace reached the token ceiling and no output
exceeded six numbered reasoning lines. Llama-3-8B had only 8/10 strict final
markers, one truncation and two line-budget violations. Both failures recurred
on Section 58 failure items. `math-intermediate_algebra-00000` again entered a
repetitive loop and reached 2048 tokens without an answer marker;
`math-intermediate_algebra-00026` placed an incorrect answer on numbered line 7
rather than on a plain final line. Failure/control CQ is descriptive because
the set is outcome-enriched.

The stricter prompt therefore repairs GPT formatting but does not make Llama
operationally reliable for K=5/L=3. Section 59 remains a failed overall pilot;
no further Llama prompt retry is authorised. A future GPT-only metric-health
campaign must be separately preregistered and should retain the full-stream
validated Section 58 prompt while adding a call-free parser rule for GPT's one
recoverable LaTeX marker form.

Section 60 preregisters a post-hoc Llama-4 Scout model-substitution feasibility
arm. It does not replace Llama-3 or alter Sections 58-59. The saved ASU gateway
probe accepted provider model `aws/llama4_scout-17b` with history, search and
prompt-context enhancement disabled. The arm changes only the model: it reuses
Section 59's exact ten MATH QIDs, strict six-line/plain-marker prompt,
temperature 0.7, K=1/L=0 and 2048-token ceiling. The call-free preflight
verified the probe and selected-sequence SHA-256
`1a71ba588cd589bb0a07a1801d9a16c518c31e6066d44a0371e439e789213a37`.
The frozen ceiling is 10 calls / 20,480 output tokens. Its gate requires 100%
execution, strict marker/parser agreement and line-budget compliance with zero
truncation. CQ is descriptive. A pass permits planning only a full 90-question
Llama-4 Scout K=1/L=0 validation, not K=5/L=3.

The completed Section 60 arm passed every frozen operational check. All 10/10
calls were successful and unique, strict final-line observability and
stored/parser agreement were 100%, and there were no truncations or six-line
contract violations. A post-run call-free audit found that the MATH normalizer
misread two correct dollar-delimited fractions as their denominators. The
normalizer now removes only a complete outer LaTeX delimiter pair before its
existing comparison. Raw responses and hashes are unchanged. Corrected
descriptive CQ is 3/4 on prior Section 58 failures and 5/6 on controls (8/10
overall). The pass supports preregistering a homogeneous 90-question Llama-4
Scout K=1/L=0 validation; it is not evidence for K=5/L=3 readiness.

Section 61 preregisters that homogeneous validation before any new target
response is generated. It uses the exact historical S1 sequence of 30 GSM8K,
30 ARC-Challenge and 30 MATH questions (sequence SHA-256
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`).
The sole model is `aws/llama4_scout-17b`; temperature is 0.7, K=1/L=0, W=10,
and the output ceiling is 2048 tokens. The Section 60 strict contract is kept:
at most six numbered one-sentence reasoning lines, an immediate best answer,
and one plain final `FINAL_ANSWER` or `FINAL_CHOICE` line outside LaTeX and
code blocks. The call-free preflight verified all 90 real-data QIDs in order,
the accepted gateway probe and the provider model. The frozen ceiling is 90
calls / 184,320 output tokens.

The Section 61 gate requires 100% execution, strict final-line observability
and stored/parser agreement, zero truncations, zero six-line violations,
GSM8K-MATH CQ gap at least 0.15 and one-sided ordered-trend p no greater than
0.05. It remains on hold for explicit execution. A pass permits only
preregistration of a separate Llama-4 K=5/L=3 metric-health pilot.

The completed Section 61 validation passed every frozen check. All 90 calls
were successful and unique in the historical order; final-line observability
and stored/parser agreement were 100%; and no response was truncated, used a
code fence or violated the six-line contract. Llama-4 Scout CQ was
0.967/0.900/0.567 for GSM8K/ARC-Challenge/MATH, giving a 0.400 gap and a
one-sided ordered-trend p-value of 0.0000378. The raw record SHA-256 is
`35e2e561f702e305f810cd5486825870634ba3af2ba9b1147efb61aecfa8860b`.
Although the aggregate correct counts equal Section 58 GPT-4o-mini, a call-free
paired audit found zero identical raw completions and 12 item-level correctness
disagreements split evenly between the models. The equality is genuine, not a
cache or record-routing error. Section 61 therefore authorises preregistration,
not automatic execution, of a Llama-4 K=5/L=3 metric-health pilot.

Section 62 returns to the third model named in the accepted proposal,
Mistral-7B. Because the planned local RTX 4090 is unavailable, the model is
frozen to the already probed ASU endpoint `aws/mistral-7b`; this infrastructure
change does not alter the question stream or response protocol. The validation
uses the exact historical 30 GSM8K, 30 ARC-Challenge and 30 MATH sequence
(SHA-256
`791e6090518090213559432c62ad2d6fe7a30588311f3a9cf816276f63a78cca`),
temperature 0.7, K=1/L=0, W=10, the Section 61 strict six-line/plain-marker
contract and a 2048-token ceiling. Resume is enabled. The call ceiling is 90
and the output-token ceiling is 184,320.

The call-free real-data preflight passed before target responses were
generated. It reproduced all 90 historical QIDs and the frozen sequence hash,
and verified the exact provider model plus accepted `disable_all` gateway
probe.

The Section 62 gate requires 100% execution, strict final-marker observability
and stored/parser agreement, zero truncations and six-line violations, a
GSM8K-MATH CQ gap of at least 0.15 and one-sided ordered-trend p no greater
than 0.05. This K=1 run cannot resolve the immutable Section 47 finding that
ASU Mistral produced zero K=5 verbatim repeat diversity. A Section 62 pass
therefore permits only preregistration of a small ASU K=5 repeat-diversity
capability check, not a full K=5/L=3 metric-health campaign.

The completed Section 62 run failed its frozen operational gate. Execution was
90/90 with the exact stream hash and no truncations. Under the strict contract,
however, only 61/90 final answers were observable and stored/parser agreement
was also 61/90; three outputs violated the six-line step contract. Strict CQ
was 0.367/0.200/0.100 for GSM8K/ARC-Challenge/MATH, with gap 0.267 and
one-sided trend p=0.00649. Because format failure was differential by dataset,
these strict CQ values do not support valid inference despite passing the two
numerical dynamic-range checks.

Section 63 records a call-free sensitivity analysis of the immutable Section
62 file. All 29 strict failures still contained explicit typed answer markers:
11 were embedded in numbered steps, 16 ARC markers added text after the choice
letter, and two markers were not the strict final line. A deterministic
explicit-marker parser therefore observes all 90 answers and gives CQ
0.400/0.833/0.100, gap 0.300 and one-sided trend p=0.00969. These cell means
are not monotonic because ARC-Challenge exceeds GSM8K. The sensitivity result
diagnoses the format confound but cannot retroactively pass Section 62; neither
the ASU K=5 repeat-diversity check nor K=5/L=3 is authorised.

Section 64 preregisters a small, disjoint K=5/L=3 metric-health pilot for
GPT-4o-mini and Llama-4 Scout. Mistral is not included because its prior ASU
K=5 diversity was zero and Section 62 failed. The selection uses seed-42
per-dataset ranks 31-35, interleaved GSM8K/ARC-Challenge/MATH, so its 15 QIDs
do not overlap the historical 90-question S1 stream. The selection SHA-256 is
`9ef33e895ac902ea42383e30f4812231c76297635f468c8c91a1fb4d5333d1a5`.
With two models and eight calls per question, the frozen ceiling is 240 calls
and 491,520 output tokens. The strict six-line answer-last contract, 2048-token
ceiling, temperature 0.7 and resumable checkpoints are retained.

The call-free preflight passed before target responses were generated. Both
ASU provider probes are accepted, the real NLI and BERTScore scorers report
ready without fallback, and all 15 real-data QIDs are disjoint and in the
frozen order. Each model must complete 120 records with 100% strict marker and
stored/parser agreement, zero truncations and step violations, K-repeat
diversity at least 0.80, RS coverage both overall and in every dataset,
complete finite metric coverage and at least two distinct values for each of
the six metrics. This is a development metric-health gate, not early-warning
evidence. A pass permits only preregistration of a separate full 90-question
K=5/L=3 campaign.

The completed Section 64 pilot failed its frozen conjunctive gate despite
clean execution. Both models returned all 120 records with perfect strict
marker/parser agreement, zero truncations and step violations, complete finite
metric coverage, and genuine NLI/BERTScore scoring. GPT-4o-mini achieved K
raw-completion diversity on 15/15 questions; Llama-4 Scout achieved only 9/15
(0.600), below the frozen 0.80 threshold. RS was evaluable on 12 GPT and 11
Llama-4 questions but equalled 1.0 in every case, leaving RS degenerate for
both models. The full 90-question campaign is not authorised.

Section 65 is a call-free diagnostic of those immutable records. All 90 L=3
prompts retained the complete original question and output instructions
verbatim, adding only one of three fixed prefixes or suffixes. The current
perturbation layer therefore measures wrapper sensitivity rather than genuine
meaning-preserving paraphrase robustness. RS evaluability exactly matched the
rqeval correctness subset, so the constant value is not a parsing failure.
GPT had answer-level K diversity on only 5/15 questions despite raw-text
diversity on all 15. Llama-4 had answer diversity on 3/15, and six of its K=5
raw-completion groups were fully identical. Repeating Section 64 cannot repair
these failures; a new protocol must first freeze substantive, semantically
checked paraphrases and separately resolve Llama-4 sampling diversity.

Section 66 implements those repairs as two independent development arms. The
RS arm freezes 45 question-body rewrites over the same 15 QIDs, preserves the
answer choices and strict output contract, and uses K=1/L=3 so it cannot be
mistaken for a K-diversity retest. Its call-free preflight passes; the ceiling
is 120 calls across GPT-4o-mini and Llama-4 Scout. The gate requires complete
format observability, RS coverage in all three datasets and at least two RS
values per model. The paid arm remains on hold, and the bank still requires
independent human review before confirmatory use.

The Llama-4 repair is a separate five-call capability probe. ASU accepted all
five per-call `seed` values on the frozen formerly raw-identical QID, but every
raw completion and extracted answer was identical. The capability gate
therefore failed, and no larger seeded K=5 pilot is authorised. Seed
forwarding remains opt-in and does not alter existing configurations. The
independent RS arm remains on hold; neither Section 66 arm authorises the full
90-question campaign.

The Section 66 genuine-paraphrase arm subsequently completed all 120 calls and
formally passed its frozen operational and RS-nondegeneracy gate. Both models
had complete records, perfect marker/parser agreement, zero truncations or
step violations, and raw RS values {2/3, 1}. A call-free Section 67 semantic
audit found that GPT-4o-mini's only disagreement was `95 minutes` versus the
equivalent `5700 seconds`; its unit-normalised RS is therefore 1.0 everywhere.
Llama-4 Scout retained a genuine `384` versus `480` arithmetic failure and
thus real RS variation. The formal Section 66 result remains immutable, but
the conjunctive semantic interpretation fails. No fresh holdout or full
campaign is authorised until semantic answer equivalence is prospectively
frozen.

Section 68 prospectively freezes the opt-in `semantic-v1` answer policy. It
leaves all historical configurations unchanged. For GSM8K it converts only
explicit units within the same frozen dimension and refuses missing-unit,
cross-dimension and currency inferences; ARC uses exact choice identity; MATH
uses symbolic scalar or ordered-sequence equivalence. CS uses the policy for
pairwise repeat agreement, while RS preserves the rqeval C-subset and permits
gold correctness or semantic equivalence to the accepted primary answer. The
call-free preflight passes all 20 frozen positive and negative cases. This
permits preregistration of a fresh disjoint development holdout only, not the
full campaign.

Section 69 closes the experimental chain with a call-free, hash-verified
evidence synthesis:

```cmd
python scripts\synthesize_final_evidence.py
```

The synthesis verifies 24 frozen Section 47-68 source files and reproduces 19
cross-section consistency checks. The central claim that non-CQ metrics
reliably precede CQ is not supported: in the only track eligible for frozen
detector timing, CQ detected both transitions and no non-CQ metric led it in
both intervals; the structure-aware LS/SS reparse did not change the decision.
CQ difficulty sensitivity is supported, and the framework's ability to expose
prompt, truncation, parsing, perturbation, semantic-equivalence and provider-
sampling failures is a positive methodological result. CS/SS cross-model
generalisability remains not evaluable on the current ASU endpoints, while RS
evidence is mixed and inconclusive.

No additional paid pilot or full K=5/L=3 campaign is authorised. Section 70
may be opened only if a newly demonstrated software defect invalidates frozen
evidence used by the synthesis. A future confirmatory study is a new protocol,
not a continuation of this numbered chain. The final report is under
`results/final_evidence_synthesis/FINAL_EVIDENCE_SYNTHESIS.md`.

### Independent RAT-v1 successor protocol

RAT-v1 is a separate prospective study, not Section 70. It replaces unrelated
difficulty bins with 60 exact-solver problem families, each observed at six
nested structural levels. Entire families are assigned to calibration,
development and untouched holdout splits (24/18/18 families). Metrics at level
`t` forecast the first CQ failure at level `t+1` only while the trajectory is
still correct. The confirmatory comparison is a risk-adjusted baseline
(`level + task template + model`) versus the same model plus RS, LS and ES.

The deterministic 360-item bank and frozen protocol pass every call-free
structural check. Their SHA-256 values are respectively
`8404ed2743e2bdb61b004a6cb1f88fb2acfec60e3305270d183d857225016c31` and
`2e79ad402c4918632a53e3d76a25cada3b2323791978cddb711ceac045fb906f`.
This pass authorises preparation of a calibration-only configuration; it does
not authorise model calls or opening development/holdout outcomes.

The frozen ASU calibration configuration now also passes its call-free
operational preflight. It sees only 24 calibration families (144 items/model),
uses K=1/L=3 and preregisters GPT-4o-mini, Llama-4 Scout and Mistral-7B. The
ceiling is 576 calls/model, 1,728 calls total and 1,327,104 maximum output
tokens. Its SHA-256 is
`3b6ffe15e12056d8ddd0878cad5756b6fa96dd9815e316a38285cdf09ce98345`.
Passing this gate still does not authorise execution.

The analysis is frozen before any calibration output. The executable path
builds a first-failure risk set, tunes ridge logistic hazard models with
family-grouped folds, and evaluates a locked holdout with 2,000 family-cluster
bootstrap samples and Holm correction. The post-run calibration gate requires
usable first-failure and censored-family counts, event coverage across task
templates, finite nondegenerate RS/LS/ES and complete operational records in
at least two models. Synthetic tests recover a planted prospective signal;
the structural preflight verifies hashes of the analysis code and gate.

The completed calibration did not pass that frozen gate. GPT-4o-mini and
Llama-4 Scout failed only the eight-line output contract; Mistral-7B also
failed answer observability and event-density requirements. A separately
frozen call-free signal diagnostic on the GPT/Llama records then found combined
nested-CV log-loss improvement of -0.0215 (1/4 positive folds). GPT alone was
positive (+0.0341; 3/4), but Llama was negative (-0.0248; 2/4), so the
two-model continuation screen failed. Development and holdout calls remain
unauthorised.

The final finite ablation of all seven non-empty RS/LS/ES subsets also found no
cross-model candidate. GPT-4o-mini favored RS (+0.0738; 4/4 positive folds),
whereas Llama-4 Scout favored LS (+0.0145; 3/4); neither signal transferred to
the other model. RAT-v1 therefore stops at calibration, with development and
holdout still untouched.

```cmd
python scripts\generate_risk_adjusted_trajectory_bank.py
python scripts\preflight_risk_adjusted_trajectory.py
python scripts\preflight_risk_adjusted_trajectory_calibration.py
python scripts\estimate_campaign.py configs\risk_adjusted_trajectory_v1_calibration_asu.yaml
# After an explicitly approved and completed calibration run:
python scripts\assess_risk_adjusted_trajectory_calibration.py
python scripts\diagnose_risk_adjusted_trajectory_signal.py
python scripts\diagnose_risk_adjusted_trajectory_metrics.py
```

The full design is in `RISK_ADJUSTED_TRAJECTORY_PROTOCOL.md` and
`configs/risk_adjusted_trajectory_v1.yaml`.

### Independent CDR-v1 dialogue protocol

CDR-v1 is a new prospective protocol, separate from both the numbered Section
chain and RAT-v1. It replaces independent questions with 12-turn exact-scored
dialogues whose facts and intermediate results accumulate within one case.
Three matched conditions separate the total context effect (`self_history`),
error propagation (`oracle_history` versus `self_history`), and long-history
load (`state_summary` versus `oracle_history`).

The frozen feasibility pilot uses three calibration dialogues, GPT-4o-mini and
Llama-4 Scout, K=1/L=0, and 216 resumable calls. It first asks whether a real
context-linked CQ decline exists; RS/LS/ES are deferred until that prerequisite
passes. Development and holdout remain closed, and paid calls require explicit
approval. See `CONTROLLED_DIALOGUE_PROTOCOL.md` for the design and commands.

The completed CDR-v1 pilot is integration-invalid. Although locally recorded
message arrays contained the full history, both models were correct only at
turn 1 in both native-history arms and failed every later turn, while the
two-message state-summary arm was perfect. A frozen call-free audit therefore
forbids scientific interpretation and identifies a shared history-transport
failure signature. CDR-v1.1 packs the caller-managed transcript into one
system context while retaining `enable_history: false`. A 12-call turn-2 probe
must pass before any fresh pilot is prepared. See
`CONTROLLED_DIALOGUE_TRANSPORT_PROTOCOL.md`.

The transport probe produced 11/12 valid cells. Its only failure reached the
256-token ceiling after correctly recovering and updating the prior state. A
separately frozen exact-message retry at 512 tokens passed with the correct
answer and marker. The fresh CDR-v1.1 packed-history pilot now passes call-free
preflight: 216 resumable calls, two models, three dialogues and three matched
conditions. Execution still requires explicit approval.

That fresh run completed operationally cleanly but failed its preregistered
oracle-history solvability requirement. A preserved call-free diagnostic still
found an exploratory cross-model state-summary advantage over real self
history: 12/33 paired GPT cells and 6/33 paired Llama cells, with no reverse
discordance. Both models shared a persistent budget failure beginning at turn
7. Because the short oracle responses were not information-format matched to
the verbose self responses, the next design replaces that control with
`user_only_history`. The current evidence is promising calibration evidence,
not yet a scientific claim.

CDR-v1.2 now has a frozen, call-free-validated mechanism probe for that
replacement. It adds only the `user_only_history` arm to the same three
calibration dialogues: all earlier user facts are packed explicitly, while
all earlier assistant responses are omitted. The 72 new resumable calls are
compared with 144 frozen CDR-v1.1 self-history and state-summary cells. The
analysis and source hashes were fixed before execution. Passing this probe
would support only preparation of a larger calibration validation, not a
context-degradation, error-propagation, or early-warning claim.

The completed CDR-v1.2 arm retains a failed formal gate because 3 GPT and 7
Llama responses ended exactly at the 512-token ceiling, accounting for every
missing final marker. A frozen call-free sensitivity diagnostic shows that the
GPT state-summary advantage is robust even if all three censored answers were
correct (minimum gap 0.182; p=0.015625), while the Llama result is not (minimum
gap 0.061; p=0.25). The exact-message retry of only those 10 censored cells is
now frozen and passes call-free preflight. It retains the original messages
and call seeds, raises only the output ceiling to 1,024 tokens, and permits
resumable execution. Its 10 paid calls still require explicit approval.

The CDR-v1.2a retry completed cleanly. Capacity-adjusted `user_only_history`
accuracy is 0.818 for GPT and 0.848 for Llama, versus 1.0 under verified state
summary. The paired state-summary advantages are 0.182 (6-0, p=0.015625) and
0.152 (5-0, p=0.03125), respectively, so both models meet the frozen mechanism
thresholds. Prior assistant prose has an additional adverse association for
GPT (user-only versus self-history +0.182, p=0.035156) but not Llama (+0.030,
p=0.5). Thus the cross-model primary phenomenon is raw-history integration
load and its mitigation by structured state, not general error propagation.
The original failed gate remains unchanged, and three dialogue clusters are
insufficient for a claim; the next step is call-free preparation of a
nine-dialogue calibration validation.

CDR-v1.3 is the prospective primary validation. It uses all nine unused
calibration dialogues and only the cross-model primary contrast,
`state_summary` versus `user_only_history`, for 432 resumable calls. The
conditional 216-call self-history arm is deferred until the primary result is
known. Statistical inference treats each dialogue as one independent cluster:
the primary test uses equal-weight dialogue accuracy differences and an exact
one-sided sign-flip test. A separately frozen early-versus-late interaction
tests whether the summary advantage grows from turns 2-6 to turns 7-12. The
complete call-free preflight passes; paid execution still requires explicit
approval.

The completed CDR-v1.3 run retains a failed formal gate. GPT's state-summary
accuracy was 94/99 (0.949495), one correct answer short of the frozen 0.95
minimum; all five misses were fully observed arithmetic errors rather than
parser or capacity failures. Nevertheless, GPT showed a dialogue-level
history-load gap of 0.283 (exact sign-flip p=0.005859; 8/9 positive) and a
late-minus-early interaction of 0.193 (p=0.023438; 7/9 positive). Llama showed
a gap of 0.182 (p=0.003906; 8/9) and interaction 0.211 (p=0.03125; 6/9), but one
user-only response reached the 1,024-token ceiling. A hash-frozen call-free
sensitivity analysis leaves Llama's primary gap at 0.172 (p=0.003906) even if
that censored answer is treated as correct. Its interaction p then becomes
0.054688, while GPT still meets the frozen longitudinal requirement. These are
strong calibration results, not a relabelled pass or a scientific claim; the
deferred self-history arm remains closed.

CDR-v1.4 is now prepared as a fresh prospective replication without opening
the existing development or holdout splits. A new deterministic seed produces
nine separately namespaced calibration dialogues. The frozen comparison still
uses `state_summary` versus `user_only_history`, both models, dialogue-cluster
sign-flip inference, and the early-versus-late interaction. The 432-call runner
is resumable, uses a 2,048-token ceiling plus an eight-line response constraint,
and prints completed/total, percentage, calls per minute, and ETA after every
saved call. Its call-free preflight passes. The absolute state-summary
assay-validity floor is frozen prospectively at 0.90; the primary effect
thresholds are unchanged. This does not revise the failed CDR-v1.3 decision,
and paid CDR-v1.4 execution still requires explicit approval.

The completed CDR-v1.4 replication and its call-free warning diagnostic are
recorded in DEVIATIONS_FROM_PROPOSAL.md. The original failed gate is preserved;
GPT's three format failures are correct baseline answers with trailing periods.
Both models have perfect post-baseline marker compliance. Simple token and
latency predictors do not show stable cross-template value beyond turn and
context length. This is exploratory calibration evidence, not a validated
early-warning system. Reproduce the audit and grouped prediction analysis with:

```cmd
python scripts\analyse_dialogue_warning.py
```

The command makes no model calls and writes a JSON report and per-turn
out-of-group predictions under
`results/controlled_dialogue_v1_4_warning_diagnostic`.

Content diagnostics and temporal PNG/PDF figures can also be reproduced
without model calls:

```cmd
python scripts\analyse_dialogue_content.py
```

Outputs are under `results/controlled_dialogue_v1_4_content_diagnostic`.
Numeric equality and state-update checks improve some next-turn predictions
but do not directly precede first errors in this sample. These are exploratory
monitoring signals, not validated early-warning metrics.

## Detector Assignment Strategies (H2 ablation)

Three named strategies, all configurable via `detector_strategy` in `MonitorConfig` or `default.yaml`:

| Strategy | Description |
|----------|-------------|
| `distribution_based` | Theory-derived mapping; tested as hypothesis, not established result |
| `proposal_mapping` | As stated in project proposal: CQ→CUSUM, CS→ADWIN, RS→DDM, LS→PH, ES→EWMA, SS→ADWIN |
| `adwin_all` | ADWIN applied to all metrics — H2 baseline |

H2 ablation (`run_strategy_comparison=True`) compares all three automatically.

---

## Datasets

| Dataset | Domain | Items | Gold format |
|---------|--------|-------|-------------|
| GSM8K | Math word problems | 300 | Last number |
| ARC-Challenge | Science exam | 300 | Letter A-D |
| StrategyQA | Commonsense reasoning | 300 | yes / no |
| MATH | Competition mathematics | 300 | `\boxed{}` expression |

---

## Quick Start

```bash
pip install -e ".[nli]"   # include DeBERTa-v3-small for LS scorer

# Run the full experiment with default config
python -m reasoning_monitor.cli --config configs/default.yaml

# Quick smoke test (simulated models, no API keys needed)
python -m reasoning_monitor.cli --quick

# Inspect the real-model pilot budget, then run it (Windows Miniconda Prompt)
python scripts\estimate_campaign.py configs\pilot_asu.yaml
python -m reasoning_monitor.cli --config configs\pilot_asu.yaml

# Watch an already-frozen checkpointed run in a second Miniconda Prompt
python scripts\watch_jsonl_progress.py ^
  --target "GPT=results\controlled_dialogue_v1_3_primary_validation\data\records_CDR13_gpt4o_mini.jsonl,216" ^
  --target "Llama=results\controlled_dialogue_v1_3_primary_validation\data\records_CDR13_llama4_scout.jsonl,216" ^
  --interval 5

# CDR-v1.4 preparation checks only; neither command makes model calls
python scripts\estimate_dialogue_replication.py
python scripts\preflight_dialogue_replication.py

# Proposal S1 CQ pilot (GPT-4o-mini + Llama-3-8B; 180 resumable calls)
python scripts\estimate_campaign.py configs\proposal_s1_pilot_gpt_llama_asu.yaml
python -m reasoning_monitor.cli --config configs\proposal_s1_pilot_gpt_llama_asu.yaml

# Call-free replay controls for the completed S1 pilot (no API key required)
python scripts\analyze_s1_controls.py results\proposal_s1_pilot_gpt_llama

# Preregistered 1,000-permutation CQ/CUSUM reassessment (also call-free)
python scripts\analyze_s1_permutations.py results\proposal_s1_pilot_gpt_llama

# Split-sample threshold calibration and locked validation (also call-free)
python scripts\calibrate_s1_cusum.py results\proposal_s1_pilot_gpt_llama

# Completed paid K/L run (commands retained for reproducibility/resume)
python scripts\estimate_campaign.py configs\proposal_s1_kl_validation_gpt_llama_asu.yaml
python -m reasoning_monitor.cli --config configs\proposal_s1_kl_validation_gpt_llama_asu.yaml

# Apply the frozen Section 37 gate after the run completes (no model calls)
python scripts\assess_s1_kl_validation.py results\proposal_s1_kl_validation_gpt_llama

# Preregistered proposal-mapped detector development (saved calls only)
python scripts\develop_s1_detectors.py results\proposal_s1_kl_validation_gpt_llama

# Reopened controlled E2H Llama K/L pilot (720 resumable calls)
python scripts\estimate_campaign.py configs\e2h_stepband_llama_kl_pilot_asu.yaml
python -m reasoning_monitor.cli --config configs\e2h_stepband_llama_kl_pilot_asu.yaml
python scripts\assess_e2h_llama_kl_pilot.py results\e2h_stepband_llama_kl_pilot

# Run tests
pytest tests/ -v   # rqeval parity tests skip when rqeval is unavailable
```

---

## Key Config Parameters

```yaml
# configs/default.yaml (excerpt)
k_repeats: 5           # CS/SS: K=5 repeated calls per query (per proposal)
l_paraphrases: 3       # RS: L=3 paraphrases per query
window_sizes: [20]     # W=20 sliding window

detector_strategy: distribution_based   # or proposal_mapping / adwin_all

coherence_scorer: lexical   # use "nli" for publication experiments
stability_scorer: auto      # use "bertscore" for publication experiments
publication_mode: false     # true = refuse silent scorer fallbacks (see below)
```

**API (programmatic):**

```python
from reasoning_monitor.experiment import ExperimentConfig, run_experiment

manifest = run_experiment(ExperimentConfig(
    detector_strategy="distribution_based",
    run_strategy_comparison=True,   # runs all three strategies for H2
))
```

---

## CQ Two-Level Detection

CQ is monitored at two levels:
- **Windowed** (`level="drift"`): CUSUM on the W=20 window mean (Beta-distributed)
- **Raw** (`level="drift_raw"`): DDM on the per-query 0/1 Bernoulli stream

Results CSV carries a `level` column so the two channels are never merged.

---

## Preflight and degenerate metrics

Two of the six metrics depend on a model that is loaded lazily and can fail
silently: LS needs an NLI checkpoint, SS needs BERTScore. rqeval swallows the
loading error and falls back — LS then returns a constant 1.0 and SS a
near-constant Jaccard score, with no warning. A third failure mode is
configurational: K repeated calls issued at `temperature=0` are identical, so
CS and SS are 1.0 by construction.

`ReasoningMonitor.preflight()` checks all three **before the first model call**
and writes its report into the manifest. `publication_mode: true` turns its
findings into errors, and additionally fails any run in which a windowed metric
never varied (`MonitorResult.degenerate_metrics()`).

```bash
python -m reasoning_monitor.cli --config configs/live_asu.yaml \
    --coherence nli --stability bertscore --publication-mode
```

## Reproducibility

Every run writes `results/manifest.json` with the package version, rqeval
provenance, dataset content hashes, scorers, detector strategy, the preflight
report and all config values. A run is reproducible if it produces the same
manifest.

GSM8K difficulty tiers preserve reference-solution operation counts and write
their audit to `difficulty_proxy_validation.json`. Runs containing ST3-ST5 also
write `difficulty_cq_validation.csv`, including the preregistered monotonic CQ
and adjacent-gap checks.

The default campaign runs the smaller ARC-Challenge replication automatically
after the primary GSM8K campaign. It uses ST2/ST3/ST5 and three independent
seeds, writes to `results/main/replication_arc_challenge`, and records a compact
replication summary in the primary manifest. Set `run_replication: false` for
development and gateway pilots.

Live calls use bounded exponential-backoff retries. Every individual call is
flushed to `records_<tag>.jsonl` before the next call and before scoring;
`resume: true` reconstructs observations and issues only missing calls, even
inside a partially completed K-repeat/L-paraphrase group. A truncated final
JSON line from a hard interruption is discarded while valid records are kept.
Failed calls remain in the raw record with `ok=false` and are excluded from
metric scoring.

Before a paid ASU campaign, inspect the optional gateway controls with a dry
run. No network request is sent unless `--execute` is present; the executing
form sends five minimal probes and requires `ASU_API_KEY`:

```powershell
$env:PYTHONPATH = "src"
python scripts/probe_gateway.py --model openai/gpt5_5
python scripts/probe_gateway.py --model openai/gpt5_5 --execute `
  --output results/gateway_probe_gpt5_5.json
```

Only controls reported as accepted should then be copied into a model's
`request_options` in `configs/live_asu.yaml`.

`configs/pilot_asu.yaml` is an infrastructure and metric-health pilot, not a
drift result. It uses one seed, ST3, 12 queries, both live models, K=5 and L=3:
192 model calls in total. `publication_mode` keeps scorer preflight strict,
while `fail_on_degenerate: false` records short-pilot degeneracy instead of
discarding the paid run. The full campaign leaves the query cap unset and
retains fatal degeneracy checks.

`configs/methodology_pilot_asu.yaml` records the completed frontier-model
exploratory pilot and is retained for provenance. It uses 12 queries per regime: ST2 has 36
queries with pilot change points at 12 and 24, while ST3/ST4/ST5 contribute 12
queries each. Across two models at K=5 and L=3 this is 1,152 model calls. Check
the budget and run it from Windows Miniconda Prompt with:

```cmd
python scripts\estimate_campaign.py configs\methodology_pilot_asu.yaml
python -m reasoning_monitor.cli --config configs\methodology_pilot_asu.yaml
```

The accepted proposal's main models are GPT-4o-mini, Llama-3-8B and
Mistral-7B. Their separate ASU-hosted validation pilot is defined in
`configs/main_models_pilot_asu.yaml`; Llama and Mistral confirmatory runs remain
assigned to the local GPU as stated in the proposal.

That main-model pilot completed but did not pass the preregistered CQ tier
gate. The next calibration is deliberately limited to Llama-3-8B and CQ. It
uses fixed GSM8K reference-solution bands (Easy 1-2 operations, Medium 3-4,
Hard 5+) with 30 questions per stationary tier, K=1 and L=0: 90 calls total.
From Windows Miniconda Prompt:

```cmd
python scripts\estimate_campaign.py configs\step_band_calibration_llama_asu.yaml
python -m reasoning_monitor.cli --config configs\step_band_calibration_llama_asu.yaml
```

The output is isolated under `results/step_band_calibration_llama`. Passing
requires `Easy > Medium > Hard` and both adjacent CQ gaps to be at least 0.10.
Auxiliary metric columns may still be emitted by the framework, but this
calibration makes no inference or decision from drift or the other five metrics.

After Llama passes, validate the unchanged CQ gate on GPT-4o-mini and
Mistral-7B. This is another CQ-only run with 30 questions per tier and 180
calls total:

```cmd
python scripts\estimate_campaign.py configs\step_band_calibration_gpt_mistral_asu.yaml
python -m reasoning_monitor.cli --config configs\step_band_calibration_gpt_mistral_asu.yaml
```

Its output is isolated under `results/step_band_calibration_gpt_mistral`.
Both models must independently pass before the ST2/K/L validation begins.

The custom solution-step route is now frozen after its failed replication.
The next, separate method-development run uses the official Easy2Hard-Bench
E2H-GSM8K ratings with a fixed calibration/holdout split. It contains 270
primary calls and does not use synthetic fallback:

```cmd
python scripts\estimate_campaign.py configs\e2h_benchmark_calibration_main_models_asu.yaml
python -m reasoning_monitor.cli --config configs\e2h_benchmark_calibration_main_models_asu.yaml
python scripts\assess_e2h_benchmark_calibration.py results\e2h_benchmark_calibration_main_models
```

Do not run the second command until explicitly authorising the paid campaign.
Only models passing the frozen continuous-rating CQ gate may proceed to the
disjoint holdout K/L run.

The calibration completed with 270/270 successful calls. Mistral-7B alone
passed the frozen gate (CQ 0.633/0.467/0.233, Easy-Hard gap 0.400, one-sided
permutation p=0.000200). GPT-4o-mini and Llama-3-8B each had an Easy-Hard gap
of 0.100 and failed. The next eligible experiment is therefore a Mistral-only
K=5/L=3 run on the untouched holdout partition. That Section 42 plan was
superseded before execution so GPT-4o-mini and Llama-3-8B can first be tested
outside their ceiling-limited range. Do not run the old Mistral holdout config.

The active Section 43 range calibration uses disjoint data, bottom/central/top
20% official GSM8K quantile bands, K=1/L=0 and 180 calls total:

```cmd
python scripts\estimate_campaign.py configs\e2h_extreme_calibration_gpt_llama_asu.yaml
python -m reasoning_monitor.cli --config configs\e2h_extreme_calibration_gpt_llama_asu.yaml
python scripts\assess_e2h_extreme_calibration.py results\e2h_extreme_calibration_gpt_llama
```

These commands reproduce the completed Section 43 run. Models passing Section
43 become eligible for a later final-holdout K/L design. A failed model moves
only to the declared official E2H-AMC fallback, not to post-hoc GSM8K threshold
tuning.

Section 43 completed with 180/180 successful calls. Llama-3-8B passed at
CQ 1.000/0.833/0.633, Easy-Hard gap 0.367 and p=0.000200. GPT-4o-mini had a
significant negative slope (p=0.001600) but failed the independent gap rule at
0.133 versus the frozen 0.15 minimum. Llama is eligible for final-holdout K/L;
GPT proceeds only to the preregistered E2H-AMC fallback.

Section 44 used GPT-4o-mini only, the pinned official E2H-AMC `eval` split,
K=1/L=0, and 30 questions from each published difficulty tercile. Its
corrected call-free rescore passed the statistical gate at CQ
0.267/0.033/0.033, but 75/90 outputs hit the 512-token cap. Because truncation
was difficulty-dependent, the run is execution-invalid and does not authorise
holdout K/L.

Section 45 keeps the exact questions and statistical gate, records provider
`finish_reason`, and changes only the mechanical output cap to 2048 tokens:

```cmd
python scripts\estimate_campaign.py configs\e2h_amc_calibration_gpt4o_mini_retry_asu.yaml
python -m reasoning_monitor.cli --config configs\e2h_amc_calibration_gpt4o_mini_retry_asu.yaml
python scripts\assess_e2h_amc_calibration.py results\e2h_amc_calibration_gpt4o_mini_retry
```

The Section 45 retry completed with 90/90 provider-normal stops and zero
truncations. A pre-interpretation audit found answer-extraction false negatives
in seven `\\text{Answer: ...}` wrappers and one algebraically equivalent tuple.
The call-free assessor now recomputes CQ from immutable raw records and retains
the original online CQ values in `data/e2h_amc_cq_rescore.csv`. Eight of 90
values changed. Final Easy/Medium/Hard CQ is 0.600/0.100/0.100, the Easy-Hard
gap is 0.500, slope is -0.808150 and one-sided permutation p=0.000200. The
statistical and execution gates both pass, so GPT-4o-mini is eligible only for
a separately preregistered E2H-AMC holdout K/L design. The second command above
would repeat paid calls and must not be rerun; the first and third are
call-free.

Section 47 preregisters the final K=5/L=3 metric-health family before any
holdout calls. Llama-3-8B uses the extreme-band E2H-GSM8K `final_holdout`,
GPT-4o-mini uses the official-tercile E2H-AMC `holdout`, and Mistral-7B uses the
official-tercile E2H-GSM8K `final_holdout`. Each track has 90 unique ordered ST2
questions and 720 resumable calls. All three configs are frozen now; executing
Mistral later is an operational sequence and must not depend on the preceding
Llama/GPT outcomes.

```cmd
python scripts\preflight_e2h_final_holdouts.py
python scripts\estimate_campaign.py configs\e2h_final_holdout_llama3_8b_asu.yaml
python scripts\estimate_campaign.py configs\e2h_final_holdout_gpt4o_mini_asu.yaml
python scripts\estimate_campaign.py configs\e2h_final_holdout_mistral_7b_asu.yaml
```

The completed call-free preflight selected 30 Easy, 30 Medium and 30 Hard
questions per track, with no overlap against any prior calibration item. The
three tracks total 2,160 calls; the intended first operational stage, Llama plus
GPT, totals 1,440. Paid execution has not started. Section 47 uses a common
2,048-token mechanical ceiling, a zero-truncation execution gate, a
family-adjusted one-sided slope alpha of `0.05/3`, and an independent minimum
Easy-Hard CQ gap of 0.15. Raw CQ levels are not pooled across the model-specific
task ranges; any later cross-model analysis must use preregistered within-model
normalization.

The Llama Section 47 track subsequently completed all 720 calls with zero
failures, zero truncations and no CQ rescore changes. Its holdout CQ profile was
0.900/0.867/0.633, with Easy-Hard gap 0.267, slope -0.536946 and one-sided
permutation p=0.000200, so the frozen CQ generalisation gate passed. The overall
track nevertheless failed the independently frozen metric-health gate: only
60/90 questions had at least two distinct verbatim completions among K=5, a
diversity rate of 0.667 versus the 0.80 minimum. Identical-repeat counts were 14
Easy, 11 Medium and 6 Hard questions. The threshold is not changed after this
result. The GPT-4o-mini track remains the next unexecuted Section 47 arm.

The GPT-4o-mini Section 47 track then completed all 720 calls with zero
failures, zero truncations and no CQ rescore changes. K-repeat diversity was
1.000, 32 questions had evaluable RS, every metric-health check passed, and CQ
was 0.767/0.233/0.067. The Easy-Hard gap was 0.700, slope was -1.155497 and the
fixed-seed one-sided permutation p-value was 0.000200. Both the CQ and overall
track gates pass. These saved records are eligible for a separately
preregistered call-free normalized detector analysis, but that analysis is not
started before the already frozen Mistral track is resolved.

The Mistral-7B Section 47 track also completed all 720 calls with zero failures
and zero truncations. Its corrected CQ profile was 0.767/0.433/0.267, with
Easy-Hard gap 0.500, slope -0.949548 and p=0.000200, so CQ generalisation
passed. The overall track failed because every question produced five identical
primary completions: K-repeat diversity was 0/90 = 0.000. RS remained evaluable
for 44 questions and the remaining metric checks passed, but the frozen 0.80
diversity requirement cannot be relaxed.

During the call-free Mistral audit, the generic numeric normalizer was found to
remove commas before applying a three-digit thousands regex, which could reduce
both `14,000` and `4,000` to `0`. The parser now accepts arbitrary-length plain
integers and comma-grouped integers separately, with regression tests. All
Section 47 reports were regenerated from immutable raw records: Llama was
unchanged, GPT's AMC-specific symbolic path was unaffected, and the corrected
Mistral CQ values above match online CQ. Across the complete family, only
GPT-4o-mini passes the overall gate and is eligible for the next preregistered
call-free analysis.

Section 49 completed that GPT-only analysis without issuing model calls or
refitting detectors. CQ detected both frozen transitions at t=37 and t=79. LS
led CQ by one query at the first transition but did not alarm at the second; ES
trailed CQ by two queries at the first transition and also missed the second.
CS and SS did not alarm in either interval. Consequently, no non-CQ candidate
passed the preregistered requirement to lead CQ in both transitions. This is a
single-model negative result for early warning under the frozen proposal
mapping, not grounds for post-hoc threshold tuning. The report is under
`results/e2h_final_holdout_gpt4o_mini/SECTION49_GPT_DETECTOR_ANALYSIS.md`.

A subsequent call-free parser audit found that final-answer extraction does not
explain the CQ, CS or raw K-diversity results: after correcting one nested AMC
answer-label edge case, Llama and GPT have zero raw-rescore CQ or CS changes.
The reasoning-trace splitter is a genuine limitation, however. It kept only the
first 12 period-delimited fragments; all 450 GPT K-runs were capped and retained
a median 26.3% of fragment characters, while Llama capping increased from 14.7%
on Easy to 58.7% on Hard. LS and SS therefore require a structure-aware,
call-free reparse sensitivity analysis before they receive scientific weight.
The historical outputs remain unchanged. See
`results/answer_parsing_audit/ANSWER_PARSING_AUDIT.md`.

Section 51 completed that sensitivity analysis on the same 90 GPT questions.
Corrected LS differed sharply from the prefix-based values (Pearson 0.286,
mean absolute change 0.186), and the old t=36 LS alarm disappeared. Corrected
LS produced no alarm. SS was comparatively stable (Pearson 0.861, mean
absolute change 0.010) and still produced no alarm. Neither corrected metric
passes the original two-transition early-warning rule, so the parser fix does
not rescue the Section 49 hypothesis. Future runs use the complete
structure-aware trace parser; historical outputs remain preserved for audit.

Section 52 adds a call-free multiple-choice robustness track using the pinned
`m-a-p/SuperGPQA` revision `4430d445`. The structural audit found 75 rows with
duplicate options and two duplicate-question pairs, all excluded before
selection. A model-blind rule selected the largest balanced ten-option
subfield, Science > Chemistry > Analytical Chemistry, and froze disjoint pilot
and validation pools of 20 Easy, 20 Medium and 20 Hard questions each. The
pilot answer labels are exactly balanced across A-J, the integration preflight
passes, and its ceiling is 60 calls / 30,720 output tokens. Difficulty is partly
LLM-informed upstream and calculation prevalence rises from 0.133 on Easy to
0.975 on Hard, so this is an exploratory composition-shift check rather than a
confirmatory replacement for the primary design. The paid GPT-4o-mini pilot
remains on hold for explicit execution approval. See
`results/supergpqa_pool_audit/SECTION52_SUPERGPQA_POOL_AUDIT.md`.

The first 60-call pilot subsequently failed for an operational reason: 16
outputs hit the exact 512-token ceiling while the gateway reported `stop`, and
all ended before the required choice marker. Censoring rose from 0/20 Easy to
4/20 Medium and 12/20 Hard. Conservative CQ was 0.45/0.15/0.05, but the
missing-outcome bounds were [0.45,0.45]/[0.15,0.35]/[0.05,0.65], so the apparent
difficulty gap is not identified. Section 53 therefore freezes a homogeneous
rerun of all 60 questions: choice on the first line, at most four short
reasoning sentences, and a 1024-token ceiling. This retry is separate from the
failed pilot and remains on hold for explicit execution approval.

The homogeneous Section 53 retry passed every screening gate with zero
truncations and 100% first-line choice compliance. GPT-4o-mini CQ was
0.35/0.25/0.10 (Easy/Medium/Hard), the Easy-Hard gap was 0.25, the one-sided
ordered-trend p-value was 0.0308 and the descriptive Easy-Hard Fisher p-value
was 0.0637. Section 54 freezes the already reserved disjoint 60-question
validation pool under the identical answer-first protocol. Its confirmatory
gate additionally requires one-sided ordered-trend p<=0.05.

The completed Section 54 validation had clean execution and parsing but failed
the frozen conjunctive gate. CQ was 0.25/0.30/0.05, so Medium exceeded Easy by
one item, Hard fell below the preregistered 0.10 nondegeneracy floor, and the
ordered-trend p-value was 0.0569. The Easy-Hard gap equalled its 0.20 threshold.
These results retain a directional Easy/Medium-to-Hard contrast but do not
validate a stable three-tier dose response. Thresholds are not relaxed and the
result does not automatically authorise expansion to Llama.

**Raw generations are persisted first.** Every model call is written verbatim to
`results/<...>/data/records_<tag>.jsonl` (prompt, completion, extracted answer,
tokens, finish reason, latency, success flag) before anything interprets it, and per-query
metric values to `per_query_<tag>.csv`. A scoring bug found after a campaign is
therefore fixed by re-reading those files, not by re-running the campaign.

---

## Deviations from Proposal

See `DEVIATIONS_FROM_PROPOSAL.md` for the complete protocol and deviation
record. Section 69 contains the terminal evidence synthesis and stop rule.

## Six-Metric Dialogue Collection

### Limited pilot (336 calls, not the full campaign)

Use `python scripts\run_six_metric_mini.py --preflight` for call-free validation,
then `python scripts\run_six_metric_mini.py --execute` only to start the limited
pilot. The same execution command resumes checkpoints. `--score-only` regenerates
scores and plots locally after completion. Default with no flag is a dry run.
The pilot fixes budget-002, both models, both conditions and all 12 turns at
K=5/L=2. It adopts the explicitly post-hoc v2 sampling rule only for this pilot;
the original 3,024-call execution gate is unchanged. Outputs reside in
`results/dialogue_six_metrics_mini_v1`. At most 688,128 output tokens are planned
before any provider retry overhead. This reused calibration dialogue is not a
holdout. Descriptive first-error alignment is not an early-warning validation.

`python scripts\audit_six_metric_sampling.py` audits the existing 20-call probe
without new remote calls. It validates provenance, separates raw text and
normalized-answer diversity, and computes actual CS/BERTScore SS. Results and
a sparse-turn six-panel plot go to `results/dialogue_six_metrics_sampling_audit`.
RS/ES remain unavailable because this probe has no paraphrases or consecutive
window. The original probe JSON/report and execution gate are not changed.
The separately reported v2 readiness rule is post-hoc, not an independent pass:
all cells must be operationally valid, with diversity in at least one sampled
cell per model, rather than every cell. Further execution requires explicit
adoption of the amended protocol; this audit does not unlock paid execution.

`python scripts\dialogue_six_metrics.py` prints the call-free acquisition plan.
`--preflight` checks cached real NLI and BERTScore before any paid calls.
`--historical` scores existing CDR-v1.4 responses locally (CQ, LS, ES only).
Missing CS/RS/SS are explicitly unmeasured, never assigned perfect scores.

The new `--probe` mode makes 20 paid sampling checks. Inspect its report first.
`--execute` requires the probe to pass and makes up to 3,024 additional calls:
432 cells, five same-context samples including the primary, and two fixed
paraphrases per cell. Temperature 0.7 and L=2 are explicit calibration changes;
this is not a continuation of the old temperature-zero single-sample run.
Neither paid mode is the default. All calls have progress/ETA and checkpoint
resume; failed calls remain visible and are not silently replaced on resume.
`--score-only` checks completeness and rescores without paid calls.

Each completed analysis writes six-panel PNG/PDF plots per model and per
dialogue, a CSV and a provenance manifest. RS excludes incorrect primaries;
ES depends on correctness and uses a causal five-turn window. Missing scores
and warmup are reported separately. Flat curves are retained, not hidden.
Candidate early warning must predict later errors, survive dialogue-level
validation and account for testing multiple metrics. A curve alone is not
evidence of early warning.
