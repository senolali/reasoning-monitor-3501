# CDR-Stream: Purpose-Built Long Dialogue Streams for Reasoning-Drift Monitoring

Status: **DESIGN DRAFT, D1–D10 accepted; D11 drafted at the investigator's request (2026-10-03). Not frozen. No data generated, no model call.**
This document proposes a dataset and study design. Nothing here authorizes
generation of a frozen bank, paid calls or any change to earlier evidence.
CDR-Stream is a new independent protocol. It is not Section 70, and it does not
reopen Section 69, RAT-v1, CDR-v1.x, DBQR or LIC-RQ-v2.

## 1. Why a new dataset

Every earlier route failed on the data, not on the monitoring idea:

| Route | What it lacked |
| --- | --- |
| Independent-question streams (S1, E2H, Sections 33–69) | No shared conversational state, so there is no context accumulation to degrade |
| CDR-v1.x (12-turn exact-solver dialogues) | Too short for W-windows; one question type; no difficulty control; only CQ measured |
| LIC-RQ-v2 (sharded math) | Gold answer only at the final turn; CQ/RS/ES unobservable before it; at most ~9 turns |
| LIC-RQ-v2-SR replay | 48 items per regime: under family-wise error control the detectors could not see even a confirmed 0.21–0.29 accuracy loss |
| DBQR-QA | Tables too large for prompting; plan-DSL route failed |

The proposal's claims (H1–H5, S1–S6, W=20 windows, metric-specific detectors)
require all of the following at once. No public dataset found in the 2026-10-03
search satisfies them together. Long-Horizon-Execution (Sinha et al., 2025)
comes closest, but its turns are pure look-up-and-add execution, which makes LS
nearly meaningless and gives no domain variety.

| ID | Requirement | Reason |
| --- | --- | --- |
| R1 | Exact gold answer at **every** turn | CQ, RS, ES and RQ6_mean observable at every turn |
| R2 | Turns depend on a shared, evolving state | Genuine context accumulation and error propagation |
| R3 | Long horizon, about 100 turns per dialogue | W=20 windows plus detector warm-up must fit inside one dialogue, with room before and after change points (the proposal's S1–S3 are 100-step streams) |
| R4 | Per-turn difficulty knob with validated effect | Gradual, sudden and recurring drift as scheduled difficulty |
| R5 | Matched control arm with the same turns and no accumulated history | Separates difficulty drift from history-induced reasoning drift (the proposal's input/task vs reasoning-drift distinction) |
| R6 | Meaning-preserving paraphrases per turn, with gold preserved by construction | RS without the wrapper-only failure of Section 65 |
| R7 | Multi-step reasoning content in more than one domain | LS/SS meaningful; the proposal's math/logic/science domains |
| R8 | Surface-only manipulations (verbosity, noise) at fixed difficulty | The proposal's S4 linguistic and S5 noisy-prompt drift |

## 2. Generator design

### 2.1 Dialogue unit

One dialogue has T=100 turns (Decision D1) over one shared world state, for
example a research budget ledger. Each user turn does two things:

1. **Update:** introduces zero or more state-changing events (an expense, a
   transfer, a correction to an earlier event).
2. **Query:** asks one question whose exact integer or choice answer is
   computed by the solver from the current state and history.

Only the model's primary response enters later dialogue history (the LIC/CDR
rule). Server-side history stays disabled. The history is sent as the packed
transcript validated in CDR-v1.1.

### 2.2 Difficulty ladder

Difficulty `d in {1,...,8}` is a per-turn label set by the generator. The ladder
is deliberately wider than any single model needs (Section 2.2b). It is composed
of separately recorded axes, so the analysis can attribute effects:

| Axis | d=1 | d=8 |
| --- | --- | --- |
| Operations needed to answer | 1 | 8–10 |
| Entities referenced | 1 | 6–8 |
| Dependency distance (oldest turn whose information is needed) | current turn | 40–80 turns back |
| Event types | additive | additive + corrections + conditional rules + retroactive reversals |
| Distractor facts in the update | 0 | 4–5 |

Each axis must have a monotone effect on single-turn accuracy. That is checked
empirically before any stream is built (Gate G1). An unvalidated difficulty
proxy is exactly what broke E2H (Section 40).

### 2.2a Query-type balance

Every template has at least six query families (for example: current balance,
cumulative total since turn k, comparison between two entities, threshold
crossing, correction-aware recount, conditional rule application). Families are
assigned to turns by a balanced, seeded schedule:

- Every query family occurs at every difficulty level it supports.
- Every family occurs in every regime of every scenario.
- No family exceeds 25% of any 20-turn window, and no family occurs twice in a
  row.

The family label is stored per turn. This prevents a stream from consisting of
one question type, which would make CS/SS artificially close to 1 and would
confound a change of question type with drift.

### 2.2b Ceiling and floor control

A model with near-perfect accuracy has no room to degrade, and its CS/SS
saturate at 1. A model near zero accuracy cannot degrade further. LIC-RQ-v2
showed both: Llama-4 Scout's dialogue arm was still 0.854, and Mistral-7B's was
0.063. CDR-Stream therefore fixes the operating range in terms of **expected
single-turn accuracy**, not raw difficulty:

| Regime | Target single-turn CQ (state_summary, K=1) |
| --- | --- |
| easy mixture E | about 0.85 (band 0.75–0.92) |
| medium mixture M | about 0.70 (band 0.60–0.78) |
| hard mixture H | about 0.50 (band 0.35–0.65) |

The procedure has four parts:

1. Gate G1 measures single-turn CQ at every ladder level d=1..8 for every model,
   on **calibration-split** turns only.
2. A frozen rule maps each (model, template) to its own centre levels
   `e < m < h`. Each centre is the level whose CQ is closest to its target,
   subject to strict ordering; ties go to the lower level. Each mixture spreads
   over neighbouring levels (Section 3).
3. **Exclusion rule, frozen before G1.** A (model, template) is excluded rather
   than tuned if any of the following holds:
   - no strictly ordered `e < m < h` exists with each centre's CQ inside its band;
   - the E–M or M–H CQ gap is below 0.10;
   - the E–H CQ gap is below 0.25.

   A model excluded on more than half of the templates is dropped.
4. The mapping is frozen before any stream is generated. It is never revised
   after stream results are seen.

Consequence: models see different absolute difficulty but matched expected
accuracy. Cross-model comparison (H4) is then about degradation profiles at a
matched operating point, which is the meaningful comparison. Absolute
difficulty is still recorded per turn for secondary analysis.

Ceiling also applies to CS/SS through deterministic sampling. Gate G2 keeps the
K-repeat diversity check (>= 0.80). A model that fails it reports CS/SS as
degenerate instead of silently constant.

### 2.3 Templates (domains)

The proposed set has four templates (Decision D2). Each has its own exact solver.

| Template | Domain | State | Example queries |
| --- | --- | --- | --- |
| `budget_ledger` | arithmetic/accounting | budget lines, transfers, corrections | remaining in line X; total spent on categories A and B since turn k |
| `inventory_logistics` | multi-entity quantitative | stock per warehouse, shipments, returns | units of item Y across warehouses; which warehouse first falls below threshold |
| `schedule_constraints` | logic | tasks, precedence and resource constraints | earliest start of task Z; how many tasks are feasible by day n |
| `lab_protocol` | science/units | reagent volumes, concentrations, dilutions | final concentration in mmol/L; remaining volume after the protocol |

The CDR-v1 templates are not reused verbatim, because they asked the same
quantity every turn.

### 2.4 Paraphrases (R6)

Each turn is realized from a semantic specification (events, query and
parameters) by a surface realizer with at least four independent templates per
query type. The primary prompt uses realization 0. RS uses realizations 1..L.
Paraphrases change wording, order of clauses and number formats, not the
specification, so the gold answer is preserved by construction. A call-free
check re-derives every paraphrase's answer from its own parsed specification. A
human-reviewed random sample (Decision D6) is required before confirmatory use.

### 2.5 Surface manipulations (R8)

- **Verbosity level:** narrative padding at fixed d (S4).
- **Noise level:** typos, casing, missing punctuation and informal phrasing at
  fixed d, never touching numbers or entity names (S5).

Both are generated deterministically from seeds and recorded per turn.

### 2.6 Answer contract

The contract reuses what passed in Sections 57, 61 and 66: at most eight short
numbered reasoning lines, then exactly one final line `FINAL_ANSWER: <integer>`
(or `<choice>`). The marker-anchored semantic parser is frozen before any model
output, and integer gold avoids unit ambiguity. Truncation is detected
explicitly with a 2,048-token ceiling.

## 3. Drift scenarios as schedules over turns

Change points follow the proposal's 100-step streams. Scenarios do not switch
between pure difficulty levels. They shift the **mixture** of levels, which is
the proposal's own concept-drift definition: a change in P(q|t). With the
model-specific centres `e < m < h` from Section 2.2b:

- **Easy mixture E:** 50% at e, 20% at each neighbouring level, 10% from M.
- **Medium mixture M:** 50% at m, 20% at each neighbouring level, 5% from E and
  5% from H.
- **Hard mixture H:** 50% at h, 20% at each neighbouring level, 10% from M.

Each mixture sums to 100%; weights are renormalized at the ladder ends.
Turn-level levels come from a seeded, stratified schedule, so the realized
mixture in every window matches the nominal one (no accidental all-hard runs).

Difficulty therefore changes **in graded steps** (easy -> medium -> hard), and
every adjacent pair of regimes shares levels, so no transition is a jump between
disjoint question pools.

| ID | Mixture schedule | Surface | Role |
| --- | --- | --- | --- |
| S0-E / S0-M / S0-H | E / M / H throughout | fixed | **Null / false-alarm calibration**, one null per regime. Context grows, task mixture does not |
| S1 staircase | E at 1–33, M at 34–67, H at 68–100 | fixed | **Proposal S1** (the proposal's three-phase easy -> medium -> hard stream) |
| S1-ramp | E -> M -> H weights interpolated continuously over 1–100 | fixed | Smooth gradual drift without step boundaries |
| S1-reverse | H at 1–33, M at 34–67, E at 68–100 | fixed | Recovery / reverse ordering (the ST6 analogue). Separates "turn position" from "difficulty" |
| S2 sudden | E at 1–50, H at 51–100 | fixed | Proposal S2 |
| S3 recurring | E / H / E at 1–33 / 34–66 / 67–100 | fixed | Proposal S3, recovery |
| S4 linguistic | E throughout | verbosity ramps over 34–67 | Proposal S4 |
| S5 noisy | E throughout | noise ramps over 34–67 | Proposal S5 |
| S6 mixed | S1 staircase plus pure-H spikes at three fixed turns in the E and M phases | fixed | Proposal S6 |

S1 staircase versus S1-reverse is the cleanest test of whether a decline over
turns comes from difficulty or from accumulated context. In S1-reverse,
difficulty falls over time while context still grows. Rising CQ would indicate a
difficulty-driven trajectory; a smaller recovery than the matched `state_summary`
arm would indicate a context-accumulation penalty.

Because the mixtures overlap, the same difficulty levels occur both before and
after every change point. This gives a second, **within-arm difficulty-matched
contrast**: CQ at the same level and query family, early versus late in the
same dialogue. It isolates context accumulation inside the dialogue arm,
complementing the between-arm contrast with `state_summary`.

Two drift sources are therefore crossed: the **scheduled task drift** above, and
the **context-accumulation drift** present in every dialogue-history run. The
control arm (Section 4) carries only the first.

## 4. Conditions

| Arm | What the model sees at turn t | Calls per turn | Purpose |
| --- | --- | --- | --- |
| `dialogue_history` (primary) | all prior user turns + own prior primary replies + current turn | K repeats + L paraphrases | the deployed, monitored system |
| `state_summary` (control) | solver-verified state summary before t + current turn | 1 (Decision D4: or K) | same task drift, no accumulated history |

`state_summary` was the strongest CDR arm (CDR-v1.3/1.4). The per-turn
difference `CQ(state_summary) - CQ(dialogue_history)` is reasoning drift net of
task difficulty. An `oracle_history` arm is not proposed: CDR-v1.1 showed it
was not a clean control.

## 5. Metrics at every turn

| Metric | Measurement in CDR-Stream | Observable every turn? |
| --- | --- | --- |
| CQ | exact match of primary answer to gold | yes (R1) |
| CS | answer agreement over K=5 same-context repeats | yes |
| RS | rqeval C-subset over L paraphrases | when the primary answer is correct (by design) |
| LS | NLI over the numbered reasoning lines | yes |
| ES | rqeval efficiency with its causal window | yes after warm-up |
| SS | BERTScore over K traces | yes |
| RQ6_mean | complete-case mean | yes, wherever RS is defined |

No metric is structurally unobservable, which removes the main LIC-RQ-v2
limitation. Missingness is still reported, never imputed.

## 6. Analysis plan (to be frozen with the bank)

- **Time axis:** turn index within a dialogue. Detectors run per dialogue on
  per-turn values with the existing `WindowMeanMetric`, W in {10, 20, 30}
  (W=20 primary).
- **Null:** S0 dialogues, not permutations. Within a dialogue, turn order
  carries accumulated context, so permuting turns would destroy the very
  structure being monitored. Thresholds are calibrated on calibration-split S0
  dialogues and false-alarm rates evaluated on held-out S0 dialogues.
- **H1:** pre-change vs post-change windows per metric in S1–S3, as dialogue
  arm minus control arm, with dialogue-clustered inference.
- **H2:** the three existing strategies (`distribution_based`,
  `proposal_mapping`, `adwin_all`) on identical runs, compared on detection
  rate, signed delay and held-out false-alarm rate.
- **H3:** first-alarm order of non-CQ metrics relative to CQ, gradual (S1)
  versus sudden (S2), with the proposal's directional predictions.
- **H4:** model-by-metric degradation profiles.
- **H5:** joint CQ/ES patterns. This is newly testable because ES is observable.
- **Prospective two-sided arm (Decision D5).** The LIC-RQ-v2-SR replay showed LS
  and SS rising under accumulated history. A secondary, prospectively declared
  arm would therefore monitor absolute change in LS/SS, alongside the proposal's
  primary decrease direction. Declaring it before CDR-Stream data exist is
  legitimate: it is exactly the way an exploratory observation becomes a
  prospective test.
- **Confirmatory inference** uses untouched holdout dialogues only, clustered by
  dialogue, with Holm correction within each hypothesis family.

## 7. Staged gates

| Gate | Calls | Passes when |
| --- | --- | --- |
| G0 generator | none | solver round-trip on every turn; paraphrase answers re-derived; answer distribution not guessable; difficulty axes logged; deterministic hash |
| G1 difficulty calibration | small, single-turn (`state_summary`, K=1): about 30 calibration turns per (model, template, level), i.e. 8 x 4 x 30 = 960 calls per model | CQ is monotone (non-increasing) over the ladder; the frozen Section 2.2b rule finds ordered easy, medium and hard levels inside their bands for each (model, template), with adjacent gaps >= 0.10 and an E–H gap >= 0.25; otherwise the frozen exclusion rule applies. Mapping frozen before any stream |
| G2 transport and diversity | small | packed history intact at t=100; no context overflow; zero truncation; K-repeat answer/trace diversity >= 0.80 (the Llama-4 failure of Section 64) |
| G3 calibration streams | moderate | detector thresholds calibrated on S0 only; technical checks only, no effect reading |
| G4 confirmatory holdout | main | one frozen run; no resizing based on earlier effects |

Each paid gate needs its own explicit investigator approval, as before.

## 8. Budget shape

Calls per dialogue per model are `T * ((K + L) + c)`, where `c` is the
control-arm calls per turn. Input tokens grow with history: at about 250 tokens
per turn (bounded reply plus short user turn), the mean input is roughly
`T/2 * 250` plus the prompt. For T=100 that is about 13k tokens, with about 25k
at the final turn.

Illustrative only, to be replaced by a call-free estimator. Take T=100, K=5,
L=2 and c=1, so about 800 calls per dialogue. The core of D3 has seven scenario
types (S0-E, S0-M, S0-H, S1 staircase, S1-reverse, S2, S3).

| Dialogues per (scenario, template) | Dialogues per model | Calls per model | Calls, three models | Input tokens (approx.) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 84 | 67,200 | 201,600 | 2.6 billion |
| 2 | 56 | 44,800 | 134,400 | 1.7 billion |

The first row is about 18 times LIC-RQ-v2. Plus G1 (about 960 calls per model)
and G2/G3. The main levers are:

- dialogues per cell;
- K=3 instead of 5 (about -30%);
- shorter null dialogues (S0 only needs to cover the window and detector span);
- moving S1-reverse to the secondary arm.

The choice must be made with the real-token estimator and the ASU price list,
before freezing.

**Context-limit risk.** The ASU `aws/mistral-7b` context window must be
verified before T is frozen; 25k tokens may exceed it. Gate G2 tests this
explicitly.

## 9. Lessons built in

- Packed-transcript transport (CDR-v1.1), with server history off.
- Answer-last bounded reasoning (Sections 56–57), typed marker, frozen semantic
  parser (LIC integration lesson), and explicit truncation detection
  (Sections 52 and 62).
- Genuine paraphrases, not wrappers (Section 65), with semantic answer policy
  `semantic-v1` (Section 68).
- Validated difficulty before streams (E2H, Section 40).
- Stochastic repeat diversity gate (Section 64).
- Error control calibrated on proper nulls; default thresholds are unusable
  (Section 36 and LIC-RQ-v2-SR).
- Dialogue-level clustering; turns are not independent units (CDR-v1.3).

## 10. Decisions for the investigator

D1–D8 were accepted as recommended on 2026-10-03. At the same time the
investigator required that no scenario consist of a single question type or a
single difficulty, and that ceiling effects be controlled. Sections 2.2a, 2.2b
and 3 implement both requirements, and D9–D10 were added for them and accepted. The investigator then asked for graded easy -> medium -> hard transitions; D11 (Section 3) implements that request.

| ID | Decision | Recommendation |
| --- | --- | --- |
| D1 | Dialogue length T | 100, matching the proposal's streams; fall back to 80 if the G2 context test fails |
| D2 | Templates | the four in Section 2.3 (math, multi-entity, logic, science) |
| D3 | Scenarios in the confirmatory core | S0-E/M/H, S1 staircase, S1-reverse, S2, S3 in the core; S1-ramp and S4–S6 as a secondary, smaller arm |
| D4 | Control-arm sampling | K=1 (CQ only) to save budget; full K/L only if budget allows |
| D5 | Prospective two-sided LS/SS arm | yes, as a secondary arm |
| D6 | Human review of paraphrases and difficulty | random 5% of turns, by the investigator or a colleague, before G1 |
| D7 | Models | GPT-4o-mini, Llama-4 Scout, Mistral-7B, the latter conditional on G2 context and format checks |
| D8 | K and L | K=5, L=2 (the LIC-RQ-v2 values); K=3 only if the budget requires it |
| D9 | Accuracy targets and bands | easy about 0.85 (0.75–0.92), medium about 0.70 (0.60–0.78), hard about 0.50 (0.35–0.65); adjacent gaps >= 0.10, E–H gap >= 0.25; mapped per (model, template) |
| D10 | Mixture spread | 50% at the centre level, 20% at each neighbour, 10% from the adjacent regime (M splits it 5/5) |
| D11 | Graded transitions | three regimes E/M/H; S1 is a three-step staircase as in the proposal, with S1-reverse in the core and S1-ramp as secondary |

## 11. Next call-free steps after approval

1. Implement the generator, solvers, realizers and G0 validator with offline tests.
2. Produce a small, non-frozen sample bank for human review (D6).
3. Implement the call-free budget estimator from the generated bank's real token
   lengths.
4. Freeze the bank and the G1 protocol, then ask for G1 approval.

## 12. Implementation status (2026-10-03, call-free)

The draft generator follows package A for its core: S0-E, S0-H, S1 staircase, S2
and S3, plus the secondary scenarios. All were implemented and checked without
model calls.

- Code: `src/reasoning_monitor/data/cdr_stream_worlds.py` (four worlds with exact
  solvers and four-variant realizers) and `src/reasoning_monitor/data/cdr_stream.py`
  (skeletons, mixture/family schedules, instantiation, surface operations, G0,
  budget estimator). Script: `scripts/generate_cdr_stream_bank.py`. Tests:
  `tests/test_cdr_stream.py`.
- Draft bank: `configs/selections/cdr_stream_v1_draft_bank.json`, skeleton SHA-256
  `e1e045b26de43b36e85c60b9a9b38cc1df4a4310e2d04013e03a09a21ed0994c`. It has 172
  dialogues and 17,200 turns: core 5 scenarios x 4 templates x 5 dialogues
  (20 calibration, 20 development, 60 holdout), and secondary 6 x 4 x 3. Status
  is draft, not frozen.
- G0 passed with zero problems: deterministic regeneration; every skeleton's
  family and slot constraints; all 172 dialogues under the reference mapping;
  and all 56 valid (E, M, H) mappings for one S1 dialogue per template.
  Realized level shares match D10 (for example E-regime: centre 50%, each
  neighbour 20%, cross 10%). Family shares are 16.6–16.7% each. Mean paraphrase
  similarity is 0.36. The gold value appears verbatim in the current turn text
  1.9% of the time.
- Budget for package A from real text lengths (character/4 proxy, declared;
  assumed reply 150 tokens): 40 dialogues per model.

  | Setting | Calls (three models) | Input tokens (three models) |
  | --- | ---: | ---: |
  | K=3, L=2, control 1 | 72,000 | about 0.68 billion |
  | K=5, L=2, control 1 | 96,000 | about 0.95 billion |

  The final-turn context has a median of 22.8k tokens and a maximum of 27.7k
  tokens. G1 adds 2,880 single-turn calls (about 1 million input tokens).

Two refinements came out of implementation:

1. **Dependency distance is recorded per turn but is not part of `d`.** The
   control arm receives the verified state, so this axis is a history-load
   property by construction and cannot be calibrated in single-turn G1. In the
   draft bank its median is 6 turns, its upper quartile 10, and its maximum 63.
   The `d` ladder uses query terms, events, distractors, event complexity and the
   SUM threshold filter.
2. **Paraphrase fidelity is checked mechanically.** Every realization must
   contain exactly the specification's multiset of numbers and every referenced
   entity label. This replaces the natural-language re-parse described in
   Section 2.4. Semantic equivalence is still confirmed by the D6 human review.

The review file `results/cdr_stream_v1_draft/review_sample.csv` holds 860 rows
(5% of all turns), with empty review columns. A readable excerpt is in
`results/cdr_stream_v1_draft/example_dialogue.md`. Next steps: human review (D6),
the G1 single-turn item bank and protocol, then freezing.

**D6 review status (2026-10-03).** The investigator looked over the review
sample and reported no problems. The per-row review columns were not filled in,
so this is recorded as an informal look-through, not the full row-by-row 5%
review. A formal review can still be completed from `review_sample.csv` before
any confirmatory use.

**G1 preparation (2026-10-03, call-free).** `CDR_STREAM_G1_PROTOCOL.md`,
`src/reasoning_monitor/cdr_stream_g1.py`, `scripts/run_cdr_stream_g1.py` and
`configs/cdr_stream_v1_g1_asu.yaml` are prepared, and preflight passes. To
address ceiling effects prospectively, two contingency levels (9–10) were
defined before any model output. They are run only for cells whose fitted
level-8 accuracy stays above the hard band (G1b). The bank skeleton hash is
unchanged by this addition. G1 execution is locked pending approval.

## 13. Stream preparation and runner (2026-10-03, call-free)

**v2 bank.** The bank uses the final templates (milestone_chains replaces
schedule_constraints) and the G1 v2 ladder. File:
`configs/selections/cdr_stream_v1_bank_v2.json`, SHA-256
`8b224aa1befcb4bb60fe89acd3890a5d56a00c738f9a7eac85af6bac6e6d03d9`. It has 172
dialogues, and skeletons of shared templates are identical to the reviewed v1
draft, whose hash is unchanged.

**Units.** Units come from the frozen G1 mappings
(`results/cdr_stream_v1_mappings.json`, `9a63512b…`). Every instance is
regenerated deterministically and checked against a hash manifest; texts are
not stored. G0 passes for all units, with 0 problems.

- Pilot: `results/cdr_stream_v1_prep/manifest_pilot.json` (`5907e6d8ccbd362df3668d51da50145e17e217aefd59e8cc0abfeca13c86a0a4`). One
  calibration-split S1 staircase dialogue per core model: GPT on chains, Llama
  on inventory, Gemma 3 on budget. 1,800 calls, about 25.3M input tokens; max
  context about 37k tokens.
- Main (option 3b): `results/cdr_stream_v1_prep/manifest_main.json` (`90e397c51ac9f861ab7f17f9028a5b57ab3099cfd4ebe16777cca709affaa35d`).
  Holdout split, core scenarios, 30 dialogues per model balanced over its mapped
  templates. That is 90 dialogues and 54,000 calls (K=3, L=2, control 1), about
  766M input tokens; max context about 40k tokens. The estimate uses a
  character/4 proxy and assumes 1,000-character replies.

**Runner.** `src/reasoning_monitor/cdr_stream_runner.py` and
`scripts/run_cdr_stream.py`, tests in `tests/test_cdr_stream_runner.py`.

- The packed transcript goes in one user message, with gateway history
  disabled. Only the primary reply enters later history; an empty primary is
  replaced by `[EMPTY_PRIMARY_RESPONSE]` in history only.
- The control arm uses the frozen G1 prompt.
- Turns run sequentially within a dialogue; dialogues and models run in
  parallel, and primaries are scheduled first.
- A sliding-window limiter keeps estimated tokens per minute below a set value
  (600k, 80% of the documented 750k).
- Single-writer checkpoint. Successes are never repeated; failures are retried
  on rerun; there is a circuit breaker and partial-line repair.
- Messages are not stored. Each row keeps `messages_sha256`, and the tests
  rebuild every request from recorded primaries.

**Throughput.** At 600k tokens per minute the main run needs at least about
21 hours of wall time (766M / 600k). The pilot needs at least about 45 minutes.

**Cost.** At public list prices (approximation; ASU prices unknown) the main
run is roughly USD 100–120, above the documented USD 100/month staff-project
default. A limit increase must be in place before the main run.

## 14. Technical pilot gate (frozen before execution, 2026-10-03)

The investigator authorized the 1,800-call technical pilot on 2026-10-03. The
gate is technical only. Accuracy, history effects, metric trends and warning
signals are not read, and they may not be used to change the design. The
pilot dialogues are never part of the main sample.

The pilot passes if all of the following hold for every model:

1. All 1,800 calls complete after reruns (resume allowed).
2. Contract validity >= 0.90, in the dialogue arm and in the control arm.
3. Zero truncated responses at the 2,048-token ceiling.
4. No context-limit errors up to turn 100.
5. K-repeat raw-text diversity >= 0.80 (share of turns whose three same-context
   responses are not all identical).

Measured and reported, but not gating: median and maximum output tokens, empty
primaries, latency, calls/min and the real-token budget for the main run.

If the gate fails, the cause is diagnosed from technical fields only. Any
correction (for example the token ceiling or the worker count) is frozen before
a fresh pilot. It does not change items, mappings, prompts or the analysis
plan.

## 15. Technical pilot result (2026-10-03): gate failed on Llama-4 K diversity

The pilot ran 1,800/1,800 calls with 0 failed attempts and was relocked.
Records SHA-256 `e227cf8a903bba755a32bec4d371afa8c7f3709a5e552afacaa68eff362f6ef6`.

| Gate item | GPT-4o-mini | Llama-4 Scout | Gemma 3 12B |
| --- | ---: | ---: | ---: |
| Contract validity, dialogue / control | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Truncated | 0 | 0 | 0 |
| Context errors up to turn 100 | none | none | none |
| K-repeat raw-text diversity (>= 0.80) | 0.98 | **0.38** | 1.00 |
| Max prompt tokens (char/4) | 23.0k | 30.0k | 21.7k |
| Median / max output tokens | 156 / 311 | 221 / 542 | 193 / 561 |

**The gate fails for Llama-4 Scout only.**

Technical diagnosis, without reading accuracy:

- In 62/100 turns all three same-context responses were identical, although
  every call had a distinct seed.
- The repeat latency of identical turns was normal (median 3.8 s), so the
  identical responses are not a cache.
- Identity is most frequent in the first third of the dialogue (27/33 turns).
- Gemma 3 on the same Bedrock gateway had 0/100 identical turns, so
  temperature is forwarded.
- This matches Sections 64/66: Llama-4 Scout on `aws/` produces near-deterministic
  output at temperature 0.7. CS and SS would be degenerate for Llama.

Under §14, a correction must be frozen before a fresh pilot. It may not touch
items, mappings, prompts or the analysis plan.

**Real-token projection for the main run.** Pilot input was 0.62–0.79 of the
character/4 estimate. The main run is projected at about 537M input and 10.5M
output tokens (previous estimate 766M), with at least about 15 h of wall time
at 600k tokens per minute.

## 16. Option A: K-diversity probe for the ASU-hosted Llama-4 Scout (prepared, locked)

The investigator chose option A: test `asu-air/llama4_scout-17b` (the same
weights, served on ASU infrastructure) before any recalibration.

- Config: `configs/cdr_stream_v1_kdiv_probe_asu.yaml`. Script:
  `scripts/probe_cdr_stream_k_diversity.py`.
- The probe replays the identical pilot prompts of the Llama dialogue at turns
  1, 4, ..., 88 (30 turns). Histories are rebuilt from the recorded pilot
  primaries and verified against the recorded request hashes. Each prompt gets
  K=3 calls with distinct seeds: 90 calls.
- The aws baseline comes from the repeats already recorded in the pilot, so no
  new aws calls are needed.
- Frozen gate: candidate K diversity >= 0.80, contract validity >= 0.90, zero
  truncation, all calls complete. A one-call production probe of the candidate
  endpoint is required first.
- A pass authorizes only the preparation of a Llama G1 recalibration on this
  endpoint (same frozen items and rule) and a Llama-only fresh pilot. A failure
  leads to the investigator's choice between options C and D.

**Probe result (2026-10-03): failed; the low diversity is intrinsic to Llama-4 Scout.**

The probe ran 85/90 calls. The 5 failures were HTTP 429 project
tokens-per-minute errors: the probe script had no token limiter, and the
client's short retry backoff (1+2+4+8 s) is shorter than the 60 s window.

- Records SHA-256 `fd924273f12095234064f37a524cd1f681d413311d75a6c7061262d65e6bdf4e`.
- Report `fca1324946a31f8d62c2debcf4bc86ce41ac0f0fbaf6a530060caecca29017f7`.

On the 26 complete turns, `asu-air` K diversity was **0.50**, against
**0.53** for aws on the same 30 prompts. Contract validity was 1.000 and
nothing was truncated. Even if the 4 incomplete turns were all diverse, the
maximum would be 17/30 = 0.57 < 0.80, so the gate fails regardless and the
missing calls are not completed. The serving stack is not the cause: Llama-4
Scout itself is near-deterministic at temperature 0.7 on these prompts. The
config is relocked.

Engineering lesson for the main run: a 429 must back off for about 60 s (the
documented guidance). The stream runner already limits estimated tokens per
minute; the client's 429 retry delay will be lengthened before the main run.
This changes no output.

## 17. Option E: Llama-4 Maverick K-diversity probe (prepared, locked)

The investigator prefers to keep a Llama-family model, as the proposal
commits to one, rather than option C or D. ASU Llama options:

- Llama-3 8B and 70B have an 8k context window. Pilot prompts reach about
  40k tokens, so these models are infeasible.
- Llama-4 Maverick 17B (`aws/llama4_maverick-17b`) has a long context window
  and is the only feasible candidate.

The probe reuses the §16 design unchanged:

- the same 30 turns;
- the same hash-verified pilot prompts;
- K=3 with distinct seeds, 90 calls;
- the same frozen gate.

Config: `configs/cdr_stream_v1_kdiv_probe_maverick_asu.yaml`, locked.

Steps before the probe can run:

- A one-call production probe of Maverick must be accepted first.
- To avoid the §16 429 losses, the probe uses 4 workers, 6 retries and a
  65 s minimum wait after HTTP 429. The client's new `rate_limit_backoff`
  parameter is opt-in with a default of 0, so historical runs are unchanged.

If the probe passes, Maverick goes through:

1. its own G1 calibration (same frozen items and rule);
2. a Maverick-only fresh pilot.

The frozen Scout pilot failure stays recorded.

**Authorization (2026-10-03).** The Maverick production probe was accepted
(`results/probe_llama4_maverick_production.json`). The investigator explicitly
authorized the 90-call Maverick K-diversity probe only. It does not cover a
Maverick G1, a pilot or the main run.

**Probe result (2026-10-03): failed.** All 90/90 calls completed. Contract
validity was 1.000 and no call was truncated, but Maverick's K diversity was
**0.633** (19/30 turns), below the 0.80 gate. On the same prompts, Scout aws
scored 0.533.

All three repeats were identical on 11 turns:

| Turns | Identical repeats |
|---|---|
| 1–33 | 5/11 |
| 34–67 | 4/12 |
| 68–100 | 2/7 |

- Records SHA-256 `b23f49dc06f0492692cd1f626d5e8fe211e1f6ae5ae3b68f99d4afbc88beab94`.
- Report `9dbe0450092abdcb40a5916e3d802f218187fc30ac78daf078bf61b0806b2a36`.

The config is relocked. No feasible ASU-hosted Llama model passes the frozen
repeat-diversity gate at temperature 0.7. The decision returns to the
investigator.

## 18. Final Llama probe: asu-air Llama-4 Maverick (authorized 2026-10-03)

The investigator decided to try `asu-air/llama4_maverick-17b` as the last ASU
Llama candidate. The authorization covers only:

- a one-call production probe;
- the 90-call K-diversity probe, which has an identical design, prompts and
  frozen gate.

Config: `configs/cdr_stream_v1_kdiv_probe_maverick_asuair_asu.yaml`.

Outcomes:

- **Pass:** this permits only a Maverick G1 calibration on this endpoint, then a
  Maverick-only pilot.
- **Failure:** the ASU Llama search closes, and no further Llama endpoint is
  probed.

**Result (2026-10-03): failed.** All 90/90 calls completed. Contract validity
was 1.000 and nothing was truncated. K diversity was **0.700** (21/30 turns),
below the 0.80 gate.

All three repeats were identical on 9 turns (4, 13, 25, 55, 61, 70, 79, 85
and 88):

| Turns | Identical repeats |
|---|---|
| 1–33 | 3 |
| 34–67 | 2 |
| 68–100 | 4 |

Artifact hashes:

- Records SHA-256 `0beb2b8d9aad10f80c30e7d6b11843f7e1eee74aab0ba8894a2f857ed525ffdd`.
- Report `d366829fc5879a62d4ad3d31d5c37bf8de7db6fda0c03f94246bd89acaac0ce8`.

The config is relocked. The ASU Llama search is closed, as decided in advance.

Summary of the four Llama K-diversity results on the same 30 prompts:

| Model and endpoint | K diversity |
|---|---|
| Scout aws | 0.533 |
| Scout asu-air | 0.50 (26 turns) |
| Maverick aws | 0.633 |
| Maverick asu-air | 0.700 |

## 19. Llama-3 context-capacity probe (authorized 2026-10-03)

The investigator asked to measure, not assume, whether `aws/llama3-70b` and
`aws/llama3-8b` can hold the packed history.

Design:

- Script: `scripts/probe_cdr_stream_context_capacity.py`.
- Config: `configs/cdr_stream_v1_context_probe_llama3_asu.yaml`.
- Pilot Llama prompts are hash-verified and sent at turns 10, 25, 40, 60, 80
  and 100. By the character/4 estimate they are about 2.5k to 30k tokens.
- At each length the probe makes two calls:
  - a "real" call that sends the pilot prompt;
  - a "needle" call that asks the model to quote the oldest user message,
    which detects silent truncation.
- This is 24 calls in total. The gateway's error text and reported prompt
  tokens are recorded.

Decision rule:

- A model is feasible only if the turn-100 real call succeeds and the turn-100
  needle is recalled.
- Feasibility permits only a production probe and the identical 90-call
  K-diversity probe. Calibration (G1) and a pilot would then follow.

**Result (2026-10-03): both models are feasible.** All 24/24 calls succeeded,
and every call finished with `stop`.

| Turn | Gateway prompt tokens (both models) |
|---|---|
| 10 | 3,085 |
| 25 | 7,572 |
| 40 | 13,013 |
| 60 | 20,950 |
| 80 | 28,912 |
| 100 | 37,139 |

- The assumed 8k context limit does not apply on the ASU gateway.
- From turn 25 onward, both models quoted the oldest user message verbatim at
  every length up to turn 100. There is no sign of silent truncation.
- All 12 real calls met the response contract.
- The turn-10 needle for 70B paraphrased instead of quoting. This is not a
  capacity failure.

Artifact hashes:

- Records SHA-256 `7820b1458ea4828c3060b22fc488aa13af4f8606786677fedf047ae942b2d560`.
- Report `a81daa5d5a1f11eb90d51acb9536a9658e33c0cb4be573851d3cd1220cafa28f`.

The config is relocked. The 8k assumption made in §17 and §18 was wrong, and
it is corrected here.

Next, the identical 90-call K-diversity probes are prepared and locked:

- `configs/cdr_stream_v1_kdiv_probe_llama3_70b_asu.yaml`;
- `configs/cdr_stream_v1_kdiv_probe_llama3_8b_asu.yaml`.

The 8B production probe already exists and was accepted. The 70B model needs
a one-call production probe first.

**Authorization (2026-10-03).** The investigator explicitly authorized:

- the one-call Llama-3 70B production probe;
- the two 90-call K-diversity probes, for 70B and 8B.

This does not cover G1, a pilot or the main run.

**Llama-3 70B result (2026-10-03): passed.** The production probe was accepted.
The K-diversity probe ran 90/90 calls:

| Gate item | Result | Threshold |
|---|---|---|
| K diversity | **1.000** (30/30 turns) | ≥ 0.80 |
| Contract validity | 0.967 | ≥ 0.90 |
| Truncated responses | 0 | 0 |

The three invalid responses (turns 49, 58 and 70) ended with `stop` at about
250 tokens. In each, the model wrote its reasoning but omitted the
`FINAL_ANSWER` line. This is a missing marker, not a ceiling hit.

- Records SHA-256 `86db54c8de9a3c45932e6d604fefe3a7086ede761619c1fbdcb5f98ea7c8d40d`.
- Report `8157fedf0354df7a22a0412a7e2786702b5079eab36c13799af25201e5902257`.

The config is relocked. The pass permits only a Llama-3 70B G1 calibration
(same frozen items and rule), then a fresh pilot.

**Llama-3 8B result (2026-10-03): failed.** The probe ran 90/90 calls.

| Gate item | Result | Threshold |
|---|---|---|
| K diversity | **0.633** | ≥ 0.80 |
| Contract validity | 1.000 | ≥ 0.90 |
| Truncated responses | 0 | 0 |

- Records SHA-256 `ea8b442b15ced05c3595e982d18aceda642a4e54fe9f761eb09c4116f07065b5`.
- Report `d964ba31b6c78bcbe3804340f3073e53771ab08bc85081acb07df4ee8bfe30c2`.

The config is relocked. Llama-3 70B is the only Llama that passes.

## 20. Llama-3 70B G1 calibration (prepared, locked)

- Config: `configs/cdr_stream_v1_g1v2llama3_70b_asu.yaml`.
- Spec: `v2llama3_70b`.
- It uses the identical frozen items (`d943ab9f…`), prompt, v2 ladder, bands
  and rule across all four templates: 1,440 primary calls, smoke first. G1b
  needs separate approval.
- Preflight passes.

`--combine` writes a NEW file, `results/cdr_stream_v1_mappings_llama3.json`:

- Llama-4 Scout is dropped from it, with its reason recorded under
  `dropped_models`.
- GPT-4o-mini and Gemma 3 come from the same source assessments, unchanged.
- The original `results/cdr_stream_v1_mappings.json` (`9a63512b…`) is never
  modified.

If 70B is retained, a new Llama-only technical pilot and a new main manifest
follow, both bound to the new mappings file.

**Authorization (2026-10-03).** The investigator explicitly authorized the
1,440 primary Llama-3 70B G1 calls only, smoke test first. G1b, the Llama
pilot and the main run stay locked.

**G1 result (2026-10-03): Llama-3 70B not retained under the frozen rule.**
All 1,440/1,440 calls completed with 0 failed attempts. Contract validity was
0.9965. One response was truncated, and the frozen operational rule
(CDR_STREAM_G1_PROTOCOL: "any response is truncated") therefore sets
`operational_pass: false` and `retained: false`. No cell needs G1b.

The truncated response:

- item `milestone_chains/L09/SHARE/01`;
- 2,048 tokens, with the gateway reporting `stop`;
- long step-by-step reasoning cut off mid-formula, not a repetition loop.

Per-template outcome, recorded for the investigator's decision and never to
be used as a retained mapping:

| Template | Status | Mapping (E/M/H) |
|---|---|---|
| budget_ledger | mapped | 3/5/10 |
| inventory_logistics | mapped | 5/9/10 |
| lab_protocol | mapped | 4/5/10 |
| milestone_chains | excluded | fitted level 1 = 0.67, below the E band |

Artifact hashes:

- Records SHA-256 `e2e9e9331acc29620b269b4b59696a48672d6bf5b3817fa5f5d20466c7afbc91`.
- Assessment `c5f1a7845fe6cfffbf8751ae4821f63a1e6d8955fb0c1606ded88224ab31f525`.

The config is relocked. `--combine` was NOT run. Retaining this model would
require relaxing the frozen truncation rule, which only the investigator can
decide, as a recorded deviation.

**Investigator deviation (2026-10-03): Llama-3 70B retained.** The investigator
decided to override the single-truncation operational failure for Llama-3 70B.
The deviation is recorded in DEVIATIONS_FROM_PROPOSAL (first entry) and in
`operational_overrides` in the G1 config.

`--combine` wrote `results/cdr_stream_v1_mappings_llama3.json`
(SHA-256 `4ab2af3d7af6deff18af72baf3cd349bd30bee133ee4185ca33c8e6efea8a68b`):

| Model | Status in the new file | Mappings |
|---|---|---|
| GPT-4o-mini | retained | identical to the original file |
| Gemma 3 12B | retained | identical to the original file |
| Llama-3 70B | retained | budget E3/M5/H10, inventory E5/M9/H10, lab E4/M5/H10 |
| Llama-4 Scout | dropped (listed under `dropped_models`) | — |

The following are unchanged:

- the original mappings file (`9a63512b…`);
- the 70B assessment (`c5f1a784…`, still `operational_pass: false`);
- the 70B records.

Next steps, which need preparation and separate approval:

- a fresh Llama-3 70B technical pilot under the unchanged §14 gate, including
  zero truncation;
- new main manifests bound to the new mappings file.

## 21. Llama-3 70B technical pilot and new manifests (prepared, locked)

`scripts/prepare_cdr_stream_llama3.py` (call-free) reads the frozen v2 bank
file and the new mappings, and writes to `results/cdr_stream_v1_prep_llama3/`.
All original preparation artifacts stay unchanged. G0 passes.

| File | SHA-256 | Contents |
|---|---|---|
| `manifest_pilot.json` | `ed839b37…` | 1 unit: the same S1/inventory_logistics/00 skeleton as Scout's pilot, with the 70B mapping E5/M9/H10; 600 calls |
| `manifest_main.json` | `5351be89…` | 90 units (30 per model), 54,000 calls, about 774M input tokens (char/4) |

The GPT-4o-mini and Gemma 3 main units are verified byte-identical to the
original main manifest. Llama-3 70B's main budget is about 271M input tokens
(char/4), with a maximum context of about 39.9k.

The pilot config is `configs/cdr_stream_v1_pilot_llama3_asu.yaml`:

- 8 workers, 600k TPM limiter, 6 retries;
- a 65 s minimum wait after HTTP 429, through the new opt-in
  `rate_limit_backoff_s`, which `run_cdr_stream.py` now forwards. Older
  configs keep 0 and their fingerprints are unchanged.

Preflight passes.

The unchanged §14 gate applies, including zero truncation and K diversity
>= 0.80. The G1 truncation override does not extend to this gate. Execution
is locked pending approval.

**Authorization (2026-10-03).** The investigator explicitly authorized the
600-call Llama-3 70B technical pilot only, under the unchanged §14 gate. The
main run stays locked.

**Pilot result (2026-10-03): the §14 gate FAILED on truncation.** The pilot ran
600/600 calls with 0 failed attempts.

| Gate item | Result | Threshold |
|---|---|---|
| Contract validity, dialogue / control | 0.99 / 1.00 | ≥ 0.90 |
| K diversity | **0.99** | ≥ 0.80 |
| Context errors | none (max prompt 33.7k char/4) | none |
| Truncated responses | **2** | 0 |

Both truncated responses hit 2,048 tokens and the gateway reported `stop`:

- turn 26, repeat 1: a self-correction loop, 59 lines with only 22 distinct, one
  line repeated 10 times;
- turn 98, paraphrase 1: a run-on loop of the form "we assume the correct
  number ...".

Neither is a primary response. Both are degenerate loops, not long valid
reasoning, so a larger ceiling would not prevent them. The 99th percentile of
output length was 1,005 tokens.

Records SHA-256 `6f28ae2158eb411e8f2e49c2c72499d8febb4bea9e5591bc40975f94d3e4b583`.

The config is relocked. The decision returns to the investigator. The G1
override explicitly did not extend to this gate.

## 22. Option 3: Llama-3 70B proceeds; main-run truncation policy (2026-10-03)

The investigator chose option 3, recorded as the second deviation in
DEVIATIONS (first entry). The failed 70B pilot gate stays failed in the record.

**Truncation policy `truncated_primary_placeholder_v1`.** It is frozen before
any main output and applies to all three models.

- **Record.** A response at the 2,048-token ceiling is kept verbatim. It is
  never retried and never imputed.
- **Primary analysis.** The response is scored as missing for every metric.
- **History.** A truncated primary enters later packed history as
  `[TRUNCATED_PRIMARY_RESPONSE]`.
- **Sensitivity analysis (prespecified).** Truncated cells are scored as
  incorrect for CQ.
- **Reporting.** Truncation counts are reported per model, scenario, regime
  and turn.

**Runner.** `acquire_streams(..., mask_truncated_primaries=True)`. On resume,
the runner rebuilds the masking from the checkpoint's `truncated` flags. A
test covers both the opted-in and default behaviour. Earlier fingerprints are
unchanged.

**Main config.** `configs/cdr_stream_v1_main_asu.yaml`:

- 90 units and 54,000 calls;
- about 774M input tokens by the char/4 estimate, or about 480–610M real
  tokens at the pilot ratio of 0.62–0.79;
- maximum context of about 39.9k tokens (char/4);
- 16 workers, a 600k TPM limiter and a 65 s wait after HTTP 429;
- fingerprint `e5a9895be395fa0020256b18e158caf6ec42417a1a261749582f74c099507cbe`.

Preflight passes and all three probes are accepted. Execution is LOCKED until
the investigator checks the ASU monthly budget and explicitly approves.

**Authorization (2026-10-03).** The investigator checked the ASU budget and
explicitly authorized the 54,000-call main acquisition under
`configs/cdr_stream_v1_main_asu.yaml` (fingerprint `e5a9895b…`).

During and after acquisition, the following must not change: the sample,
prompt, models, mappings, K/L contract, truncation policy and analysis plan.
Accuracy and effects are not read until acquisition is complete.

**Acquisition paused (2026-10-03): ASU monthly budget exhausted.**

- **Progress.** 7,287/54,000 calls (13.5%) were saved, about 2,400 per model.
  All 90 units have started.
- **Failures.** 25 failed attempts, all HTTP 429 "exhausted its monthly
  budget" (usage 100.07%). The breaker stopped the run.
- **Truncations so far.** 1 (Llama-3 70B).
- **Integrity.** The checkpoint is intact and resumable. Nothing is
  re-run or changed.

**Cost diagnosis (estimate only).** The estimate uses public list prices per
million input tokens: GPT-4o-mini $0.15, Gemma 3 12B about $0.09, Llama-3 70B
on Bedrock $2.65. Real-token factors are about 0.8 for GPT and Gemma and 1.24
for Llama-3 (from the context probe).

| Model | Spent so far | Remaining |
|---|---|---|
| Llama-3 70B | about $98 | about $790 (plus output) |
| GPT-4o-mini | about $3 | about $27 |
| Gemma 3 12B | about $1 | about $17 |
| **Total** | **about $102** (observed: $100) | **about $850–900** |

The pre-approval projection of about $100–120 was made with Llama-4 Scout,
whose price is about $0.17/M. It was not recomputed after the switch to
Llama-3 70B, which is roughly 15× more expensive per token. This was an
omission in the preparation, and it is recorded here.

The decision belongs to the investigator. The options are:

- an ASU one-time limit increase;
- waiting for the monthly reset;
- a recorded change of scope.

## 23. Analysis pipeline (call-free; protocol DRAFT)

`CDR_STREAM_ANALYSIS_PROTOCOL.md` operationalizes §5–§6 and §22. It was
written before any main-run metric was seen. The code:

- `src/reasoning_monitor/cdr_stream_analysis.py` covers scoring, H1 sign-flip
  tests with Holm, S0-null-calibrated monitoring with leave-one-out FAR across
  the three strategies and W ∈ {10, 20, 30}, H3 alarm order, H4/H5, and the
  D5 two-sided LS/SS arm.
- `scripts/analyze_cdr_stream.py` has three modes:
  - check only (the default);
  - `--synthetic`, which runs the full pipeline on synthetic records and writes
    figures to `results/cdr_stream_v1_analysis_synthetic/`;
  - `--analyze`.
- `tests/test_cdr_stream_analysis.py` (6 tests) uses synthetic data only.

`--analyze` refuses to run until both conditions hold:

- all 54,000 records exist;
- `configs/cdr_stream_v1_analysis.yaml` has `protocol_frozen: true`, set only
  after investigator approval.

One deviation is pending: G3 calibration streams were never run, so nulls come
from the main-run S0 dialogues (protocol §10).

## 24. Budget pause at 46.4% and the Stage A analysis (2026-10-04)

**Acquisition at the pause.** The ASU budget ran out again, at 25,039/54,000
calls (records `5e1b66cc…`, 25,115 lines).

- All 90 primary chains are complete, apart from one missing Llama turn-100
  primary.
- GPT-4o-mini is almost complete: 17,880/18,000.
- Llama-3 70B has about 8% of its repeats, paraphrases and control calls.
- Gemma 3 12B has primaries only.

**Decisions approved by the investigator:**

- The analysis protocol is FROZEN (sha `684fecef…`), with its §11 decisions
  as recommended.
- The Stage A/B split of protocol §12.
- A commitment, made before any result was seen, to complete Stage B
  regardless of Stage A results.
- The deviation is recorded in DEVIATIONS.

Stage A runs through `python scripts\analyze_cdr_stream.py --stage-a`. It
uses the pinned checkpoint prefix (`stage_a_lines`, `stage_a_prefix_sha256`
in `configs/cdr_stream_v1_analysis.yaml`) and writes to
`results/cdr_stream_v1_main/analysis_stage_a/`.

**Stage A result (2026-10-04).** Outputs are in
`results/cdr_stream_v1_main/analysis_stage_a/`, with hashes in `manifest.json`.

*Coverage.*

- CQ and LS are measured on 100% of turns for all models (Llama 99.9%).
- ES is measured on 96% (5-turn warm-up).
- For GPT, CS, SS and CQ_ctrl are measured on 99.2% of turns. RS covers 14.2%,
  which is the C-subset by design.
- Llama has 2 truncated primaries, scored as missing.

*Dominant observation (descriptive).* In the dialogue arm, CQ collapses with
accumulated history in every model and every scenario, including S0-E:

| Model | Turn 1 | Turns 20–40 | Turns 80–100 |
|---|---|---|---|
| GPT-4o-mini | 0.90 | 0.13 | 0.08 |
| Gemma 3 12B | 0.73 | — | 0.04 |

GPT's control arm (solver state summary) stays at 0.60–0.85 and follows the
scheduled regime.

- The dialogue arm sits near the floor from about turn 20 onward, so scheduled
  task drift is largely masked.
- The level-matched within-arm contrast (late minus early, at the same
  difficulty level) is −0.118 pooled (p < 1e-4, n = 90). It is negative in
  every scenario, including S0-E (−0.185).

*H1a.* Holm-significant post-minus-pre declines:

| Metric | Transitions |
|---|---|
| CQ | S1-a −0.094, S2 −0.072, S3-a −0.161 |
| ES | S1-a, S2, S3-a |

No CS, SS or RS test (GPT only) and no LS test survived Holm.

Caveat: H1a as frozen compares post with pre windows, so the background
context decline confounds it. An exploratory comparison at the same turns in
the S0 nulls gives CQ −0.057 at 34, −0.029 at 51 and −0.008 at 68. The drift
declines (−0.128, −0.072, −0.039) are larger, but this contrast was not
preregistered.

*H1b (GPT only).* The net contrast is positive at every transition (+0.03 to
+0.23, all p > 0.9). Because the dialogue arm is already at the floor, it
cannot decline further, while the control arm declines with task difficulty.
H1b is therefore uninformative under the floor. It is not evidence against a
context penalty, which the absolute gap shows directly: about 0.10 against
0.65.

*Monitoring (H2).*

- The null S0 dialogues themselves trend downward, so CQ-CUSUM, CQ-DDM and
  several other cells found no grid value with null FAR ≤ 0.05 (CQ is
  uncalibratable under `distribution_based` and `proposal_mapping`).
- Calibrated detectors detected essentially nothing. The best was ES-CUSUM:
  2% detection, delay 31.
- adwin_all had 0 detections and 0 FAR.

*H3.* It is not evaluable: no CQ alarms occurred and no prediction held.

*H5.* The ΔCQ–ΔES Spearman correlation is 0.86, and 74% of transitions show a
joint decline.

The truncation-as-incorrect sensitivity analysis changes nothing.

Exploratory note: GPT's CS still visibly follows the regime (for example the
S3 recovery at turn 67) while its CQ is floored.

**Interpretation for the investigator.** The CDR-Stream dialogue condition
produced the context-accumulation collapse, not the intended graded task-drift
stream. The G1 calibration and the technical pilot gate both used the
`state_summary` format or technical criteria only, so neither checked
dialogue-arm accuracy. The proposal's monitoring hypotheses (H2/H3) are
therefore not testable on the dialogue arm in Stage A. The control arm is a
candidate stream for task-drift monitoring, but it is not the frozen primary
arm.

## 25. Control-arm H2 monitoring (exploratory GPT screen, 2026-10-04)

The investigator asked to test H2 first on GPT. The new protocol
`CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md` (sha `f7783a2c…`) was frozen
before computing. It monitors the `state_summary` arm, which has no history,
so its S0 dialogues are true nulls.

- **Metrics.** CQ, LS and ES; the arm has no CS, SS or RS because K=1.
- **Calibration.** Per model, on its own 12 nulls.
- **Screening criterion, fixed before computing.** At W=20,
  `distribution_based` or `proposal_mapping` must give CQ all three of:
  - detection ≥ 0.50;
  - LOO FAR ≤ 1/12;
  - median delay ≤ 25.
- **Status.** The GPT step is exploratory, because its CQ_ctrl trajectories
  were already seen. Validation applies the same frozen rules to the
  Llama-3 70B and Gemma 3 12B control arms, which are cheap: under about $10
  of control calls, not yet acquired. That validation needs budget and
  approval.

Script: `scripts/analyze_cdr_stream_control.py` (`--screen-gpt`, `--validate`).

**GPT control-arm screen result (2026-10-04): NOT promising.** The frozen
criterion failed. Outputs are in
`results/cdr_stream_v1_control_monitoring/gpt_screen/`.

Control-arm CQ clearly follows the schedule. Means per segment (turns 1–33 /
34–50 / 51–67 / 68–100):

| Scenario | 1–33 | 34–50 | 51–67 | 68–100 |
|---|---|---|---|---|
| S0-E | 0.83 | 0.83 | 0.85 | 0.78 |
| S0-H | 0.48 | 0.44 | 0.46 | 0.43 |
| S1 | 0.80 | 0.69 | 0.70 | 0.50 |
| S2 | 0.82 | 0.78 | 0.58 | 0.50 |
| S3 | 0.81 | 0.52 | 0.54 | 0.80 |

Per-dialogue detectors calibrated to 0/12 null alarms nevertheless detected
little. At W=20, CQ-CUSUM (threshold 7.5, LOO FAR 1/12) detected 0/18:

- S1: 2 late alarms, at turns 86 and 94;
- S2: 4 late alarms, at turns 86–99;
- all other drift dialogues: misses.

W=10 gave CQ detection 4/18 (median delay 30), and W=30 gave 1/18.

Other results:

- LS: Page-Hinkley 4/18 at W=20 (delay 32.5), better than `adwin_all` by
  +0.22 (CI 0.06–0.44).
- ES: 0/18.
- `adwin_all`: 0 detections for every metric.
- Order: LS alarmed before CQ in 2/6 S1 and 3/6 S2 dialogues. This is
  descriptive only.

**Exploratory reading, not a finding.** Single 100-turn dialogue streams of
Bernoulli CQ are low-powered once thresholds are calibrated on only 12 nulls.
Two further factors push alarms past the 33-turn detection span:

- the detector warm-up (20 values after the window fills);
- warm-up references that straddle the turn-34 change.

The screen did not pass, so the protocol does not require the Llama/Gemma
validation spend.

## 26. Power analysis and ADWIN defect (2026-10-04)

The results are in `CDR_STREAM_STAGE_A_REPORT.md` and
`results/cdr_stream_v1_power/`.

**Power analysis.**

- A single dialogue stream needs an accuracy drop of at least 0.4–0.5 for 80%
  detection within 33 turns at FAR 0.05. Aggregating 6 dialogues per turn
  reduces this to 0.2, and 18 dialogues reduce it to 0.1.
- The number of nulls (12 versus 2,000) barely matters.

**ADWIN defect.** `detectors.ADWIN` cannot alarm on streams of 100 or fewer
values in [0, 1], because its bound lacks the variance term. The `adwin_all`
H2 baseline is therefore non-functional. The detector was NOT modified; a fix
needs a recorded protocol change and a check of earlier evidence that used
ADWIN.

## 27. Per-dialogue early-signal analysis, GPT control arm (EXPLORATORY, 2026-10-04)

The rule was fixed before computing in
`CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md`. The script is
`scripts/control_early_signal.py`, and outputs are in
`results/cdr_stream_v1_control_monitoring/gpt_screen/gpt4o_mini/early_signal/`.

The rule:

- the baseline is turns 1–30;
- the threshold is μ₀ − 2σ₀/√W;
- a signal needs 3 consecutive turns below the threshold;
- all 18 drift dialogues are used, with the 12 S0 dialogues as nulls.

Results at W=20:

| Metric | Drift dialogues signalling | Timing | S0 (null) dialogues signalling |
|---|---|---|---|
| CQ | 18/18 | median 16 turns after the scheduled change | 3/12 |
| LS | 10/18 | before CQ in 7/18, median lead −3 (range −21 … +18); after CQ in 2; same turn in 1 | 4/12 |
| ES | 14/18 | before CQ in 5; after CQ in 8 (it is CQ-derived) | 3/12 |

W=10 and W=30 give similar results; at W=30 the median LS lead is −9 (7/18
before CQ).

Reading: LS's signal rate in drift dialogues (10/18) is barely above its null
rate (4/12). Its "before CQ" cases are therefore not distinguishable from
noise in this sample. This is not evidence of an early warning; it is a
candidate to test, unchanged, on the unseen Llama and Gemma control arms.

## 28. Aggregated monitoring of 6 dialogues per turn, GPT control arm (EXPLORATORY, 2026-10-04)

The rules are fixed in addendum Part 2. The script is
`scripts/control_aggregated_monitoring.py`, and outputs are in
`.../gpt_screen/gpt4o_mini/aggregated/`. Thresholds were calibrated on 2,000
permuted S0-E streams (null FAR 0.00–0.04).

**Detection.** CQ-CUSUM detected every drift stream at every W, with no alarm
on the real S0-E or S0-H aggregated streams.

| W | S2 (change at 51) | S3 (change at 34) | S1 (M→H at 68) |
|---|---|---|---|
| 10 | +13 | +15 | +3 |
| 20 | +16 | +20 | +3 |
| 30 | +21 | +25 | +10 |

Page-Hinkley detected all three as well. No detector caught S1's first step
(E→M at 34, δ ≈ 0.1) before turn 68. ADWIN never alarmed (known defect), EWMA
rarely, and DDM was slower.

**LS versus CQ, same detector type.** LS led by 1–5 turns on S1 and was
simultaneous on S2. On S3 it lagged by 9–13 turns. ES mostly tracked CQ.

**Reading.**

- Aggregation turns per-dialogue failure (0/18 timely at W=20) into timely
  detection of the 0.2–0.3 drops, consistent with the power simulation.
- There is no consistent early warning from LS or ES.
- n = 3 streams, from data that were already seen.

## 29. ADWIN fix (2026-10-04)

The opt-in detector `adwin_v` and the strategies `proposal_mapping_v` and
`adwin_all_v` were added, with tests and a DEVIATIONS entry. The historical
`adwin` is unchanged.

The corrected ADWIN is faithful but needs far more than 100 samples. On
CDR-Stream drops (δ ≤ 0.5) it detected nothing within 33 turns, at every W
and every m, with a true FAR of 0. It also raised no alarm on the GPT
aggregated control streams.

See `CDR_STREAM_STAGE_A_REPORT.md` §4a.

## 30. Control-arm completion prepared (2026-10-04, call-free)

The investigator approved `MERGED_PAPER_CLAIMS.md` and
`CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md`; the protocol is frozen (sha
`d9362b9f…`).

**Runner change.** `acquire_streams(..., arms=...)` and
`run_cdr_stream.py --arms control` submit only the selected arms.

- Seeds, request content, fingerprint (`e5a9895b…`) and the checkpoint format
  are unchanged.
- A control-only filter opens every turn, because control calls never depend
  on history.
- A later run without the filter completes the remaining calls without
  repeating any control call. This is tested.

**Analysis.** `scripts/analyze_cdr_stream_control_completion.py` covers:

- H-C1 (turns 1–100; Holm over Llama-3 70B and Gemma 3);
- H-C1b (turns 21–100);
- task validity;
- figures;
- the frozen monitoring validations, run unchanged.

It refuses to run until both models have all 3,000 control calls.

For the already-seen GPT data, the same statistic gives a gap of +0.52 (turns
1–100), as a code check.

Paid acquisition (about $5) needs separate explicit approval.

**CRL gate (post-hoc, 2026-10-04): failed.**

- The last pre-final readiness gap alone has AUROC 0.672 (p=0.0015).
- CRL has AUROC 0.627.
- Within strata of the last gap, CRL has AUROC 0.468 (p=0.67).

CRL reduces to persistence and is not a main contribution
(`MERGED_PAPER_CLAIMS.md`).
