# CDR-Stream G1: Single-Turn Difficulty Calibration Protocol

Status: **prepared call-free (2026-10-03); paid execution locked.** The items,
prompt, rule and runner below are fixed and hash-recorded. Running G1 (2,880
calls) needs an explicit investigator approval recorded in
`configs/cdr_stream_v1_g1_asu.yaml` and AGENTS.md. The conditional G1b needs a
separate approval. Design context: `CDR_STREAM_DESIGN.md`, Sections 2.2b and 7.

## Purpose

Before any stream exists, G1 measures how single-turn accuracy falls along the
difficulty ladder, for every model and template. A frozen rule then maps each
(model, template) to its own easy, medium and hard centre levels. This
prevents ceiling and floor effects: every retained model operates at matched
expected accuracy. G1 is not an effect test of history or drift.

## Items

- File: `configs/selections/cdr_stream_v1_g1_items.json`, SHA-256
  `f8ea8b9329c10241bfaed9032eb8d5566123324737fd0777e1ece8d5c9a309da`.
- 1,200 items: 4 templates x 10 levels x 30 items, i.e. 5 per query family.
  - Levels 1–8 (960 items) are the primary ladder.
  - Levels 9–10 (240 items) are contingency items used only by G1b.
- Each item is a fresh state reached by a seeded random walk of 0–99 events in a
  separate calibration namespace (seed 20261005). No bank dialogue, and in
  particular no holdout dialogue, is used.
- Call-free checks: every turn text contains exactly the specification's
  numbers; gold is a non-negative integer; no item has fewer events than
  planned; cell sizes are equal; generation is deterministic.

## Prompt (shared with the CDR-Stream control arm)

The system message and response contract are in `cdr_stream_g1.py`; the prompt
SHA-256 is `717fd4b12b77e84d6173267aecddcd2df2c9cc04908587133eec97a9711947de`.
The user message is the verified record followed by the new message and the
contract: at most eight short numbered reasoning lines, then
`FINAL_ANSWER: <integer>`. Parsing uses the frozen marker-anchored LIC
semantic parser. A missing or invalid marker counts as incorrect.

## Execution

- Models and endpoints: GPT-4o-mini, Llama-4 Scout and Mistral-7B through ASU,
  with history, search and prompt enhancement disabled (production probes
  verified by preflight).
- Sampling: temperature 0.7, one call per item, 2,048-token ceiling; the per-call
  seed is derived from the fingerprint and sent to the gateway.
- Call order is shuffled per model with seed 20261006, so gateway drift during
  the run cannot pile up on particular levels.
- The runner is resumable: every raw response is appended to
  `results/cdr_stream_v1_g1/data/records.jsonl` before it is used. A changed
  request on resume is an error. Progress prints count, percentage, calls/minute
  and ETA.
- Parallelism: 8 concurrent API calls (`workers: 8`; ASU permits 8). Only the
  main thread writes the checkpoint, counts failures and runs the breaker. Seeds
  depend on the call key, not on completion order, and the worker count is not
  part of the fingerprint. Tests confirm that parallel and sequential runs
  produce identical records.
- Failure handling:
  - A successful call is never repeated.
  - A call that still fails after the client's four transient-error retries
    (`ok=false`) stays in the file for audit, is not counted as completed, and is
    retried on the next run of the same command.
  - Ten consecutive failures stop the run (circuit breaker), so a gateway outage
    cannot fill the plan with failures.
  - An interrupted final line is moved to `records.partial.jsonl`.
  - Model responses that arrive successfully but are empty or malformed are real
    model behaviour; they are kept and never retried.
  - `--assess` uses only successful records and refuses to run while any
    primary call is still missing.
- Budget (character/4 proxy): 2,880 calls, about 0.83 million input tokens,
  output ceiling 5.9 million tokens. G1b adds at most 720 calls (all 12 cells,
  2 levels x 30).

## Frozen assessment rule (per model and template)

1. CQ per level, with invalid answers counted as 0.
2. Fit a non-increasing isotonic curve (pool-adjacent-violators, weighted by n).
3. Trend gate: the one-sided Cochran–Armitage test for decreasing CQ must give
   p <= 0.01; otherwise the cell is excluded.
4. Mapping: choose ordered levels e < m < h on the fitted curve, subject to:
   - each centre lies inside its band: E 0.75–0.92, M 0.60–0.78, H 0.35–0.65;
   - E–M and M–H gaps are each at least 0.10;
   - the E–H gap is at least 0.25.

   Among qualifying triples, minimize the squared distance to the targets
   (0.85, 0.70, 0.50); ties go to the lower levels.
5. **Ceiling contingency (G1b).** If no mapping exists, the trend gate passed,
   and fitted CQ at level 8 is still above 0.65, the cell is marked
   `needs_G1b`. Only these cells receive the contingency items at levels 9–10.
   The rule is then reapplied once to levels 1–10. If it still fails, the cell
   is excluded. The ladder is never extended further.
6. **Exclusion.** A model is dropped if any of the following holds:
   - its contract validity is below 0.90;
   - any response is truncated;
   - more than 2 of its 4 templates are excluded.
7. The resulting mappings are frozen before any stream is instantiated and are
   never revised after stream results are seen.

## Outputs

- `assessment.json`: per-cell raw and fitted CQ, trend p, mapping, status and
  model retention.
- `g1_calibration.png` / `.pdf`: CQ by level per model and template, with
  bands and mappings marked.

## Commands (Windows Miniconda Prompt)

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\run_cdr_stream_g1.py --preflight
REM Only after explicit approval is recorded in the config:
set ASU_API_KEY=...
python scripts\run_cdr_stream_g1.py --execute
python scripts\run_cdr_stream_g1.py --assess
```

## Observed result (2026-10-03): G1 v1 calibration failed for the three-model design

The run completed, the assessment applied the frozen rule unchanged, and the
config was relocked (`execution_authorized: false`).

- 2,880/2,880 calls; 0 failed attempts; 8 workers; about 114 calls/min.
- Records SHA-256 `f9d882c640a6604e8bae87cb4857157d237659b6bb21b0427507f850beef2be2`.
- Assessment SHA-256 `1f419c5663c5e09e46a87c139060783ac62c6fd51a74589cabadbaf02dda409c`.
- Figure: `results/cdr_stream_v1_g1/g1_calibration.png` / `.pdf`.

| Model | Contract validity | Truncated | Mapped templates | Retained |
| --- | ---: | ---: | --- | --- |
| GPT-4o-mini | 1.000 | 0 | none (4/4 excluded) | no |
| Llama-4 Scout | 1.000 | 0 | budget_ledger (E4/M6/H7), lab_protocol (E1/M4/H8) | yes (2 excluded) |
| Mistral-7B | 0.907 | 3 | none | no (operational failure, and floor: CQ <= 0.40 at every level) |

Every trend test was significant. G1b was not triggered, because no cell hit a
ceiling. The failure is the **opposite of a ceiling**:

- **GPT-4o-mini.** Accuracy fell in a cliff of 0.3–0.4 between levels 3–4 and
  level 5. No level landed in the medium band:
  - budget_ledger: fitted 0.80 -> 0.60 -> 0.33, so no hard level in 0.35–0.65;
  - inventory_logistics: fitted 0.78 is just above the 0.78 medium bound;
  - lab_protocol: fitted 0.84 -> 0.53, so no medium level;
  - schedule_constraints: 0.47 already at level 1.

The frozen outcome is preserved. G1 v1 does not support the three-model,
four-template CDR-Stream design.

## Call-free diagnosis (post hoc; not part of the frozen rule)

Diagnosis from the saved records only:

1. **Instrument defect: degenerate single-entity MAX and COUNT_ABOVE.** At
   level 1 these ask, for example, "Among the Maintenance line, what is the
   highest remaining amount?" Models reasonably read this as a question about
   all lines. Inspected failures had correct gold; the wording is the defect.
2. **Instrument defect: schedule_constraints is too hard at every level.** The
   summary gives durations and prerequisites only, so even a level-1 query
   needs a full longest-path computation over a 10-task DAG. Inspected errors
   are genuine DAG-propagation mistakes.
3. **Ladder too coarse.** Accuracy is driven mainly by the number of query terms
   in the aggregate families (SUM, DIFF, SHARE, WHATIF). For GPT over the three
   non-schedule templates, CQ was 0.80 / 0.88 / 0.73 / 0.37 / 0.40 / 0.26 /
   0.27 at 1 / 2 / 3 / 4 / 5 / 6 / 8 terms. COUNT_ABOVE and MAX stay easy. The
   ladder changes several axes at once, so it jumps over the medium band.
4. **Mistral-7B** is at the floor on every template, with 9.3% invalid
   contracts. This is consistent with Sections 47/62 and with LIC-RQ-v2 (dialogue
   final success 0.063).

Any successor (G1 v2) must be a new, prospectively frozen calibration on new
items. Neither the v1 rule nor its bands may be relaxed to rescue v1. The draft
stream bank and holdout remain untouched.

## G1 v2 (prepared call-free 2026-10-03; execution locked)

G1 v2 is the prospectively frozen successor to the failed G1 v1. It changes only
the measuring instrument. The following are **identical to v1** and none was
relaxed: prompt (SHA-256 `717fd4b1…`), targets, bands, adjacent and E–H gaps,
trend test, G1b trigger, contract-validity and truncation rules, and the
template-exclusion limit.

| | v1 | v2 |
| --- | --- | --- |
| Primary ladder | 8 levels, several axes change per step | 12 levels, exactly one axis changes per step (tested) |
| Minimum query terms | 1 (degenerate single-entity MAX/COUNT) | 2 |
| Highest primary level | 8 terms, 3 events, 4 distractors, complexity 2 | 6 terms, 3 events, 3 distractors, complexity 2 |
| Contingency (G1b) | levels 9–10 | levels 13–14 = (8,3,4,2), (10,4,5,2) |
| schedule_constraints record | durations + prerequisites | also states each task's verified earliest finish day |
| Items per (template, level) | 30 | 30 |
| Item seed | 20261005 | 20261007 (new items) |

v2 ladder (terms, events, distractors, complexity):

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spec | 2,1,0,0 | 2,1,1,0 | 2,2,1,0 | 3,2,1,0 | 3,2,1,1 | 3,2,2,1 | 4,2,2,1 | 4,3,2,1 | 5,3,2,1 | 5,3,2,2 | 6,3,2,2 | 6,3,3,2 |

The SUM threshold filter applies when complexity is 2. This is identical to v1
behaviour, where levels >= 7 were exactly the complexity-2 levels.

- Items: `configs/selections/cdr_stream_v1_g1v2_items.json`, SHA-256
  `d8cb886ef69a0b81f2d78951b625322c43a00a21ca3a783d844dfb9d168790a6`. There are
  1,680 items: 1,440 primary and 240 contingency.
- Config: `configs/cdr_stream_v1_g1v2_asu.yaml`. Fingerprint
  `f3cbfdd4a1879c030c8f3d92f0ccf0e27063e48b2a973a57da3bedb7ac219d1b`. Order seed
  20261008, 8 workers.
- Budget: 4,320 calls, about 1.34 million input tokens. G1b adds at most 720 calls.
- Preflight passes. Versioning leaves v1 intact: the v1 items hash, the
  stream-bank hash and the v1 assessment are reproduced exactly (tested).
- **Pre-stated Mistral rule (option a):** if `mistral_7b` is not retained under
  the frozen rule in G1 v2, it is replaced by a separately approved substitute
  model (option b). No further Mistral prompt or ladder work is done.
- If GPT-4o-mini or Llama-4 Scout fails again, the result is reported as such.
  A G1 v3 would again need new items and a new approval.

```bat
python scripts\run_cdr_stream_g1.py --config configs\cdr_stream_v1_g1v2_asu.yaml --preflight
REM Only after explicit approval is recorded in the v2 config:
python scripts\run_cdr_stream_g1.py --config configs\cdr_stream_v1_g1v2_asu.yaml --execute
python scripts\run_cdr_stream_g1.py --config configs\cdr_stream_v1_g1v2_asu.yaml --assess
```

**Authorization and staging (2026-10-03).** The investigator explicitly
authorized the 4,320-call G1 v2 primary run. G1b stays locked. At the
investigator's request, a technical smoke comes first: `--smoke` runs one
primary item per (model, template, level), i.e. 144 calls, with the same keys,
messages and seeds as the full plan. It reports only technical health (API
success, contract validity, truncation, output length, latency) and deliberately
reports no accuracy, so the ladder cannot be adjusted to observed results. The
following `--execute` reuses the smoke records (tested) and completes the plan.

## G1 v2 observed result (2026-10-03): two models calibrated, Mistral fails over to replacement

The run completed after the technical smoke and was relocked. The frozen rule
was applied unchanged.

- 4,320/4,320 calls, 8 workers.
- Records SHA-256 `869790ded249b279f90373745cd17ba3d428a24ed307ed9db0acc3dbfd90d9a0`.
- Assessment SHA-256 `a5e35e05db6e5a7b6e7dc2c07b3c09865cb43ac062d0bf224bab1b87f6ac36e2`.
- Figure: `results/cdr_stream_v1_g1v2/g1_calibration.png` / `.pdf`.

| Model | Validity | Truncated | budget_ledger | inventory_logistics | lab_protocol | schedule_constraints | Retained |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| GPT-4o-mini | 1.000 | 0 | **E1/M4/H6** | excluded | **E3/M4/H5** | excluded | yes |
| Llama-4 Scout | 1.000 | 0 | excluded | **E4/M10/H12** | **E4/M5/H10** | excluded | yes |
| Mistral-7B | 0.914 | 5 | excluded | excluded | excluded | excluded | no |

All trend tests were significant (p <= 6e-4). No cell hit the ceiling, so G1b
was not triggered.

Why the excluded cells failed:

- **schedule_constraints:** excluded for every model in both versions. Even with
  stated finish days, level-1 single-turn CQ is only 0.55 (GPT) and 0.63 (Llama).
- **GPT inventory_logistics:** no medium level at least 0.10 below the easy
  level. The fitted curve goes 0.77 / 0.70 / 0.57.
- **Llama budget_ledger:** no easy level inside 0.75–0.92. The fitted curve
  goes 0.97 -> 0.83.

Mistral-7B was not retained: it truncated 5 responses and sat at the floor
(CQ <= 0.37) on every template. Under the pre-stated rule (option a ->
option b), it is to be replaced by a separately approved substitute model,
which needs its own G1 v2 calibration on the same frozen items. No further
Mistral work.

The mapped (model, template) cells are now frozen for stream instantiation:
GPT on budget and lab, Llama on inventory and lab. They must not be revised
after any stream result.

## G1 v2 extension (prepared call-free 2026-10-03; execution locked)

The investigator made three decisions after G1 v2:

- Mistral-7B is replaced by **Claude 4.5 Sonnet** (`aws/claude4_5_sonnet`; ASU
  marks it "Experimental"; 200k context). This applies the pre-stated rule.
- **Option 2b:** the v2 schedule template is replaced by a shallower logic
  template, `milestone_chains`. It has ten milestones in independent chains of
  lengths 3, 3, 2 and 2, at most one prerequisite each, so changes propagate at
  most two steps. The record states every finish day.
- **Option 3b:** 30 stream dialogues per model.

Calibration plan (config `configs/cdr_stream_v1_g1v2ext_asu.yaml`):

- **Unchanged:** prompt, v2 ladder, targets, bands, gaps, trend test and
  exclusion rule.
- **Claude** is calibrated on budget, inventory and lab (the frozen v2 items:
  1,260 items regenerated bit-identically and checked against the v2 file) plus
  milestone_chains: 1,440 primary calls. `send_seed: false`, because Anthropic
  models on Bedrock take no sampling seed.
- **GPT-4o-mini and Llama-4 Scout** are calibrated on milestone_chains only:
  360 calls each.
- Items: `configs/selections/cdr_stream_v1_g1v2ext_items.json`, SHA-256
  `d943ab9f0263bb815f1874119b94164354751be807657ddec7e178ff014dbed8`.
- Fingerprint `e986892d10410b8103515b1d7ad4c2a61eb7c0724ab68e1bb77a186364783ab0`.
- Budget: 2,160 calls, about 0.70 million input tokens. G1b (levels 13–14) is
  likely to be needed for Claude ceiling cells (at most 4 cells x 60 calls) and
  needs its own approval.
- `--combine` merges the v2 and extension assessments into the final
  per-(model, template) mappings over {budget, inventory, lab,
  milestone_chains}, in `results/cdr_stream_v1_mappings.json`. The unchanged
  retention rule allows at most 2 of 4 templates to be excluded.
- **Before any Claude call**, a one-call production probe must show that the
  gateway accepts the disabled history/search/enhancement options. Preflight
  fails until the probe exists.
- Versioning leaves v1/v2 untouched: v1/v2 fingerprints, v1/v2 item hashes and
  the bank hash are unchanged (tested).

## Model change after the Claude 4.5 technical smoke (2026-10-03)

The technical smoke for the extension (72 calls) passed for GPT-4o-mini and
Llama-4 Scout: contract validity 1.000, no truncation. Claude 4.5 Sonnet's
contract validity was 0.854, and all 7 invalid responses were on
`lab_protocol`. Four responses were empty (1 token) and three were cut
mid-sentence, all with `finish_reason=stop`. Claude's other three templates
were 100% valid. This pattern is consistent with a gateway or model safety
classifier reacting to chemistry terms. Accuracy was not inspected.

Investigator decisions:

- **Third core model: Gemma 4 31B IT** (`asu-air/gemma4_31b_it`). It is
  open-weight, from a third model family (Google, alongside OpenAI and Meta),
  ASU-hosted outside Bedrock, and not a reasoning model. The per-call seed is
  sent (`send_seed: true`). If the gateway rejects it, the smoke trips the
  breaker and the setting is changed to false. Gemma is calibrated on all four
  final templates.
- **Frontier external validation: Claude Sonnet 5** (`aws/claude5_sonnet`), the
  model the proposal names for this role. It is calibrated separately
  (`configs/cdr_stream_v1_g1v2frontier_asu.yaml`; 1,080 calls; fingerprint
  `760aeee6…`). It runs on budget, inventory and milestone_chains with the same
  frozen items, prompt and rule. `lab_protocol` is a declared exclusion recorded
  before any accuracy was seen. Its production probe (2026-09-18) is accepted.
  Execution is locked pending approval.
- The Claude 4.5 smoke records are preserved in
  `results/cdr_stream_v1_g1v2ext/superseded_claude45_smoke/` (records SHA-256
  `f956e00ca592af44c7c40beb3b2bc9e95a83cdfe3c38c155b40a3416c4b37a5f`).
- The extension config now has fingerprint
  `1d45c4b16396f93023457d3428152a1f71becbf1eea7affc2224dbe027e3a724` and still
  2,160 calls. It needs the one-call Gemma production probe before its smoke.
  The investigator's instruction to continue with Gemma authorizes the probe
  and the 2,160 calls, smoke first.

## G1 v2-ext observed result (2026-10-03)

Gemma technical smoke: 72/72 calls, contract validity 1.000 for every model,
no truncation, and the per-call seed was accepted. The full run used 16
workers (about 163 calls/min, versus about 63 with 8): 2,160/2,160 calls, 0
failed attempts, 0 truncated, contract validity 1.000. The primary run is
relocked.

- Records SHA-256 `a25f39163e0957bdccdf4be3f51ebcb36c0b72b9733cf51d3af907fd884824fb`.
- Assessment `299845aa91ae82d3ea7bb80507fd663cb513e3b127621a46949993e1ae77f2c2`.
- Combined mappings `f046d2636f23d4ddce8ae1c524209628d062b36dd5e42b3b68c676b4e6526f0e`
  (`results/cdr_stream_v1_mappings.json`).

| Model | budget | inventory | lab | milestone_chains | Retained |
| --- | --- | --- | --- | --- | --- |
| GPT-4o-mini | E1/M4/H6 | excluded | E3/M4/H5 | **E1/M3/H8** | yes |
| Llama-4 Scout | excluded | E4/M10/H12 | E4/M5/H10 | **E1/M3/H10** | yes |
| Gemma 4 31B | excluded (trend p=0.016 > 0.01) | needs G1b | needs G1b | needs G1b | pending G1b |

**Gemma hits the ceiling.** At level 12 its fitted CQ is 0.77 (inventory), 0.76
(lab), 0.77 (milestone_chains) and 0.92 (budget). None of these reaches the hard
band.

Under the unchanged frozen rule:
- The three cells with a significant trend are routed to G1b, the contingency
  levels 13–14 defined before any model output. `g1b_cells` was written
  mechanically from `assessment.needs_G1b`.
- budget_ledger is nearly flat (0.92–0.99) and fails the trend gate, so it is
  excluded rather than routed to G1b, exactly as the rule specifies.

G1b adds 180 calls (3 cells x 2 levels x 30 items) and needs explicit approval.
Afterwards the rule is reapplied once to levels 1–14. A cell that still fails is
excluded. The ladder is not extended again.

## G1b result (2026-10-03): Gemma 4 31B not retained (ceiling)

G1b ran 180/180 calls with 0 failures, no truncation and validity 1.000, and
was relocked.

- Records SHA-256 `3f11e758474df16d0b148917256c81cea8411c1c633a749da82aac8467f1c169`.
- Assessment `f89440494113ea3674807ba994c0d33c99a4487f3158e707930c057c91d72519`.
- Mappings `0d1954cbc847d194f858eecfcb3e4754cc7a92ab142eb55295183fa136559fb7`.

The rule was reapplied once to levels 1–14. Gemma's fitted CQ at the hardest
level was 0.80 (inventory), 0.67 (lab) and 0.73 (milestone_chains). Lab and
chains had levels inside the easy and medium bands, but **no level reached the
hard band (0.35–0.65) in any template**, so all four templates are excluded and
Gemma is not retained. As pre-stated, the ladder is not extended further.

This is a capability finding, not a format failure. On this task family Gemma 4
31B sits clearly above GPT-4o-mini and Llama-4 Scout: its single-turn CQ stays
>= 0.67 even at 10 query terms with 5 events and 6 distractors.

Final core mappings (`results/cdr_stream_v1_mappings.json`):

| Model | budget | inventory | lab | milestone_chains |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | E1/M4/H6 | — | E3/M4/H5 | E1/M3/H8 |
| Llama-4 Scout | — | E4/M10/H12 | E4/M5/H10 | E1/M3/H10 |

Implication for the frontier role: a model stronger than Gemma 4 (Claude Sonnet
5) is very likely to hit the same ceiling under the matched-accuracy rule. The
frontier calibration (1,080 calls) should therefore not be run until its
design is decided prospectively.

## Third core model, attempt 3: Gemma 3 12B IT (prepared 2026-10-03)

The investigator selected `aws/gemma3_12b_it` (open weights, Google family,
128k context) to replace Gemma 4 31B, which hit the ceiling. The choice uses
calibration data only, because no stream result exists yet.

- Config: `configs/cdr_stream_v1_g1v2gemma3_asu.yaml`, fingerprint
  `cfbaeb78c08a00751fd37bfdb690dab746ff4e25f7063d715fcc72ec9906085a`.
- Plan: 1,440 calls on the same frozen items (all four final templates), with
  the identical rule. `send_seed: true`.
- Authorized: the one-call probe and the 1,440 primary calls, smoke first.
  G1b needs its own approval.
- `--combine` merges the v2, v2-ext and Gemma 3 assessments into
  `results/cdr_stream_v1_mappings.json`. The previous mappings hash
  `0d1954cb…` is recorded above.

## Gemma 3 12B result (2026-10-03): retained; G1 calibration complete

The technical smoke (48 calls) had validity 1.000, no truncation, and the seed
was accepted. The full run used 16 workers: 1,440/1,440 calls, 0 failures,
validity 1.000. The config is relocked.

- Records SHA-256 `3ce8fddc8ea828da3fa3088f419b54c8c42e1eccfc6c63d74bb565b39c62c3a0`.
- Assessment `4e08bba21c2ea715578c8d0d60eb4d980f4a41439bb53a90a272f22223c2bec4`.

Gemma 3 mapped on budget (E1/M4/H8) and inventory (E2/M4/H8). Lab and
milestone_chains were excluded: both curves drop straight past the medium band
(lab 0.80 -> 0.58; chains 0.77 -> 0.53). With 2/4 templates excluded, Gemma 3
is retained under the unchanged rule. No ceiling, so no G1b.

**Final frozen core mappings** (`results/cdr_stream_v1_mappings.json`, SHA-256
`9a63512bde255b96394d3dc31ca365daf57644ee5e38e0f80b95b9a867db6571`). These must
not be revised after any stream result.

| Model | budget | inventory | lab | milestone_chains |
| --- | --- | --- | --- | --- |
| GPT-4o-mini | E1/M4/H6 | — | E3/M4/H5 | E1/M3/H8 |
| Llama-4 Scout | — | E4/M10/H12 | E4/M5/H10 | E1/M3/H10 |
| Gemma 3 12B | E1/M4/H8 | E2/M4/H8 | — | — |

Not retained: Mistral-7B (floor and format), Gemma 4 31B (ceiling). Claude 4.5
Sonnet was superseded after the lab classifier issue. Claude Sonnet 5 (frontier
validation) has not been run and its design is pending.

Coverage: every template is mapped for exactly two of the three core models;
no template is mapped for all three. Templates therefore enter the stream
study as a crossed random factor, and every core model contributes 2–3
templates.

Total G1 calls across all rounds: 2,880 (v1) + 4,320 (v2) + 2,340 (v2-ext incl.
G1b) + 1,440 (Gemma 3) = 10,980, plus the superseded 72-call Claude 4.5 smoke
and three one-call probes.
