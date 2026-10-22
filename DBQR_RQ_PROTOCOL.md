# DBQR-RQ-v1 protocol direction

Date: 2026-10-01. Status: the first paid Practice review failed and is locked.
No broader model campaign is authorized.

Exception recorded after the failed review: on 2026-10-01 the investigator
authorized only the six-call DBQR-RQ-v1.1 corrected plan-contract probe. A fresh
84-call review, the remaining Practice calls, Training and Test remain locked.
The probe completed and failed: no response produced the correct Practice
answer under strict typed scoring. The single-shot plan-synthesis path is
terminated; no further prompt-only repair is authorized.

## Current gate status

The independently implemented restricted interpreter executes all 50 pinned
Practice reference plans and reproduces all 50 typed labels. The assessment is
saved at `results/dbqr_v1_audit/reference_plan_assessment.json`; it makes no
model calls. Parser tests reject nested calls, attribute/subscript access,
multiple statements, unknown operations, duplicate targets and oversized
statements/results. The compact-schema audit also passes all 50 Practice turns:
it is deterministic, bounded to at most 1,455 rendered characters in the pinned
data, and exposes neither private table keys, program targets nor resolved text
variables. The strict JSON DSL and checkpointed fixture replay also pass: all
five dialogues traverse both history conditions, producing 100/100 correct
locally executed records, and a second run reuses every checkpoint without new
work. The exact three-model budget, two invariant framing variants, progress
contract and seven-panel plot layout are also frozen and rendered. The causal
oracle broker, resumable acquisition runner and label-blind stage boundary pass
a 420-record fixture preflight. The locked real runner also passes preflight
with the frozen provider probes and real local NLI/BERTScore scorers. All seven
call-free gates passed. The explicitly approved 84-call Practice review then
failed its operational gate because the model-facing prompt omitted the strict
DSL contract. That review is complete and all paid execution is disabled.

## Decision

Use DBQR-QA as a context-dependent table-programming stream, not as a raw-table
reading prompt. The model receives dialogue history, the current question and a
compact schema for the current query-result tables. It emits numbered reasoning
steps plus a declarative JSON operation plan. A local allowlisted interpreter
executes the plan and produces the typed final answer. Model-authored Python,
shell, SQL, Cypher, `eval` and `exec` are forbidden.

This adaptation is necessary, not cosmetic. In Training, only 9/20 dialogues
keep every raw CSV turn below the declared character/4 proxy of 8,192 tokens.
Some turns reach a 161,925-token proxy. Table size is also section-dependent,
so a raw-table experiment would mix context length with turn and task type.

## Scientific estimand

Primary question: within a ten-turn, topic-consistent dialogue, does executable
reasoning-plan quality change with turn position, and does any preregistered
RQEval metric at turn t predict an incorrect executed answer at a later turn?
The dataset is suitable even if no decline or early warning is observed.

The unit of independence is a dialogue, never a turn or K repeat. Turn position,
program length, operation family and compact-schema size must be reported and
controlled. Results must not be described as generic long-context degradation:
DBQR has ten turns and a financial table-programming domain.

## Data roles

- Practice (5 dialogues): parser, interpreter and formatting engineering only.
- Training (20): protocol calibration using dialogue-grouped resampling. Do not
  split turns from one dialogue across folds.
- Blind/Test (15): one locked confirmation after all code, thresholds,
  exclusions and hypotheses are frozen. `Test answers` is the same split with
  labels; its JSON content remains unopened until that point.

No additional outcome-driven pilot sequence is allowed. Practice engineering
must first reproduce all 50 reference answers. Training is opened once for
calibration. The test split is opened once for confirmation.

## Model input contract

Per turn, provide only:

1. Frozen system instructions and allowlisted JSON DSL specification.
2. Prior user questions and the condition-specific prior assistant record.
3. Current question.
4. Current table aliases with shape, axis names, years, concepts, entity count,
   dtypes and bounded non-answer-revealing schema metadata.

Never provide the reference answer, reference program, resolved DBQR variables,
future questions or statistics derived from the gold answer. Do not list all
table values. The compact schema transformer must be deterministic and frozen
before Training outcomes are scored.

## Output and execution contract

The response contains numbered public reasoning steps and exactly one JSON DSL
plan in a final marker. The parser accepts no executable source text. Every
operation, argument type, table alias, intermediate reference, output size and
step count is checked before evaluation. Runtime and memory are bounded.
Failures remain visible and score incorrect; they are not repaired by a model.

The current public corpus uses 28 operations across Practice+Training. Implement
only observed, documented operations. Independently written semantics must be
verified by executing the 50 Practice reference plans and matching every typed
answer. The official GPL implementation may be used as behavioral documentation
but is not copied into this MIT repository.

## Measurement contract

- CQ: typed equality of the locally executed primary-plan answer to gold.
- CS: agreement of executed answers across K isolated same-context plans.
- RS: agreement under L frozen semantic paraphrases; only when primary is
  correct, preserving the established correct-primary subset.
- LS: real NLI over numbered public reasoning steps.
- ES: causal five-turn correctness/token window; turns 1-4 are warmup.
- SS: real BERTScore across same-context reasoning traces.
- RQ6_mean: complete-case mean only; no imputation or changing denominator.

Every result must show six temporal panels and the complete-case mean, including
constants, missingness and sample counts. Early warning uses only pre-error
features, dialogue-grouped uncertainty and multiplicity control. A metric at the
same turn as failure is contemporaneous, not early warning.

## Conditions

The minimum defensible comparison is:

- `self_history`: prior model reasoning plans and executed answers are carried.
- `oracle_answer_history`: the same prior questions and format-matched canonical
  answers are carried, while reference programs remain hidden.

This separates propagation of prior model mistakes from shared turn difficulty
and growing dialogue load. A compact state-summary arm is deferred unless a
deterministic, non-label-leaking state constructor passes Practice audit.

## Call-free gates before any paid run

1. Restricted table loading and archive hashes pass.
2. Compact schema is deterministic, bounded and contains no answer/program.
3. JSON DSL parser rejects arbitrary code and unknown operations.
4. Reference-plan interpreter reproduces 50/50 Practice labels, including list
   and dictionary answers under the frozen typed comparator.
5. All five Practice dialogues can be replayed through both history conditions
   with no model call, using fixture responses and resumable checkpoints.
6. Exact call/input/output budget, progress display and plot outputs are frozen.
7. Training/test labels cannot be read by runner code before their declared stage.

Passing these gates authorizes only a separately approved, small Practice model
transport check. It does not authorize Training or Test execution or a claim.

Current status: gates 1-7 pass on the pinned Practice archive. The fixture
checkpoint and assessment are under `results/dbqr_v1_fixture/`; all 100 records
explicitly state `no_model_calls: true`. The full acquisition preflight report
is `results/dbqr_v1_acquisition_preflight/assessment.json`. The real runner's
preflight loads the frozen provider probes and real local NLI/BERTScore scorers
without making an ASU model call. The investigator authorized only the 84-call
Practice review on 2026-10-01. It completed and failed; the remaining 336
Practice calls, Training and Test retain separate locks. A corrected prompt
requires a new prospective gate.

## Frozen acquisition budget

The contract is `configs/dbqr_rq_v1_budget.yaml`; its call-free assessment is
`results/dbqr_v1_contract/assessment.json`. It uses three models, two history
conditions, K=5, L=2 invariant instruction framings, temperature 0.7 and a
2,048-token output ceiling. Calls per model/condition/turn cell are 7.

- Practice transport: one structurally median dialogue (`chat-4-01`), 420
  logical calls. The first 84 are an included review stop, not a separate pilot.
- Training calibration: 8,400 logical calls.
- Locked Test confirmation: 6,300 logical calls.
- Training plus Test: 14,700 calls and 30,105,600 maximum output tokens.

The two L variants preserve the original question and schema verbatim while
changing only the instruction frame. RS must therefore be described as
robustness to semantically invariant instruction framing, not as robustness to
independently authored paraphrases. Identical repeat outputs are valid measured
agreement, not automatic evidence that sampling parameters were ignored.
