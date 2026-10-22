# DBQR-RQ-v1: practice input audit

Date: 2026-10-01. Status: structural inspection and reference-plan execution
complete. Independent semantic review remains pending. No model calls. No paid
run authorized.

## Decision and preservation

The investigator selected DBQR-QA after the dataset comparison. LIC is frozen
as a reference design, not deleted or retroactively declared unsuccessful.
CDR, RAT, E2H, historical results and gates remain unchanged. This is a
supplied-table conversational reasoning adaptation, not a reproduction of the
complete database-query task. Do not use existing dialogue runners unchanged.

## Reproduce in Windows Miniconda Prompt

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\audit_dbqr.py --fetch --write
python scripts\audit_dbqr.py --safe-tables --write
python scripts\assess_dbqr_reference_plans.py
```

After the first download, omit `--fetch` for fully offline replay. Paths resolve
relative to the repository, not the terminal working directory. The pinned
source hash is in the report; a changed upstream archive fails closed.

Report: `results/dbqr_v1_audit/assessment.json`.
Reference execution report:
`results/dbqr_v1_audit/reference_plan_assessment.json`.
Compact-schema report:
`results/dbqr_v1_audit/compact_schema_assessment.json`.
Fixture replay report: `results/dbqr_v1_fixture/assessment.json`.
Budget/manifest/plot report: `results/dbqr_v1_contract/assessment.json`.
Source: https://dbqr-qa.s3.us-east-1.amazonaws.com/practice.tar.gz
Documentation: https://dbqr-qa.github.io/quickstart.html
Paper: https://aclanthology.org/2024.findings-acl.900/

## Observed results

- 5 practice dialogues, 10 distinct turns each, 50 questions total.
- 50 reference programs and 50 matching table files present.
- Answers: 32 float, 9 integer, 4 string, 4 list, 1 dictionary.
- Archive folder IDs and metadata conversation IDs are not always identical;
  preserve both, never infer one from the other.
- Restricted table inspection uses a static pickle-global allowlist before
  loading. It found 50 pickle files, 82 DataFrames, 24,713 total rows and
  77,478 cells. Only the ten documented Pandas/NumPy/builtin constructors were
  referenced; arbitrary globals are rejected.
- The largest table has 2,183 rows and 6,549 cells. Raw-table prompting is not
  yet feasible: representation and token budgets need an independent gate.
- A reproducible CSV rendering has 2,592,510 characters in total. The largest
  turn is 281,426 characters (character/4 is about 70,357 tokens; this is an
  explicit proxy, not a tokenizer measurement). Per-dialogue proxies range
  from 581 to 515,352 tokens, so table load is strongly confounded with section.
- No training/test archive was downloaded or inspected by this audit.
- No model metric was measured, so no model-quality trajectories are presented.
- The restricted interpreter executed 50/50 Practice reference plans and the
  frozen typed comparator matched 50/50 labels: 10/10 in every section. This
  validates the executable scoring path, not the natural-language correctness
  of every gold program and not any model's reasoning quality.
- The compact-schema transformer passed all 50 Practice turns. It was
  deterministic, never exceeded 1,455 rendered characters, and emitted no
  private table key, program target or resolved string variable. It publishes
  shapes, axis structure/cardinality, year candidates, dtype counts, missing
  counts and entity counts, but no table cell values or entity/concept labels.
- The strict JSON DSL and fixture replay passed 100/100 records: five Practice
  dialogues, ten turns and two isolated history conditions. Every translated
  fixture plan executed to the typed gold answer. Per-turn checkpoints resumed
  with zero repeated records. This is transport engineering with no model calls,
  not model evidence.
- The frozen contract selects `chat-4-01` for the single Practice transport
  check by median total compact-prompt characters, not by model outcome. It
  fixes K=5, L=2, three models, two conditions and seven plots. Practice and
  Training prompt manifests contain 50 and 200 cases respectively and exclude
  answers, programs, variables and queries. No stage is execution-authorized.
- The causal oracle broker and acquisition runner passed a 420-record call-free
  preflight, including the 84-call chronological review stop, per-call resume,
  branch isolation and local execution. The locked real runner then loaded the
  frozen provider probes and real NLI/BERTScore scorers successfully. No ASU
  model call was made. The investigator subsequently authorized only the
  84-call Practice review. It completed with 0/84 valid executable plans and
  failed its gate; all further paid execution is locked.

## Review flags, not confirmed dataset errors

1. Section 2 / conversation 11 / turn 8 asks for the average within the
   preceding threshold. Its program queries `concept_1`, whereas the nearby
   investment questions use `concept_2`. Review the full context and tables
   before accepting, correcting or excluding it. Do not silently repair gold.
2. Section 2 / conversation 11 / turn 3 says "more than 500M" but the
   reference program uses `condition='ge'`. Equality rows must be inspected
   to determine whether this affects the answer.
3. Section 5 programs mix complex filtering, changes to financial quantities
   and aggregation. Program-statement count is NOT reasoning-step ground truth.
4. Website tutorial answer examples are not authoritative labels: the first
   practice question's downloaded answer is `higher`, whereas a tutorial code
   example uses `lower`. Use the pinned data, then independently recalculate.

These are first-pass assistant observations, not independent semantic review.
The audit does not claim that all other questions are free of issues.

## Published split inventory

The four downloads are not four independent sample pools. Practice contains 5
chats/50 questions; Training has 20/200; Blind Test has 15/150. Test answers is
the released-label version of the same 15 Blind Test chats, not another 150
questions. Every chat has ten turns, giving 40 chats and 400 questions overall.
Practice has one chat per section. Training chat counts by section are 5, 5, 5,
3 and 2; test counts are 4, 4, 4, 1 and 2.

The Blind and released-test layouts match exactly and all 150 table files have
identical SHA-256 hashes; all 150 question JSON files differ, consistent with
the published addition of programs and answers. Their contents were not parsed
during this audit, preserving the test labels. The released archive also has
333 macOS AppleDouble `._*` metadata files that must be ignored. Practice and
Training labels were inspected; their conversation IDs are disjoint.

## Measurement contract

CQ can potentially be scored each turn after reference validation. CS/SS need
isolated same-context repeats; RS additionally needs frozen, reviewed
paraphrases and remains undefined outside the correct-primary subset. LS needs
actual explanation steps and real NLI. Never feed reference programs, answers,
resolved variables or future turns into the model prompt.

ES window 5 leaves turns 1-4 as warmup. At most 30 of these 50 practice turns
can have ES, before response failures and other missingness. RQ6_mean requires
all six finite measured values; do not impute or vary its denominator. All six
panels plus the mean must show constants and missingness explicitly in later
experiments. These are eligibility rules, not observed metric results.

## Next controlled step

The paid acquisition runner is implemented and call-free preflighted against
only the validated prompt manifest and causal oracle-answer broker. Its first
possible paid action is the included 84-call Practice review stop, and that
action was explicitly authorized by the investigator on 2026-10-01 and failed.
The frozen records show a shared prompt-contract omission rather than model-
quality evidence: the prompt named the marker but did not expose the required
plan envelope, references, operation allowlist or valid example. The corrected
v1.1 contract is call-free only; any minimal transport probe needs new approval.
The remaining 336 Practice calls, Training and Test stay locked.

The investigator subsequently authorized only the six-call DBQR-RQ-v1.1
contract probe (three models by two instruction framings). Its conjunctive gate
requires two valid locally executable plans per model and zero truncations. The
authorization does not extend to a fresh 84-call review or any larger stage.

The six-call probe completed and failed. Syntax improved, but strict offline
typed scoring found 0/6 correct plans for the gold answer `2020`: GPT produced
one executable but incorrect answer (`440`), Llama's two executable plan bodies
did not yield a scalar answer, and both Mistral plans used unsupported filter
arguments. The original operational report and the stricter call-free report
are both preserved; the latter is
`results/dbqr_v1_1_contract_probe/scientific_assessment.json`. No fresh 84-call
review is authorized. Prompt-only single-shot plan synthesis is closed.
Independently review the four semantic flags before Training. Practice is
engineering data, not confirmatory evidence.

Ten turns offer a short trajectory; do not promise long-horizon conclusions.
Use dialogue/template clusters, not turns, as independent units. Different
question difficulty and table size can confound turn position. Dataset
suitability is not conditional on producing declining accuracy or warning
signals. There is no automatic expansion or execution path in this tool.

## Training context audit

The pinned Training archive contains 20 dialogues/200 turns. Restricted table
loading passed with the same ten Pandas/NumPy globals as Practice. Only 9/20
dialogues keep every raw CSV turn at or below the descriptive 8,192 token proxy:
all five Section 1 dialogues and four of five Section 2 dialogues. Sections 3-5
are not raw-prompt feasible; the largest Training turn reaches a 161,925 proxy.

Across Practice+Training, all 250 reference plans parse into 28 documented
operation names and no statement is malformed. Consequently DBQR-RQ-v1 will use
compact table schemas and an allowlisted JSON operation plan, as specified in
DBQR_RQ_PROTOCOL.md. This is a table-programming adaptation; it must not be
reported as direct natural-language reasoning over complete raw tables.

The comparator is frozen for practice engineering: the final non-empty line
must be `FINAL_ANSWER_JSON: <JSON>`. Integers are exact without truncation;
floats compare after cent-level half-even rounding; strings use case-folded,
whitespace-normalized equality; dictionaries require exact keys. Lists use the
dataset's documented unordered-set semantics for the primary score, with an
order-sensitive result retained as a prespecified sensitivity analysis. This
also avoids the official evaluator's `answer = list` implementation defect.

The ordinal words `greater`/`higher`/`more` and `lower`/`less` are prospectively
canonicalized because DBQR programs emit one form while labels use another.

## Reference execution gate

The interpreter is an independent, restricted implementation. It accepts only
one assignment to an allowlisted operation per statement, never evaluates
source, and bounds statement complexity, plan steps and intermediate output.
Query calls select already supplied, hash-pinned table aliases; model-authored
SQL, Cypher, Python, shell, attributes, subscripts and nested calls are rejected.
The official GPL code was used only as behavioral documentation and was not
copied into this MIT repository.

All 50 pinned Practice programs executed and reproduced their typed labels.
This closes only the reference-execution gate. The four semantic review flags
above remain review items because reproducing a supplied program does not prove
that the program faithfully represents its natural-language question.
