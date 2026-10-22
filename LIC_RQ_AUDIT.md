# LIC-RQ-v1 input audit and measurement limits

2026-10-01. Call-free preparation, not model performance, final preregistration,
an independently validated annotation set or execution approval.

## Completed work

`scripts/audit_lost_conversation.py` checks the pinned source, scans only the
explicit allowlist of completed experiment records, validates per-task semantic
annotations and produces deterministic task-grouped sample/budget alternatives.
Reports are in `results/lost_conversation_v1_audit/`. Changed artifacts cannot be
silently overwritten. The original LIC snapshot and historical results are intact.

| Audit item | Result |
| --- | ---: |
| Public math tasks / prefixes | 103 / 585 |
| Completed historical record files checked | 75 |
| Tasks with exact normalized question overlap | 11 |
| Tasks provisionally quarantined on source semantics | 8 |
| Intersection of those two exclusion groups | 2 |
| Unmatched, non-quarantined candidate tasks | 86 |
| Candidate tasks sufficient before last shard, whole source | 7 |
| Additional candidate-sufficient partial prefixes | 10 |

Exposure matches use input questions, not local qid numbering, correctness or
model outputs. A nonmatch means no match in the declared completed-campaign
allowlist, not proof of never being seen or of no pretraining contamination.
Previously evaluated historical E2H final runs are included as input evidence;
still-closed CDR/RAT development/holdout banks are not opened.

## Source concerns

This is an assistant first-pass review of all 103 questions and shard sequences.
It does NOT grant new prefix scoring eligibility. Independent review is pending.
Quarantine is conservative and provisional, not a claim that all eight gold
answers are conclusively wrong. Decisions precede any LIC model responses.

| GSM8K source ID | Concern |
| --- | --- |
| 267 | Last shard compares cooking time with itself minus five minutes, losing the original comparison with oil-warming time. |
| 201 | Shards omit the award date and blur whether weekly salary is before or after the raise. |
| 288 | Indivisible 75-pound stones allow 26 per 2,000-pound truck: 80 need four trucks, versus gold three from aggregate weight division. |
| 40 | "Four times as old" becomes the ambiguous "four times older". |
| 369 | Exact quarter of the seniors becomes "about" a quarter while gold stays exact. |
| 510 | Year/leap-day and day-count endpoints are unspecified for the calorie calculation. |
| 234 | Gold includes girlfriend's consumption but shards describe supplying Steve's needs. |
| 144 | Shards omit the original only-expenses clause; original also changes the person's name. |

Candidate items can still contain ordinary word-problem assumptions. Notes for
1166, 1019, 1180 and 752 explicitly record examples. No upstream text or gold was
corrected. Excluding ambiguous items changes the population and must be disclosed.

## Important scientific consequence

Among the 95 non-quarantined tasks, the first pass places sufficiency at the last
shard for 88. Only seven have candidate earlier sufficiency. These are annotations,
not observed model behavior and not proof that every earlier prefix is unsolvable.

Consequently, original CQ/RS/ES cannot honestly be a dense per-turn series of
final-answer quality on this source. Clarification under insufficient information
must not count as a wrong answer. LS/SS need actual explanations; CS needs valid
answer attempts. None is guaranteed observable just because repeated calls exist.

All six panels should still be shown with gaps and support counts. RQ6_mean
remains an all-six-measured, gold-dependent, correct-primary-selected summary.
For the provisional 48-task option, only 39/271 prefixes (14.4%) could have all
six under the candidate sufficiency and ES-window assumptions, BEFORE additional
missingness from clarification, absent traces or the RS subset. This is an upper
bound, not measured coverage and not authorization to score partial prefixes.

Do not sell this design as six continuous deterioration curves. It can study
observable trajectories and whether earlier label-free measurements are associated
with final failure, with a paired fresh-prefix endpoint comparison. These are
different questions from within-dialogue first-failure early warning. A terminal
failure prediction is not evidence of progressive reasoning deterioration.
Cross-task turn averages also change task composition as shorter tasks end.

## Budget alternatives, not a pilot ladder

Fixed hash ranking uses source IDs and the declared seed, not performance or
early-answerability enrichment. Four separate integration tasks are reserved:
646, 465, 1207 and 562. All shards/repeats/conditions/models for a source task
stay together. Integration is for technical integrity, never a positive-effect gate.

| Science tasks | Science prefixes | Science calls | Calls including four integration tasks |
| ---: | ---: | ---: | ---: |
| 24 | 133 | 3,724 | 4,452 |
| 48 | 271 | 7,588 | 8,316 |
| 64 | 363 | 10,164 | 10,892 |

Each prefix costs 2 models * 2 conditions * (K=5 including primary + L=2).
The four integration tasks cost 728 calls. The 48-task option including them has
a 17,031,168-output-token ceiling. Input tokens, retries, auxiliary extraction,
paraphrase preparation and review are extra; currency cost is unknown.
These are nested alternatives to choose ONCE, not sequential runs until significant.
No option is selected as an execution manifest by this audit.

Illustrative two-sided exact McNemar power for FINAL paired task success, at
unadjusted alpha 0.05 and hypothetical discordance probabilities 0.25 versus 0.05:
24 tasks 28.3%, 48 tasks 67.5%, 64 tasks 82.7%. At 0.20 versus 0.10, even 64 tasks
provide only 23.6%. These assumptions are not estimated from this source. Neither
the second model nor repeated turns increase the independent source-task count.
These are not power calculations for early warning or multiplicity-adjusted tests.

## Next bounded step

No new effect-seeking pilot. First finalize independently reviewed prefix rules
and paraphrases, then implement natural-response classification/extraction and
the adapted causal ES bookkeeping with offline fixtures. Keep all six dimensions
unchanged and explicit missingness; do not redefine correctness to fill the plots.
Freeze one affordable sample and comparisons before collecting outputs. A single
bounded integration check and the campaign each still need execution approval.

Windows Miniconda CMD, optional audit reproduction (no API calls):

```bat
cd /d "E:\AcademicWorks\Makaleler\Papers After PostDoc\3501 Projesi\reasoning-monitor"
python scripts\audit_lost_conversation.py
```

Use `--write` to reproduce identical reports. This is not an experiment runner.
