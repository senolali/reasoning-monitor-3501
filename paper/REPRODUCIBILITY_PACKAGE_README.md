# Reproducibility package: merged LIC-RQ-v2 + CDR-Stream manuscript

This package accompanies *The Silent Cost of Conversation History in LLM
Reasoning*. It contains the frozen protocols, analysis configurations, source
code, derived assessments, manuscript sources and generation scripts needed to
audit every reported result without issuing model calls.

## Evidence status

- Study 1: frozen task-paired LIC-RQ-v2 analysis, 48 tasks per model.
- Study 2 dialogue arm: 30 dialogues x 100 turns per model.
- Study 2 control arm: complete for GPT-4o-mini, Llama-3 70B and Gemma 3 12B.
- Frozen H-C1 family: Llama-3 70B and Gemma 3 12B; both supported after Holm
  correction. GPT-4o-mini was already observed and is reported outside the
  family.
- Monitoring analyses: Study 1 confirmatory where stated; Study 2 monitoring,
  stream replays and aggregation analyses are exploratory.

## Call-free rebuild

From Windows Miniconda Prompt in the repository root:

```bat
C:\Windows\System32\.venv\Scripts\python.exe scripts\paper_lic_rq_v2_assets.py --paper paper\merged
C:\Windows\System32\.venv\Scripts\python.exe scripts\paper_merged_cdr_assets.py
C:\Windows\System32\.venv\Scripts\python.exe scripts\paper_merged_supplement_assets.py
```

These commands regenerate the manuscript macros, tables and figures from
hash-checked inputs. They do not contact an inference provider.

## Raw records and licensing

The local research repository retains immutable raw JSONL responses. They are
not bundled in the lightweight review archive because of size and provider
redistribution constraints. A public archival release will include derived data
and every raw response that provider terms permit. The release manifest should
record SHA-256 hashes for any raw file distributed separately.

## Claim boundaries

`MERGED_PAPER_CLAIMS.md` is binding. In particular, the verified state summary
is a packaged intervention that simultaneously shortens context, removes noisy
assistant text and supplies a verified state. It is not a mechanism test.
