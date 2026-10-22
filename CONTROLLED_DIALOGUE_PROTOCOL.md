# CDR-v1: Controlled Longitudinal Dialogue Reasoning

CDR-v1 is a separate prospective protocol. It is not Section 70, does not
relabel the Section 69 evidence synthesis, and does not reopen RAT-v1. It tests
whether reasoning accuracy changes as correlated facts, intermediate results,
and model responses accumulate within one continuing conversation.

## Identifiable Question

Independent benchmark questions cannot identify a within-conversation context
effect. CDR-v1 therefore uses deterministic 12-turn cases in which every turn
updates a shared state and later questions depend on that state. Three matched
conditions receive the same current user turn:

- `self_history`: all previous user turns and the model's actual previous
  responses. This contains context load and error propagation.
- `oracle_history`: all previous user turns with canonical correct assistant
  responses. The difference from `self_history` isolates error propagation.
- `state_summary`: only the verified state immediately before the current turn.
  The difference from `oracle_history` isolates long-history load.

The primary feasibility contrast is
`CQ(state_summary) - CQ(self_history)`. Turn is conversational position, not
wall-clock time, so the estimand is degradation under accumulated dialogue
context rather than temporal drift between unrelated questions.

## Frozen Bank

The exact-solver bank contains three templates: research-budget accounting,
software-incident state tracking, and study-cohort accounting. Each template
has ten 12-turn dialogue families. Whole dialogues, never individual turns,
are assigned to 12 calibration, 9 development, and 9 untouched holdout
families. Every turn stores its pre-state, post-state, exact integer answer,
two future paraphrases, and a canonical oracle response.

The frozen bank file is
`configs/selections/controlled_dialogue_stream_v1.json`. Its canonical content
SHA-256 is
`5386d4680e713bb5dc6acb77923b5e5f0bbbde68473c46f39cb5c719c778cedb`.

## Feasibility Pilot

The first pilot uses one calibration dialogue per template, 12 turns, all
three history conditions, GPT-4o-mini and Llama-4 Scout, K=1, L=0, temperature
0, and a 512-token output ceiling. This is 216 calls and at most 110,592 output
tokens. Server-side history, search, and prompt enhancement are disabled;
message history is constructed explicitly by the local runner.

The operational gate requires complete calls, at least 0.99 strict final-answer
observability and stored-answer agreement, and no truncation. The phenomenon
gate requires a context penalty of at least 0.10 and a first self-history
failure in at least two dialogues for at least one model. Passing authorises
only preparation and call-free preflight of a larger calibration metric pilot.
It is not an early-warning claim and does not open development or holdout.

The runner checkpoints every completed call. Re-running the same command
resumes from the JSONL records instead of starting over. Paraphrase branches,
when introduced later, will never be appended to the main dialogue history.

## Staged Measurement

Stage 1 measures only CQ, answer-format validity, context size, response size,
and the three matched condition contrasts. RS, LS, ES, and dialogue-state
fidelity are added only if Stage 1 demonstrates a reproducible context effect.
This prevents spending calls on early-warning metrics before showing that the
target failure phenomenon exists in the new design.

## Windows Miniconda Commands

```cmd
python scripts\generate_dialogue_stream_bank.py
python scripts\preflight_controlled_dialogue_pilot.py
python scripts\estimate_dialogue_campaign.py

REM Only after explicit paid-run approval:
python scripts\run_dialogue_stream_pilot.py ^
  --config configs\controlled_dialogue_v1_pilot_asu.yaml

REM After the run completes; this makes no model calls:
python scripts\assess_controlled_dialogue_pilot.py
```

