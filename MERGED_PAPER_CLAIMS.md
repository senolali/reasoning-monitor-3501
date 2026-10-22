# Merged paper (LIC-RQ-v2 + CDR-Stream): research questions and claim boundaries

APPROVED by the investigator on 2026-10-04 (binding for the manuscript). This document is written before
the manuscript, and the manuscript must stay inside these boundaries. It
records the corrections agreed after two external reviews (ChatGPT, relayed by
the investigator) and our own verification of the literature.

## Central claim (the only headline the evidence supports)

> Assistant-generated history and long-horizon state accumulation can reduce
> multi-turn reasoning success in different settings. The single-stream
> quality metrics we examined did not provide robust, model-general early
> warning of this degradation. A verified, compressed state representation may
> be an effective mitigation at long horizons.

## Positioning against prior work (verified 2026-10-04)

**Laban et al. (ICLR 2026).** Multi-turn underspecified conversations lose
performance, mainly through unreliability. This is the source of our Study 1
tasks.

**Huang et al. (arXiv:2602.24287v2), "Do LLMs Benefit From Their Own Words?".**
They compare full context with assistant-omitted context, including on the 103
Lost in Conversation GSM8K tasks, using Qwen3-4B, R1-Distill-Llama-8B,
GPT-OSS-20B and GPT-5.2, and they report "context pollution".

- Their main analyses report binomial 95% confidence intervals.
- Their LiC math figure carries no uncertainty estimates.
- We do **not** claim to be the first to isolate assistant-history effects.
- We do **not** use "they lack statistics" as our differentiator.

**ERGO (Khalid et al., UncertaiNLP 2025).** Token-entropy spikes serve as a
warning signal that triggers a context reset. This is the most direct
comparator for our early-warning question. Our metrics do not use token
entropy, so we contrast the two approaches but do not test ERGO.

**What we add:**

1. A prospectively frozen, task-paired estimate of the assistant-history effect
   on exactly scored math tasks (Study 1). It is a replication with a frozen
   analysis, not a discovery.
2. A long-horizon setting of 100 turns with exact per-turn gold, where accuracy
   degrades with accumulated state, and a verified state summary is compared as
   a mitigation (Study 2).
3. A direct test of whether established single-stream reasoning-quality
   metrics, and the paired measure CRL, give early warning. Neither ERGO nor
   Huang et al. address this.
4. Monitoring-layer evidence: the observed metric directions, the power limits
   of single-stream detectors, and the ADWIN implementation lesson.

## Research questions and status labels

| RQ | Question | Evidence | Status |
|---|---|---|---|
| RQ1 | Does retaining assistant-side history reduce final success on short sharded tasks? | Study 1 primary endpoint | **Confirmatory** (GPT, Mistral); same direction for Llama |
| RQ2 | Does accuracy degrade over a long horizon with accumulated state, at matched difficulty? | Study 2, level-matched early-vs-late contrast; dialogue arm | **Prespecified secondary** (Stage A protocol §4) |
| RQ3 | Does a verified state summary preserve accuracy at long horizon? | Study 2 control arm | GPT now; Llama and Gemma **pending a frozen completion** (below). Confirmatory only after that. |
| RQ4 | Do single-stream quality metrics or CRL give robust early warning? | Study 1 CS/LS/SS and CRL; Study 2 monitoring; replays | Mixed: confirmatory negatives (Study 1 CS/LS/SS); CRL frozen auxiliary; Study 2 monitoring exploratory |

## Wording rules

| Do not write | Write instead |
|---|---|
| "The only difference is the model's own text" | "The contrast estimates the net effect of retaining assistant-side history; context length and message positions also differ" |
| "Six dimensions did not warn" | "The pre-final candidates that could be measured (CS, LS, SS) did not warn; CQ, RS and ES were structurally unobservable before the final turn" |
| "Short-task harm turns into long-horizon collapse" | "The two settings reveal complementary failure modes of assistant history and accumulated state load" |
| "The state summary shows the harm is caused by the model's own replies" | "The summary is a packaged intervention: it shortens context, removes noise and supplies verified state at once. It is not a mechanism test" |
| "Aggregated monitoring works" | "Aggregating sessions is a promising, exploratory direction supported by simulation; it is not a confirmed finding" |
| "Quality metrics never warn" | "We found no robust, model-general early warning among the metrics examined" |
| CRL as "early warning of accuracy loss" | CRL as a "prospective persistence/resolution-delay signal", reported only if it adds value over simple baselines (below) |
| "Self-sabotage" in the title | Anthropomorphic and stronger than the evidence; use a descriptive title (candidate below) |

Study 2 also cannot separate three things: history length, the number of state
updates, and error accumulation. Nothing we add will claim to separate them.

**Title (investigator choice, 2026-10-04):** *The Silent Cost of Conversation
History in LLM Reasoning*.

- "Cost" means the loss of accuracy, not compute cost; the abstract makes this
  explicit.
- "Silent" means that the examined quality metrics gave no robust,
  model-general early warning. The abstract states this limit.
- In the text, "conversation history" is defined once as the accumulated
  context, and "dialogue history" is the name of the experimental condition.

## CRL gate (call-free, post-hoc)

CRL is reported as a contribution only if it adds information beyond four
simple baselines:

- the last pre-final readiness gap;
- the dialogue-arm answer-ready rate before the final turn;
- the number of turns;
- the pre-final support count.

The analysis is `scripts/lic_crl_incremental.py` and its extension. Otherwise
CRL is reported as a descriptive auxiliary measure, not as a main contribution.

**Gate result (2026-10-04, POST-HOC): CRL fails the gate.**
Source: `results/lost_conversation_v2_science/crl_incremental_posthoc/assessment.json`.
Same seed and 10,000 repetitions. The frozen CRL pooled AUROC is reproduced exactly.

| Score | Pooled model-stratified AUROC | Permutation p |
|---|---|---|
| CRL (reproduced) | 0.627 | 0.022 |
| Last pre-final readiness gap alone | **0.672** | 0.0015 |
| CRL within strata of the last gap | **0.468** | 0.67 |

CRL carries no information beyond the last pre-final gap. It measures the
persistence of the unresolved state, as the reviewer suspected. The other
baselines are unnecessary, because one baseline already dominates.

In the manuscript, CRL is therefore reported as a frozen auxiliary measure whose
signal reduces to readiness persistence. That is itself a useful negative
result: even the paired measure does not anticipate failure beyond the current
state. CRL is **not** presented as a main contribution.

## Minimum additional data

The only required addition is the CDR-Stream control arm for Llama-3 70B and
Gemma 3 12B (`CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md`). It is cheap, below
about $10, and needs budget plus explicit approval.

Adding Claude or Gemini to Study 1 is useful but optional. It is not part of
the minimum path. No new pilot chains are allowed; only the evidence gaps of
this paper may be closed.
