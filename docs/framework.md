# Framework reference

This page documents the monitoring framework implemented in `src/reasoning_monitor`: the six metrics, detectors, configuration, scorer preflight and reproducibility guarantees. It was moved here from the main README without changes to its content. For the experiment-by-experiment record, see [experiment-history.md](experiment-history.md); for every deviation from the original proposal, see [`DEVIATIONS_FROM_PROPOSAL.md`](../DEVIATIONS_FROM_PROPOSAL.md).

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


**Raw generations are persisted first.** Every model call is written verbatim to
`results/<...>/data/records_<tag>.jsonl` (prompt, completion, extracted answer,
tokens, finish reason, latency, success flag) before anything interprets it, and per-query
metric values to `per_query_<tag>.csv`. A scoring bug found after a campaign is
therefore fixed by re-reading those files, not by re-running the campaign.

