# reasoning-monitor

[![arXiv](https://img.shields.io/badge/arXiv-2611.20071-b31b1b.svg)](https://arxiv.org/abs/2611.20071)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Streaming monitoring of LLM reasoning quality under drift. The framework scores six reasoning-quality dimensions per query or turn, aggregates them over sliding windows and runs sequential change detectors (CUSUM, Page-Hinkley, EWMA, DDM, ADWIN) on the resulting streams. It was developed in TÜBİTAK 3501 project 126E663.

This repository contains the code behind **[The Silent Cost of Conversation History in LLM Reasoning](https://arxiv.org/abs/2611.20071)**.

## The paper in brief

- **Study 1 (LIC-RQ-v2).** On 48 sharded math tasks from *Lost in Conversation*, retaining the model's own replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004). Llama-4 Scout moved in the same direction without significance.
- **Study 2 (CDR-Stream).** In 100-turn state-tracking dialogues with an exact answer at every turn, accuracy under full history fell from about 0.6 to 0.1 within twenty turns at matched difficulty. A solver-verified state summary kept accuracy 0.52–0.59 higher (Holm-adjusted p < 0.001).
- **Monitoring.** None of the examined reasoning-quality metrics, a paired readiness measure or calibrated single-stream drift detectors gave robust, model-general early warning. A single 100-turn stream had power only for very large drops; aggregating across sessions is a promising, exploratory direction.

See the paper for the full claim boundaries and limitations.

## Installation

```bash
git clone https://github.com/senolali/reasoning-monitor-3501.git
cd reasoning-monitor-3501
pip install -e ".[nli]"      # adds the NLI scorer used for logical coherence (LS)
pip install -e ".[all]"      # optional: datasets, NLI and development tools
```

Python 3.10 or later. Metric scoring is delegated to [`rqeval`](https://pypi.org/project/rqeval/), pinned at `2.0.3`; changing that version changes the measurements.

## Quick start

```bash
# Smoke test with simulated models (no API key, no paid calls)
python -m reasoning_monitor.cli --quick

# Unit tests
pytest tests/ -v
```

Live runs use an OpenAI-compatible gateway configured in `configs/*_asu.yaml`. Every paid run first prints its plan and budget; calls are made only with an explicit execution flag.

## Reproducing the paper

Raw model outputs are **not** included in this repository (`results/` is git-ignored). The analysis scripts below read them from `results/<run>/data/`. Commands without an execution flag make no model calls.

| Study | Protocol | Run (paid; plan only by default) | Call-free analysis |
|---|---|---|---|
| Study 1: LIC-RQ-v2 | [`LIC_RQ_V2_PROTOCOL.md`](LIC_RQ_V2_PROTOCOL.md) | `python scripts/run_lost_conversation_v2_science.py --preflight` | `python scripts/analyze_lost_conversation_v2_science.py`<br>`python scripts/lic_crl_incremental.py`<br>`python scripts/replay_lost_conversation_v2_streams.py` |
| Study 2: CDR-Stream | [`CDR_STREAM_DESIGN.md`](CDR_STREAM_DESIGN.md), [`CDR_STREAM_ANALYSIS_PROTOCOL.md`](CDR_STREAM_ANALYSIS_PROTOCOL.md) | `python scripts/prepare_cdr_stream.py`<br>`python scripts/run_cdr_stream.py --config configs/cdr_stream_v1_main_asu.yaml --preflight` | `python scripts/analyze_cdr_stream.py --stage-a`<br>`python scripts/analyze_cdr_stream_control_completion.py`<br>`python scripts/power_cdr_stream_monitoring.py`<br>`python scripts/forensic_cdr_early_warning.py` |

`python scripts/analyze_cdr_stream.py --synthetic` runs the complete Study 2 analysis pipeline on synthetic records, so the code path can be checked without the raw data. The paper's binding claim boundaries are recorded in [`MERGED_PAPER_CLAIMS.md`](MERGED_PAPER_CLAIMS.md).

## Repository layout

```
src/reasoning_monitor/   framework: clients, metrics adapters, windows, detectors, evaluation
scripts/                 run, preflight and call-free analysis scripts
configs/                 frozen run configurations and item selections
tests/                   unit and regression tests
docs/                    framework reference and full experiment history
*_PROTOCOL.md, *_DESIGN.md, *_REPORT.md
                         frozen study protocols and reports
DEVIATIONS_FROM_PROPOSAL.md
                         every change from the original project proposal
```

## Documentation

- [Framework reference](docs/framework.md): the six metrics, detector strategies, configuration, scorer preflight and reproducibility guarantees.
- [Experiment history](docs/experiment-history.md): the full TÜBİTAK 3501 experiment chain, including stages that did not pass their preregistered gates.
- [`DEVIATIONS_FROM_PROPOSAL.md`](DEVIATIONS_FROM_PROPOSAL.md): the complete deviation register.

## Citation

If you use this code or its results, please cite:

```bibtex
@misc{senol2026silentcost,
  title         = {The Silent Cost of Conversation History in {LLM} Reasoning},
  author        = {{\c{S}}enol, Ali},
  year          = {2026},
  eprint        = {2611.20071},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.20071},
  url           = {https://arxiv.org/abs/2611.20071}
}
```

GitHub's **"Cite this repository"** button (from `CITATION.cff`) gives the same reference in APA and BibTeX.

## License and acknowledgement

Released under the [MIT License](LICENSE). This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK), 3501 program, project 126E663.
