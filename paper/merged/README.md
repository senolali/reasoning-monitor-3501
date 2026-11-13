# Merged manuscript: Study 1 (LIC-RQ-v2) + Study 2 (CDR-Stream)

The manuscript is bound by `MERGED_PAPER_CLAIMS.md` (approved 2026-10-04 and
updated with the frozen control-completion result on 2026-11-05).

The three Study 2 control arms and the H-C1 analysis are complete. The remaining
Study 2 consistency, robustness and stability samples are explicitly reported as
incomplete for Llama-3 70B and Gemma 3 12B; no additional paid calls are required
for the manuscript.

The frozen Llama-3/Gemma monitoring validation is also complete. The unchanged
single-stream rules did not pass; six-dialogue aggregation remains an explicitly
exploratory direction.

## Rebuild every number, table and figure

Run from the repository root in Windows Miniconda Prompt:

```bat
python scripts\paper_lic_rq_v2_assets.py --paper paper\merged
python scripts\paper_merged_cdr_assets.py
python scripts\paper_merged_supplement_assets.py
```

## Compile

The main article cites supplementary tables through `xr` (`supp-` labels), so
compile the supplement first:

```bat
latexmk -pdf supplement.tex
latexmk -pdf main.tex
```

In Overleaf, compile `supplement.tex` once as the main document, then switch
back to `main.tex`. Verified PDFs are under `output/pdf/`.

The current build (2026-11-05 rewrite) is 23 pages for the main article in
Elsevier's preprint layout (about 20 pages of text, 3.5 of references) and 14
pages of supplementary material. Section 3 (`sections/framework.tex`) defines
the context policies, metrics, detectors, CRL and evidence labels.

## Supplementary material

`supplement.tex` is a separate, audit-focused document: design details,
deviations, staged acquisition, secondary tables, per-model trajectories, the
power simulation, the ADWIN audit and input hashes.
