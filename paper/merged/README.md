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

Upload this folder to Overleaf, or run `latexmk -pdf main.tex`. The repository
also contains verified PDFs under `output/pdf/` and an upload-ready archive at
`paper/merged_overleaf.zip` after running the release workflow.

## Supplementary material

`supplement.tex` is a separate document. In Overleaf, open it and choose
**Menu → Main document → supplement.tex**, or compile it locally with
`latexmk -pdf supplement.tex`. Its tables are written to `generated/supp_*.tex`.
