# Merged manuscript: Study 1 (LIC-RQ-v2) + Study 2 (CDR-Stream)

The manuscript is bound by `MERGED_PAPER_CLAIMS.md` (approved 2026-10-04).

Text marked `[PENDING: ...]` (orange) depends on the frozen CDR control-arm
completion (`CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md`). Text marked
`[TODO: ...]` (red) needs author input.

## Rebuild every number, table and figure

Run from the repository root in Windows Miniconda Prompt:

```bat
python scripts\paper_lic_rq_v2_assets.py --paper paper\merged
python scripts\paper_merged_cdr_assets.py
python scripts\paper_merged_supplement_assets.py
```

## Compile

Upload this folder to Overleaf, or run `latexmk -pdf main.tex`.

## Supplementary material

`supplement.tex` is a separate document. In Overleaf, open it and choose
**Menu → Main document → supplement.tex**, or compile it locally with
`latexmk -pdf supplement.tex`. Its tables are written to `generated/supp_*.tex`.
