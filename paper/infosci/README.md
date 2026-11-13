# Information Sciences submission

Derived on 2026-11-12 from `paper/merged` (the Elsevier preprint version), which
stays unchanged. The scientific content is the same and remains bound by
`MERGED_PAPER_CLAIMS.md`.

Changes from `paper/merged`:

- `\journal{Information Sciences}`; "expert systems" framing changed to
  "information systems" (abstract, introduction, monitoring, Study 2,
  discussion, conclusion).
- Introduction: one sentence stating how this work differs from Laban et al.
- Limitations: a paragraph on why stronger models are absent (Gemma 4 31B
  could not be calibrated; it stayed above the hard band at the contingency
  levels). It cites the new supplement label `sec:s2-calibration`.
- Elsevier generative-AI declaration added above the references. The author
  must confirm its wording before submission.
- New files: `highlights.txt` (5 bullets, at most 85 characters each) and
  `cover_letter.txt`.

## Compile

Compile the supplement first, because the main text cites it through `xr`:

```bat
latexmk -pdf supplement.tex
latexmk -pdf main.tex
```

Numbers, tables and figures are the same as in `paper/merged`; rebuild them
there with the scripts listed in `paper/merged/README.md` and copy
`generated/` and `figures/` here if they change.
