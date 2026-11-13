# Expert Systems with Applications backup submission

Standby alternative, prepared in case Information Sciences (`paper/infosci`)
rejects. Derived from `paper/infosci` on 2026-11-13 after its full review-fix
history, not from the older `paper/merged`, so it carries every correctness
and wording fix made during that process (abstract rewording, the RS/SS
label-free correction, the dataset table, the worked CDR-Stream example, the
three-evaluation framing in Section 6, the S0-E text/table consistency fix,
and the stale cross-reference fixes).

Changes from `paper/infosci`:

- `\journal{Expert Systems with Applications}`.
- "information systems" / "information and decision-support systems" framing
  reverted to "expert systems" / "expert and decision-support systems" in the
  abstract, introduction and discussion, to match this journal's scope and
  name. No other wording changed.
- `cover_letter.txt` retargeted to this journal (same content otherwise).

Not yet done, needed only if this version is actually submitted:
- Confirm the current ESWA guide for authors (highlights count/length,
  reference style, peer-review model) directly on the journal's own page;
  this was not independently re-verified for this copy.
- `highlights.txt` reused unchanged from `paper/infosci`; recheck its wording
  still reads naturally under "expert systems" framing.
- Recommended route: submit via the subscription (non-open-access) option,
  which carries no APC, rather than gold open access.

## Compile

Same as `paper/infosci`: compile `supplement.tex` first, then `main.tex`.
Supplementary cross-references are fixed numbers (no `xr` dependency), so
either document compiles standalone.
