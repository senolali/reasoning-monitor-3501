# ADWIN defect audit (2026-10-04, call-free)

**Defect.** `detectors.ADWIN` and the fast path
`detector_development._adwin_first_alarms` use the cut bound
eps = sqrt((2/m)·ln(2n/δ)), which has no variance term. For n ≤ about 100
values in [0, 1], eps exceeds any attainable mean difference, so the detector
cannot alarm. The fix is the opt-in `adwin_v` (DEVIATIONS, first entry).
Nothing historical was modified.

## 1. Empirical check: every ADWIN cell in the repository has zero alarms

Every saved result table with a detector column was scanned, excluding test
temporary folders.

| Source | ADWIN rows | Alarms |
|---|---|---|
| `detection_results.csv` of 20 campaigns (E2H calibrations, holdouts, stepband; proposal S1 pilots, validations, stress, metric health, paraphrase health; SuperGPQA; RAT-v1) | 2–18 each | **0** |
| `results/proposal_s1_kl_validation_gpt_llama/detector_development/` (Section 39) | CS and SS cells | **0** (null 0/1000, ordered: none) |
| `results/lost_conversation_v2_stream_replay/` | 180,000 plus 90,000 outcome rows | **0** |
| CDR-Stream Stage A, control screen, aggregated | all | **0** |

ADWIN was used in three roles:

- in `proposal_mapping`, for **CS and SS**;
- in `adwin_all`, for every metric;
- in the Section 39 development specs, for CS and SS.

## 2. Which recorded decisions depend on it

| Evidence | ADWIN role | Recorded conclusion | Affected? |
|---|---|---|---|
| **Section 39** S1 detector development (`scripts/develop_s1_detectors.py`) | CS and SS candidates | "CS/LS/SS specificity passed but never alarmed on ordered streams; no passing candidate; H3 not supported for this S1" | **Yes, for CS and SS.** Their "specificity pass" (0/1000 null alarms) and "no ordered alarm" are artifacts, so CS and SS were never tested. The LS (Page-Hinkley), RS and ES conclusions stand. |
| **Section 47/49** E2H GPT alarm-timing analysis | Section 47 alarms under `proposal_mapping`, which gives CS and SS ADWIN; candidates CS, LS, ES, SS | "No non-CQ candidate led CQ in both intervals" | **Yes, for CS and SS.** Those two candidates could not alarm. LS (Page-Hinkley) and ES (EWMA) are valid. |
| **Section 51** LS/SS reparse sensitivity | SS under `proposal_mapping` (ADWIN) | "corrected SS still did not alarm" | **Yes.** SS could not alarm by construction. The LS conclusion stands. |
| **H2 tables** in every campaign (`h2_f1_main.csv`) | the whole `adwin_all` baseline, plus CS/SS in `proposal_mapping` | strategy comparisons and F1 | **Yes.** The generic baseline was non-functional, so any "specific detectors beat ADWIN" reading is void. |
| **Section 69** terminal synthesis | uses Sections 39, 47–51 | "non-CQ metrics reliably precede CQ: not supported" | **Partially.** The decision ("not supported") remains true as stated, since support was never demonstrated. The CS and SS part of the evidence is *untested*, not negative. The synthesis already marks CS/SS as "not evaluable" for repeat-diversity reasons. The defect adds a second, independent reason. |
| **LIC-RQ-v2-SR replay** | `adwin_all`; `proposal_mapping` for CS and SS | "near-zero detection, no early-warning candidate, no H2 claim" | **Partially.** The `distribution_based` (CUSUM) cells and the LS (Page-Hinkley) and RS (DDM) cells are valid. The `adwin_all` cells and the `proposal_mapping` CS/SS cells are void. "No H2 claim" stands. |
| **CDR-Stream** Stage A, control screen, aggregated | the same roles | H2 and H3 readings | **Partially**, as above. This was already noted in CDR_STREAM_STAGE_A_REPORT §4. |

**Not affected:**

- CQ, which uses CUSUM or DDM;
- LS, which uses Page-Hinkley;
- RS, which uses DDM;
- ES, which uses EWMA;
- every result that does not use the detector layer (accuracy, H1 window
  contrasts, LIC-RQ-v2 confirmatory tests, RAT, CDR-v1.x gates).

## 3. Consequence under AGENTS.md

AGENTS.md allows reopening only "if a newly demonstrated software defect
invalidates immutable evidence used by Section 69". The defect is
demonstrated, and it invalidates the CS and SS detector evidence used by
Section 69 (Sections 39, 49 and 51). It does **not** overturn any Section 69
decision: "not supported" stays "not supported". But the reason given for CS
and SS must change from "did not alarm" to "could not alarm (defect);
untested".

No historical file, gate or decision was edited by this audit.

## 4. Proposed call-free sensitivity re-analysis (needs investigator approval)

Re-run the CS and SS cells only, from the saved records and unchanged
otherwise, with two substitutes:

- `adwin_v`, the corrected ADWIN;
- CUSUM, the `distribution_based` choice, as a functional reference.

The cells to re-run:

1. Section 39 development (S1 K/L records).
2. Section 49 GPT E2H timing (Section 47 records).
3. The `adwin_all` and `proposal_mapping` CS/SS cells of the LIC-RQ-v2-SR
   replay.

Report each as a labelled sensitivity analysis next to the frozen results,
under AGENTS rule 10: never replace them. Expectation, from the power
simulation, is that `adwin_v` will still rarely alarm on 90–150-step streams.
CUSUM is the informative substitute.

## 5. Sensitivity re-analysis results (approved and run 2026-10-04)

Scripts: `scripts/adwin_sensitivity.py` (sections 39 and 49/51) and
`scripts/adwin_sensitivity_lic_replay.py` (LIC replay). Outputs are in
`results/adwin_sensitivity/`. Every input hash was verified. The rebuilt
Section 47 window means match the saved trace to within 1e-16. All other
settings are the frozen ones.

| Evidence | Substitute for CS/SS | Result | Recorded decision |
|---|---|---|---|
| Section 39 | `adwin_v` | Selected δ=0.05. 0/1000 null alarms. No ordered alarm in either model. No candidate. | unchanged |
| Section 39 | CUSUM | CS: no grid value met calibration specificity (null alarms too frequent). SS: selected 0.5, no ordered alarm. No candidate. | unchanged |
| Section 49 | `adwin_v` | CS and SS: no alarm | unchanged |
| Section 49 | CUSUM | CS alarmed at t=30 (baseline not clean), then lagged CQ by 14 at the first transition and led it by 7 at the second. The gate fails. SS: no alarm. | unchanged |
| Section 51 (corrected SS) | `adwin_v` and CUSUM | no alarm | unchanged |
| LIC-RQ-v2-SR replay | `adwin_v` for `adwin_all`; CS/SS of `proposal_mapping` → `adwin_v` or CUSUM | Early-warning candidates: 0 in both variants. W=20 detection with `proposal_mapping` CS + CUSUM is 0.066, identical to `distribution_based`. `adwin_all` with `adwin_v` raised 5 alarms in 180,000 rows, so detection is still 0. | unchanged |

**Conclusion.** With functional detectors, CS and SS still show no early
warning in Sections 39, 49 and 51 or in the LIC replay. Every recorded decision
stands, and the CS/SS part of those decisions is now actually tested rather
than an artifact.

The ADWIN baseline remains uninformative on these short streams even after
correction. H2 comparisons against ADWIN must say so.
