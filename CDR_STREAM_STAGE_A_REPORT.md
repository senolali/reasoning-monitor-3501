# CDR-Stream Stage A report: context collapse, control-arm monitoring, and detector power

Date: 2026-10-04. The work is call-free; it runs on the data acquired before
the budget pause (25,039/54,000 calls). Frozen protocols:

- `CDR_STREAM_ANALYSIS_PROTOCOL.md` (sha `684fecef…`), Stage A;
- `CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md` (sha `f7783a2c…`).

The power analysis is a simulation whose design is fixed in
`scripts/power_cdr_stream_monitoring.py`. Labels follow the protocols:
confirmatory (Stage A H1a/H1b), exploratory (GPT control screen, S0
comparisons) and simulation (power).

## 1. Dialogue arm: accumulated state collapses accuracy (Stage A)

- **Coverage.** CQ and LS are measured on 100% of turns for all three models
  (Llama 99.9%), and ES on 96%. GPT's CS, SS and CQ_ctrl are measured on 99.2%
  of turns. Llama and Gemma CS/SS/RS and control are Stage B.
- **CQ falls with the turn index in every model and every scenario, S0-E
  included.**

  | Model | Turns 1–5 | Turns 6–20 | Turns 21–100 |
  |---|---|---|---|
  | GPT-4o-mini | 0.73 | 0.27 | 0.10 |
  | Llama-3 70B | 0.67 | 0.38 | 0.14 |
  | Gemma 3 12B | 0.67 | 0.37 | 0.10 |

  These are S0-E (easy-only) dialogues.

- **Not question difficulty.** The same questions in GPT's `state_summary`
  arm keep about 0.80 accuracy on easy turns over all 100 turns (0.11 in the
  dialogue arm). At the same difficulty level, late turns are 0.118 lower than
  early turns (p < 1e-4, 90 dialogues).
- **Mechanism (interpretation).** Every user turn updates the state. The
  dialogue arm must reconstruct the state from the whole log, so one missed
  update corrupts every later answer. The two arms cannot separate history
  length from the number of updates.
- **H1a (confirmatory, Holm).** CQ and ES decline after S1-a, S2 and S3-a.
  The background decline in S0 confounds this. S0 shows −0.057 at turn 34
  against −0.128 in the drift dialogues; this comparison is exploratory and
  was not preregistered.
- **H1b (GPT).** Not supported, and the sign is reversed. The dialogue arm is
  already at the floor, so this is uninformative about a context penalty. The
  penalty itself is the gap of about 0.55 between the arms.
- **H2/H3 on the dialogue arm.** Not testable. The S0 nulls also decline, so
  CQ thresholds could not be calibrated, and calibrated detectors caught
  ≤ 2/54.

## 2. Control arm: task drift is real but not detected per dialogue (GPT, exploratory)

Mean CQ_ctrl in four turn segments:

| Scenario | Turns 1–33 | 34–50 | 51–67 | 68–100 |
|---|---|---|---|---|
| S0-E | 0.83 | 0.83 | 0.85 | 0.78 |
| S1 | 0.80 | 0.69 | 0.70 | 0.50 |
| S2 | 0.82 | 0.78 | 0.58 | 0.50 |
| S3 | 0.81 | 0.52 | 0.54 | 0.80 |

The frozen screen (CQ detection ≥ 0.5 within 33 turns, LOO FAR ≤ 1/12,
median delay ≤ 25 at W=20) **failed**:

- CQ-CUSUM detected 0/18 at W=20; its alarms came at turns 86–99. W=10 gave
  4/18 and W=30 gave 1/18.
- LS Page-Hinkley detected 4/18 at W=20.
- `adwin_all` detected 0 everywhere; see Section 4.

The validation spend on Llama and Gemma is therefore not required.

## 3. Detector power (simulation, unchanged detectors and grids)

Setup: p0 = 0.80, drop δ, m dialogues aggregated per turn, and detection
within 33 turns at a FAR target of 0.05. Figures:
`results/cdr_stream_v1_power/power_{oracle,n36,n12}.png/pdf`. Data:
`power_grid.csv` and `summary.json`.

**Smallest δ with at least 80% detection** (best detector; change at
turn 34 / turn 51):

| m dialogues per turn | W=10 | W=20 | W=30 |
|---|---|---|---|
| 1 (a single dialogue) | 0.5 / 0.4 | 0.5 / 0.4–0.5 | 0.5–0.6 / 0.4–0.5 |
| 3 | 0.3 / 0.2–0.3 | 0.3 / 0.3 | 0.3–0.4 / 0.3 |
| 6 | 0.2 / 0.2 | 0.2 / 0.2 | 0.2–0.3 / 0.2 |
| 18 | 0.1 / 0.1 | 0.1 / 0.1 | 0.1–0.2 / 0.1 |

Each cell gives the range across the three calibration conditions: 2,000,
36 and 12 nulls.

- **Single-dialogue monitoring needs δ ≥ 0.4–0.5.** The task drifts in
  CDR-Stream are δ ≈ 0.1–0.3. Per-dialogue monitoring is therefore
  structurally underpowered for them, whatever the window.
- **Aggregation is the lever.** Monitoring the per-turn mean of 6 dialogues
  detects δ = 0.2. With 18 dialogues it detects δ = 0.1.
- **The number of nulls barely matters.** Calibration with 12 versus 2,000
  nulls changes detection only marginally, because noise is the limit, not
  calibration. The true FAR of picks made from 12 nulls averaged 0.03–0.05.
- **W.** W=10 and W=20 are similar. W=30 is worse for a change at turn 34,
  because of the 20-value detector warm-up.
- **Gap between simulation and the real screen** (exploratory). The
  simulation predicts about 0.41/0.60 CUSUM detection for δ = 0.3 at m=1,
  W=20 with 12 nulls, while the real screen gave 0/18. Plausible reasons:
  - the real nulls pool S0-E (p ≈ 0.83) with S0-H (p ≈ 0.45), whose Bernoulli
    variance is higher, which raised the threshold to 7.5;
  - the per-turn level mixture adds extra variability;
  - the S1 first step is only δ ≈ 0.1.

## 4. Software defect found: ADWIN cannot alarm on these streams

`detectors.ADWIN` uses the bound eps = sqrt((2/m)·ln(2n/δ)) without the
variance term. For n ≤ 100 values in [0, 1], eps exceeds 0.8, which is larger
than any attainable mean difference: about 1.10 at n = 50 and about 0.81 at
n = 100. ADWIN never alarmed in the simulation, even for δ = 0.6 with m = 18,
nor on a noiseless step from 0.8 to 0.2. Two consequences:

- **H2 comparisons.** The `adwin_all` baseline in this report and in the
  LIC-RQ-v2-SR replay is non-functional. It is not a valid generic baseline.
- **Earlier evidence.** Any earlier conclusion that used ADWIN alarms needs
  re-examination under AGENTS.md's software-defect rule.

The detector is NOT changed here. A fix is a separate, recorded protocol
change.

## 5. Implications for the project

The proposal's monitoring claims (H2/H3) cannot be supported by per-dialogue
monitoring of CDR-Stream task drift. Two separate problems cause this:

- dialogue-arm accuracy collapses to the floor;
- even the clean control-arm drift is too small for single-stream detectors.

The simulation shows a feasible design: monitor **aggregated traffic**, that
is, the per-turn mean over several concurrent dialogues, which is how a
deployed system sees its query stream. With m ≥ 6, the drifts in CDR-Stream
(δ ≈ 0.2–0.3) are detectable at controlled FAR. A prospective test of this on
the control arm needs a new frozen protocol and unseen data, such as the
Llama/Gemma control arms, which cost under about $10.

### 4a. Fix and re-evaluation (2026-10-04)

The opt-in `adwin_v` (`ADWINVariance`) implements the published variance
bound (DEVIATIONS, first entry). Its unit tests pass, and the historical
`adwin` is unchanged.

**Power simulation.** Same design as Section 3, with a change at 51 and
oracle calibration (`results/cdr_stream_v1_power/adwin_v/`).

- Detection within 33 turns is 0 for every drop up to 0.5, at every W and
  every m. The only exception is W=10 with δ=0.6, at about 0.4.
- By turn 100 at W=20, δ=0.6 is caught 61–93% of the time and δ ≤ 0.4 almost
  never.
- The true FAR is 0 in every cell.
- Aggregating dialogues does not help. The (2/(3m))·ln(2/δ') term does not
  depend on variance and needs hundreds of samples per side.

**Real aggregated GPT control streams.** `adwin_v` raised no alarm on any
drift stream at any W.

**Conclusion.** ADWIN is now correct, but it is structurally unsuited to
100-turn streams: it is a long-stream detector. The proposal's generic ADWIN
baseline therefore cannot be a fair H2 comparator on CDR-Stream. CUSUM and
Page-Hinkley, which the proposal maps to CQ and LS, detect the 0.2–0.3 drops
under aggregation, while ADWIN does not.
