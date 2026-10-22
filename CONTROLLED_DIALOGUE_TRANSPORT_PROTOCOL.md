# CDR-v1.1: Packed Dialogue History Transport

CDR-v1.1 is a transport correction prompted by the completed CDR-v1
feasibility pilot. It does not reinterpret the original output as a scientific
context-degradation result.

## CDR-v1 Integration Finding

The completed CDR-v1 run produced 108 records for each of GPT-4o-mini and
Llama-4 Scout. The local records contain correctly ordered multi-message
histories. Nevertheless, both models showed the same exact pattern:

- native `self_history` and `oracle_history`: 100% accuracy at turn 1 and 0%
  across all later turns;
- `state_summary`: 100% accuracy across all 36 turns;
- identical CQ sequences for `self_history` and `oracle_history`.

The models frequently stated that earlier facts were missing even though those
facts were present in the locally recorded message array. Because the same
collapse appeared in two providers and both native-history arms while the
two-message state-summary arm remained perfect, the output is consistent with
native role history being dropped in the gateway path. It is not evidence of a
model reasoning decline. The original operational gate failed and no
scientific interpretation is authorized.

The call-free audit is stored at
`results/controlled_dialogue_v1_pilot/data/controlled_dialogue_v1_transport_audit.json`
with SHA-256
`3f2795118009581b63a81809341d59ce57a0e3528431c86298e9408fb1b31380`.

## Packed Transport

CDR-v1.1 keeps `enable_history: false` so the gateway cannot introduce hidden
server-side state. Instead, the caller serializes prior user and assistant
turns into an explicitly labelled transcript inside the system message and
sends the current user turn separately. Thus every request has exactly two
transport messages while the experimental content still distinguishes actual
model history from canonical oracle history.

This is a gateway-robust representation of accumulated textual context. Any
later report must describe this representation accurately; it must not claim
that the endpoint preserved native OpenAI role arrays.

## Twelve-Call Probe

Before another pilot, CDR-v1.1 tests only turn 2 of the same three calibration
dialogues, both `self_history` and `oracle_history`, and both models. The total
is 12 calls with a maximum of 3,072 output tokens. The probe passes only if all
12 calls succeed, all answers are correct, all final-answer markers are valid,
and no response is truncated.

A pass authorizes only preparation and call-free preflight of a fresh packed-
transport calibration pilot. It does not authorize that larger run,
development or holdout access, or a context-degradation claim.

## Probe Outcome and Fresh Pilot

The original 12-call gate remained failed because one Llama self-history
response ended at exactly the 256-token ceiling before its final answer. The
partial response reproduced every prior site value and correctly updated South
withdrawals from 4 to 6, so it did not exhibit history loss. A prospectively
frozen one-cell retry reused the exact messages with a 512-token ceiling. It
completed at 298 tokens with the correct answer 122, a valid final marker, and
no ceiling hit. All retry checks passed.

Packed transport is therefore operationally feasible for both models. A fresh
CDR-v1.1 feasibility pilot is frozen with the same three calibration dialogues,
12 turns, three conditions, GPT-4o-mini and Llama-4 Scout, K=1/L=0, and a
512-token ceiling. It requires 216 resumable calls and at most 110,592 output
tokens. Accuracy contrasts exclude turn 1 because no dialogue history exists
at baseline. A pass requires state-summary post-baseline accuracy of at least
0.95, oracle-history accuracy of at least 0.85, a state-summary minus
self-history gap of at least 0.10, and self-history failures in at least two
dialogues for at least one operationally valid model.

The complete call-free preflight passes. This permits investigator
consideration of the paid pilot but does not itself authorize execution.

## Fresh Pilot Outcome

The 216-call CDR-v1.1 pilot completed with perfect execution, answer-marker
observability, packed-transport integrity, and no truncation or token-ceiling
hits. The preregistered gate nevertheless failed because `oracle_history` did
not meet its 0.85 solvability threshold: post-baseline accuracy was 0.485 for
GPT-4o-mini and 0.576 for Llama-4 Scout.

The failed gate is preserved. A separate frozen call-free diagnostic found a
cross-model exploratory context signal in the primary real-use contrast.
`state_summary` was 1.0 accurate, while `self_history` was 0.636 for GPT and
0.818 for Llama. In 33 paired post-baseline cells, state summary uniquely
corrected 12 GPT cells (one-sided exact p=0.000244) and 6 Llama cells
(p=0.015625), with no reverse discordance. Both models failed the same budget
cells from turn 7 through turn 12; once the first error appeared, failure was
persistent in that tail.

This remains exploratory because there are only three dialogue clusters. In
addition, `oracle_history` was both shorter and less accurate than
`self_history` in both models, so the two arms were not information-format
matched and their difference cannot cleanly identify error propagation. The
next design should replace this oracle control with `user_only_history`: the
same accumulated user facts without prior assistant prose. Comparing
`state_summary` with `user_only_history` estimates raw-history integration
load, while comparing `user_only_history` with `self_history` estimates the
incremental effect of prior model outputs. No paid successor run is authorized
by this diagnostic.

The diagnostic report SHA-256 is
`ed38761a92e96dfb846a9035f392b85d1dc6b73f08b36f719bc1b847e2de3ca0`.

## CDR-v1.2 User-Only Mechanism Arm

CDR-v1.2 prospectively replaces the unmatched oracle control with
`user_only_history`. Each request has exactly two transport messages. The
system message contains the complete sequence of prior user turns, explicitly
omits every prior assistant response, and the user message contains only the
current turn. Server-side history remains disabled.

The mechanism arm uses the same three 12-turn calibration dialogues and both
models. It adds 36 calls per model, 72 calls total, with a maximum of 36,864
output tokens. The 144 existing `self_history` and `state_summary` cells are
reused under their frozen hashes and are not rerun. All contrasts exclude turn
1. `state_summary` versus `user_only_history` targets raw-history integration
load; `user_only_history` versus `self_history` describes the incremental
effect of prior model prose.

The frozen gate requires perfect execution, at least 0.99 strict-marker and
stored-answer agreement rates, no truncation or token-ceiling hit, and, for
both models, a state-summary advantage of at least 0.10 with one-sided paired
exact p <= 0.05 and user-only failures in at least two dialogues. The complete
call-free preflight passes. A probe pass would authorize only call-free
preparation of a validation on the nine remaining calibration dialogues. It
would not by itself support a scientific claim or authorize further paid
calls, development, or holdout access.

## CDR-v1.2 Outcome and Capacity Sensitivity

The 72-call arm completed, but the frozen gate failed operationally. Three GPT
and seven Llama outputs ended exactly at the 512-token ceiling, and those 10
cells account for every missing strict final marker. The provider nevertheless
reported `finish_reason=stop`, so token count rather than finish reason is the
reliable censoring indicator.

The frozen call-free sensitivity diagnostic keeps that failure unchanged. For
GPT, the 30 uncensored post-baseline cells were 24/30 correct. Even assigning
all three censored cells as correct leaves state summary ahead by 0.182, with
6-0 complete-case discordance, one-sided exact p=0.015625, and observed
failures in all three dialogues. The GPT raw-history-load signal is therefore
robust to capacity censoring. For Llama, the 26 uncensored cells were 24/26
correct. Assigning all seven censored cells as correct reduces the advantage
to 0.061, with 2-0 discordance, p=0.25, and observed failures in only one
dialogue. The Llama mechanism result is capacity-dependent.

The diagnostic is descriptive and not claim-supporting. Its SHA-256 is
`4d730828f96008315c6564c156c7513a615cb69f79559769ca675145810ebf25`.
The next experiment is now frozen and passes call-free preflight. It repeats
only the 10 censored cells (3 GPT, 7 Llama), preserves the exact messages and
original call seeds, and changes only the output ceiling from 512 to 1,024
tokens. It requires 10 resumable calls and at most 10,240 output tokens. The
retry analysis replaces only those censored cells for a sensitivity estimate;
it does not alter or rescue the original failed gate. Paid execution is not
yet authorized.

## CDR-v1.2a Retry Outcome

All 10 exact-message retries completed with valid markers, exact source-message
identity, and no truncation or 1,024-token ceiling hit. GPT answered all three
retried cells correctly; Llama answered four of seven correctly.

After substituting only these completed outputs for the censored cells,
`user_only_history` accuracy is 0.818 for GPT and 0.848 for Llama, compared
with 1.0 for `state_summary`. The paired summary advantages are 0.182 for GPT
(6-0 discordance, one-sided exact p=0.015625, failures in three dialogues) and
0.152 for Llama (5-0, p=0.03125, failures in two dialogues). Both models meet
the frozen raw-history-load mechanism thresholds.

The secondary error-propagation contrast is heterogeneous. User-only history
outperforms self history by 0.182 for GPT (7-1, p=0.035156), but only 0.030 for
Llama (2-1, p=0.5). The defensible cross-model target is therefore accumulated
raw-history integration load and mitigation by structured state. Prior model
prose is a model-dependent secondary mechanism.

The original CDR-v1.2 gate remains failed and immutable. This capacity-adjusted
result is exploratory because it follows a diagnostic retry and contains only
three dialogue clusters. It authorizes call-free preparation of a prospective
validation on the nine unused calibration dialogues, not paid validation,
development or holdout access, or a scientific claim. The retry records and
assessment SHA-256 values are, respectively,
`6520bb74cb8299543d1c03ca55bdffdbfeb3f667f3c8a90e98e5cd87cc8c1c64`
and `058ab9fee886238f2eb638f4f6aec0a075ae397fa5e826cb530f542ab9a21064`.

## CDR-v1.3 Primary Validation

CDR-v1.3 prospectively validates the cross-model mechanism on all nine unused
calibration dialogues: budget, incident, and cohort instances 002-004. The
three feasibility dialogues are not reused. Stage 1 includes only
`user_only_history` and `state_summary`, 12 turns, GPT-4o-mini and Llama-4
Scout, temperature zero, and a 1,024-token ceiling. It requires 216 calls per
model, 432 total, with a maximum of 442,368 output tokens. The 216-call
`self_history` arm is conditionally deferred.

The analysis does not treat 99 post-baseline turns per model as independent.
Each of the nine dialogues contributes one equal-weight accuracy difference,
and the primary one-sided p-value is the exact sign-flip distribution over
dialogue clusters. Both models must have a mean summary advantage of at least
0.10, p<=0.05, and positive gaps in at least seven dialogues. State-summary
accuracy must be at least 0.95.

A distinct longitudinal gate compares early turns 2-6 with late turns 7-12.
For each dialogue, the interaction is late summary advantage minus early
summary advantage. At least one model must have mean interaction >=0.05,
one-sided exact p<=0.05, and positive interactions in at least six dialogues;
the other model must not have a negative mean interaction. Thus a pass
requires both a replicated history-load effect and evidence that it increases
with accumulated dialogue history.

The call-free preflight passes. A pass after execution would authorize only
call-free metric-trajectory analysis and early-warning planning on calibration
records. It would not authorize the self-history arm, additional paid calls,
development or holdout access, or a scientific claim.

The completed run did not pass the frozen formal gate. GPT's state-summary
accuracy was 94/99 = 0.949495, one answer below the 95/99 minimum implied by
the 0.95 threshold. The five wrong outputs had valid strict markers, agreed
with their stored answers, and did not approach the token ceiling; inspection
shows actual arithmetic errors rather than parsing failures. GPT nevertheless
showed a primary cluster gap of 0.282828 (p=0.005859; 8/9 positive) and a
late-minus-early interaction of 0.192593 (p=0.023438; 7/9 positive).

Llama showed a primary gap of 0.181818 (p=0.003906; 8/9 positive) and an
interaction of 0.211111 (p=0.03125; 6/9 positive), but one user-only cell hit
the 1,024-token ceiling. The frozen-output, call-free sensitivity analysis
treats that cell as correct in the most favorable bound. The primary gap
remains 0.171717 with p=0.003906 and 8/9 positive dialogue gaps. Llama's own
interaction p becomes 0.054688, but GPT continues to meet the prospectively
frozen longitudinal requirement. The formal gate remains failed and the
conditional self-history arm remains unauthorized. The assessment and
sensitivity-report SHA-256 values are
`617e38629156f3f6f97dd929b02152be47326a3b1e1d425c6ace810728a2d145` and
`6c86f8902fa449b4393f27a9216704eebfb6b627db970d3aab7700d4f4da5548`.

## CDR-v1.4 Fresh Calibration Replication

CDR-v1.4 is a prospective replication prepared after, but not used to alter,
the failed CDR-v1.3 decision. It does not inspect the existing development or
holdout splits. A new deterministic seed (`20261001`) generates nine new
replication-calibration dialogues under the same three exact-scored template
families. Their identifiers use the separate `cdr-v14-*` namespace and the
frozen bank SHA-256 is
`f72067c5903b4cd4b1ae2cb790665e4344e1f9330f7ffaaf835b1dd3c7439cec`.

The primary conditions, models, 12-turn structure, cluster unit, exact
one-sided sign-flip test, effect-size threshold, 7/9 dialogue coverage, and
early-versus-late interaction are unchanged. The response ceiling increases to
2,048 tokens and the prompt requests at most eight short reasoning lines to
remove the observed capacity mechanism before replication. The absolute
state-summary accuracy floor is prospectively 0.90 and is explicitly treated
as assay validity rather than the primary effect test. This avoids making an
arbitrary 0.95 threshold, which maps to 95/99 = 0.9596 at this sample size, the
sole determinant of replication. The paired cluster contrast remains primary.

The run contains 216 calls per model and 432 total. It is checkpointed and
resumable. Unlike frozen older runners, it directly prints completed/total,
percentage, calls per minute, and ETA after every saved response. The call-free
preflight passes all checks. Paid execution remains locked pending explicit
investigator approval.

```cmd
python scripts\preflight_dialogue_history_transport.py

REM Only after explicit approval of the 12-call probe:
python scripts\probe_dialogue_history_transport.py ^
  --config configs\controlled_dialogue_v1_1_transport_probe_asu.yaml

REM After the probe completes; this makes no model calls:
python scripts\assess_dialogue_history_transport.py

REM Fresh pilot preparation, still call-free:
python scripts\preflight_packed_dialogue_pilot.py
python scripts\estimate_packed_dialogue_campaign.py

REM CDR-v1.2 mechanism-arm preparation, still call-free:
python scripts\preflight_user_only_history_probe.py
python scripts\estimate_user_only_history_campaign.py

REM Only after explicit approval of the 72-call mechanism arm:
python scripts\run_user_only_history_probe.py ^
  --config configs\controlled_dialogue_v1_2_user_only_probe_asu.yaml

REM After the mechanism arm completes; this makes no model calls:
python scripts\assess_user_only_history_probe.py

REM Frozen-output capacity sensitivity; this makes no model calls:
python scripts\diagnose_user_only_history_capacity.py

REM CDR-v1.2a exact-message retry preparation, still call-free:
python scripts\preflight_user_only_history_capacity_retry.py
python scripts\estimate_user_only_history_capacity_retry.py

REM Only after explicit approval of the 10-call retry:
python scripts\retry_user_only_history_capacity.py ^
  --config configs\controlled_dialogue_v1_2_capacity_retry_asu.yaml

REM After the retry completes; this makes no model calls:
python scripts\assess_user_only_history_capacity_retry.py

REM CDR-v1.3 primary validation preparation, still call-free:
python scripts\preflight_primary_dialogue_validation.py
python scripts\estimate_primary_dialogue_validation.py

REM Only after explicit approval of the 432-call validation:
python scripts\run_primary_dialogue_validation.py ^
  --config configs\controlled_dialogue_v1_3_primary_validation_asu.yaml

REM In a second Miniconda Prompt, watch the frozen run without changing it:
python scripts\watch_jsonl_progress.py ^
  --target "GPT=results\controlled_dialogue_v1_3_primary_validation\data\records_CDR13_gpt4o_mini.jsonl,216" ^
  --target "Llama=results\controlled_dialogue_v1_3_primary_validation\data\records_CDR13_llama4_scout.jsonl,216" ^
  --interval 5

REM After validation completes; this makes no model calls:
python scripts\assess_primary_dialogue_validation.py
python scripts\diagnose_primary_dialogue_validation.py

REM CDR-v1.4 fresh-replication preparation; both are call-free:
python scripts\estimate_dialogue_replication.py
python scripts\preflight_dialogue_replication.py

REM Paid execution remains locked until explicit investigator approval:
REM python scripts\run_dialogue_replication.py ^
REM   --config configs\controlled_dialogue_v1_4_replication_asu.yaml

REM After an authorized run completes; this makes no model calls:
REM python scripts\assess_dialogue_replication.py
```
