"""CDR-Stream dialogue runner: packed-history streams with K/L and a control arm.

One *unit* is one (model, dialogue instance). At every turn t the runner
issues, for that unit:

- dialogue arm: K same-context calls (call 0 is the primary) and L paraphrase
  calls, all with the packed history of turns 1..t-1. Only the primary reply
  enters later history (LIC/CDR rule); repeats and paraphrases never do.
- control arm: one call with the verified state summary before turn t plus the
  current turn (identical to the frozen G1 prompt).

Turns of one dialogue are sequential (turn t needs the primary reply of t-1);
dialogues and models run in parallel. Primary calls are scheduled first because
they are the critical path. A sliding-window limiter keeps estimated tokens per
minute under the gateway limit. Every completed call is written raw before it
is used; successful calls are never repeated; failed calls are retried on the
next run; a circuit breaker stops repeated failures. Nothing here is scored
beyond the frozen marker parser; metrics are computed later, offline.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_monitor import cdr_stream_g1 as g1
from reasoning_monitor.data import cdr_stream as cs

PROTOCOL_ID = "CDR-Stream-v1-streams"
EMPTY_PRIMARY_PLACEHOLDER = "[EMPTY_PRIMARY_RESPONSE]"
# Opt-in (main-run truncation policy, frozen before the main run): a primary
# reply cut at the output ceiling enters later history as this placeholder;
# its raw text stays in the record and is scored as missing.
TRUNCATED_PRIMARY_PLACEHOLDER = "[TRUNCATED_PRIMARY_RESPONSE]"
MAX_CONSECUTIVE_FAILURES = 10
ASSUMED_OUTPUT_TOKENS = 300  # for rate limiting only
CHARS_PER_TOKEN = 4.0

SYSTEM_PROMPT = g1.SYSTEM_PROMPT
RESPONSE_CONTRACT = g1.RESPONSE_CONTRACT


def dialogue_messages(history: Sequence[Mapping[str, str]], turn_text: str
                      ) -> list[dict[str, str]]:
    """Packed transcript in one user message (gateway history is disabled)."""
    if history:
        packed = json.dumps(list(history), ensure_ascii=False)
        context = f"Conversation so far (oldest first), as JSON:\n{packed}"
    else:
        context = "Conversation so far: none."
    content = f"{context}\n\nNew message:\n{turn_text}\n\n{RESPONSE_CONTRACT}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


def prompt_sha256() -> str:
    probe = dialogue_messages([{"role": "user", "content": "u"},
                               {"role": "assistant", "content": "a"}], "t")
    return hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()


def instance_sha256(instance: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(instance, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


# ── unit selection ──────────────────────────────────────────────────────────

def mapped_templates(mappings: Mapping[str, Any], model: str) -> dict[str, dict[str, int]]:
    entry = mappings["models"][model]
    if not entry.get("retained"):
        raise ValueError(f"model {model} is not retained in the frozen mappings")
    return dict(entry["mappings"])


def select_units(bank: Mapping[str, Any], mappings: Mapping[str, Any], models: Sequence[str],
                 *, split: str, scenarios: Sequence[str], per_model: int,
                 indices: Sequence[int] | None = None, templates: Mapping[str, str] | None = None
                 ) -> list[dict[str, Any]]:
    """Choose dialogues per model, balanced over its mapped templates and scenarios.

    ``templates`` optionally pins one template per model (pilot). Holdout cells
    are filled in index order so that the per-model count is met exactly.
    """
    units = []
    for model in models:
        mapped = mapped_templates(mappings, model)
        model_templates = [templates[model]] if templates else sorted(mapped)
        pool = [d for d in bank["dialogues"]
                if d["template"] in model_templates and d["split"] == split
                and d["scenario"] in scenarios
                and (indices is None or d["index"] in indices)]
        cells = sorted({(d["scenario"], d["template"]) for d in pool})
        if not cells:
            raise ValueError(f"no dialogues for {model}")
        by_cell = {cell: sorted((d for d in pool if (d["scenario"], d["template"]) == cell),
                                key=lambda d: d["index"]) for cell in cells}
        chosen: list[dict[str, Any]] = []
        depth = 0
        while len(chosen) < per_model:
            added = False
            for cell in cells:
                if depth < len(by_cell[cell]) and len(chosen) < per_model:
                    chosen.append(by_cell[cell][depth])
                    added = True
            if not added:
                raise ValueError(f"only {len(chosen)} dialogues available for {model}")
            depth += 1
        for skeleton in chosen:
            units.append({"model": model, "dialogue_id": skeleton["dialogue_id"],
                          "template": skeleton["template"], "scenario": skeleton["scenario"],
                          "mapping": mapped[skeleton["template"]]})
    return units


def build_manifest(bank: Mapping[str, Any], units: Sequence[Mapping[str, Any]],
                   version: str = "v2") -> list[dict[str, Any]]:
    skeletons = {d["dialogue_id"]: d for d in bank["dialogues"]}
    manifest = []
    for unit in units:
        instance = cs.instantiate(skeletons[unit["dialogue_id"]], unit["mapping"], version)
        manifest.append({**unit, "instance_sha256": instance_sha256(instance)})
    return manifest


# ── call plan ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CallSpec:
    unit: int
    turn: int
    arm: str          # "dialogue" or "control"
    role: str         # "primary", "repeat", "paraphrase"
    index: int

    @property
    def is_primary(self) -> bool:
        return self.arm == "dialogue" and self.role == "primary"


def turn_calls(unit: int, turn: int, k: int, l_paraphrases: int, control: int
               ) -> list[CallSpec]:
    calls = [CallSpec(unit, turn, "dialogue", "primary", 0)]
    calls += [CallSpec(unit, turn, "dialogue", "repeat", i) for i in range(1, k)]
    calls += [CallSpec(unit, turn, "dialogue", "paraphrase", i) for i in range(l_paraphrases)]
    calls += [CallSpec(unit, turn, "control", "primary", i) for i in range(control)]
    return calls


def call_key(unit: Mapping[str, Any], spec: CallSpec) -> tuple:
    return (unit["model"], unit["dialogue_id"], spec.turn, spec.arm, spec.role, spec.index)


def row_key(row: Mapping[str, Any]) -> tuple:
    return (row["model"], row["dialogue_id"], row["turn"], row["arm"], row["role"],
            row["call_index"])


# ── checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint: Path, fingerprint: str) -> tuple[dict, int]:
    """Latest successful row per key and the count of failed attempts.

    A partially written final line (interrupted write) is moved aside.
    """
    successes: dict[tuple, dict] = {}
    failures = 0
    if not checkpoint.exists():
        return successes, failures
    text = checkpoint.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        complete, _, partial = text.rpartition("\n")
        with checkpoint.with_suffix(".partial.jsonl").open("a", encoding="utf-8") as aside:
            aside.write(partial + "\n")
        checkpoint.write_text(complete + "\n" if complete else "", encoding="utf-8")
        text = checkpoint.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint belongs to a different frozen stream setup")
        if row["ok"]:
            successes[row_key(row)] = row
        else:
            failures += 1
    return successes, failures


# ── rate limiting ───────────────────────────────────────────────────────────

class TokenRateLimiter:
    """Sliding 60-second window on estimated tokens."""

    def __init__(self, tokens_per_minute: int, clock=time.monotonic):
        self.limit = int(tokens_per_minute)
        self.clock = clock
        self.window: deque[tuple[float, int]] = deque()
        self.used = 0

    def _expire(self) -> None:
        now = self.clock()
        while self.window and now - self.window[0][0] >= 60.0:
            self.used -= self.window.popleft()[1]

    def try_acquire(self, tokens: int) -> bool:
        self._expire()
        if self.used + tokens > self.limit and self.used > 0:
            return False
        self.window.append((self.clock(), tokens))
        self.used += tokens
        return True

    def seconds_until_free(self) -> float:
        self._expire()
        if not self.window:
            return 0.0
        return max(0.0, 60.0 - (self.clock() - self.window[0][0]))


def estimate_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    chars = sum(len(m["content"]) for m in messages)
    return int(chars / CHARS_PER_TOKEN) + ASSUMED_OUTPUT_TOKENS


# ── scheduler ───────────────────────────────────────────────────────────────

class CircuitBreakerOpen(RuntimeError):
    """Too many consecutive failed calls; stop and resume later."""


@dataclass
class UnitState:
    unit: Mapping[str, Any]
    instance: Mapping[str, Any]
    primaries: dict[int, str] = field(default_factory=dict)
    truncated_primaries: set[int] = field(default_factory=set)
    mask_truncated: bool = False

    def history_before(self, turn: int) -> list[dict[str, str]]:
        history = []
        for t in range(1, turn):
            spec = self.instance["turns"][t - 1]
            history.append({"role": "user", "content": spec["text"]})
            reply = self.primaries[t]
            if self.mask_truncated and t in self.truncated_primaries:
                reply = TRUNCATED_PRIMARY_PLACEHOLDER
            history.append({"role": "assistant",
                            "content": reply if reply.strip() else EMPTY_PRIMARY_PLACEHOLDER})
        return history


def build_messages(state: UnitState, spec: CallSpec) -> list[dict[str, str]]:
    turn = state.instance["turns"][spec.turn - 1]
    if spec.arm == "control":
        return g1.control_messages(turn["state_summary_before"], turn["text"])
    text = turn["paraphrases"][spec.index] if spec.role == "paraphrase" else turn["text"]
    return dialogue_messages(state.history_before(spec.turn), text)


def acquire_streams(units: Sequence[Mapping[str, Any]], instances: Sequence[Mapping[str, Any]],
                    clients: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]],
                    checkpoint: Path, fingerprint: str, max_tokens: int, *,
                    k: int, l_paraphrases: int, control: int, workers: int = 1,
                    tokens_per_minute: int = 600_000, callback=None, clock=time.monotonic,
                    sleep=time.sleep, mask_truncated_primaries: bool = False,
                    arms: Sequence[str] | None = None) -> dict[str, Any]:
    """Acquire the frozen call plan; resumable.

    ``arms`` (opt-in) restricts which planned calls are *submitted*, e.g.
    ("control",). It does not change seeds, request content, fingerprint or the
    checkpoint format; the remaining calls are acquired later by a run without
    the filter. Histories are still rebuilt from saved primaries.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    saved, _ = load_checkpoint(checkpoint, fingerprint)
    states = [UnitState(unit, instance, mask_truncated=mask_truncated_primaries)
              for unit, instance in zip(units, instances, strict=True)]
    turns_total = [len(s.instance["turns"]) for s in states]
    expected = {call_key(s.unit, c) for i, s in enumerate(states)
                for t in range(1, turns_total[i] + 1)
                for c in turn_calls(i, t, k, l_paraphrases, control)}
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen plan")

    primary_q: deque[CallSpec] = deque()
    other_q: deque[CallSpec] = deque()

    def open_turn(i: int, turn: int) -> None:
        for spec in turn_calls(i, turn, k, l_paraphrases, control):
            if call_key(states[i].unit, spec) in saved:
                continue
            if arms is not None and spec.arm not in arms:
                continue
            (primary_q if spec.is_primary else other_q).append(spec)

    # Rebuild histories from saved primaries, then open every reachable turn.
    for i, state in enumerate(states):
        turn = 1
        while turn <= turns_total[i]:
            key = call_key(state.unit, CallSpec(i, turn, "dialogue", "primary", 0))
            open_turn(i, turn)
            if key in saved:
                state.primaries[turn] = saved[key]["raw_text"]
                if saved[key].get("truncated"):
                    state.truncated_primaries.add(turn)
                turn += 1
            else:
                break
        # Control calls never depend on dialogue history, so a control-only
        # filter opens every remaining turn regardless of primary progress.
        if arms is not None and "dialogue" not in arms:
            for later in range(turn + 1, turns_total[i] + 1):
                open_turn(i, later)

    limiter = TokenRateLimiter(tokens_per_minute, clock=clock)
    consecutive_failures = 0
    tripped = None
    completed = len(saved)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    with (checkpoint.open("a", encoding="utf-8") as stream,
          ThreadPoolExecutor(max_workers=workers) as pool):
        in_flight: dict[Any, tuple[CallSpec, list, str, int]] = {}

        def submit_ready() -> None:
            while tripped is None and len(in_flight) < workers and (primary_q or other_q):
                queue = primary_q if primary_q else other_q
                spec = queue[0]
                state = states[spec.unit]
                messages = build_messages(state, spec)
                if not limiter.try_acquire(estimate_tokens(messages)):
                    return
                queue.popleft()
                key = call_key(state.unit, spec)
                seed = cs._derive_seed(fingerprint, *key) % (2 ** 31)
                request_hash = g1.messages_sha256(messages)
                future = pool.submit(clients[state.unit["model"]].generate_messages,
                                     messages, seed=seed, gold="")
                in_flight[future] = (spec, messages, request_hash, seed)

        submit_ready()
        while in_flight or ((primary_q or other_q) and tripped is None):
            if not in_flight:
                sleep(max(0.05, min(5.0, limiter.seconds_until_free())))
                submit_ready()
                continue
            done, _ = wait(in_flight, timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                spec, messages, request_hash, seed = in_flight.pop(future)
                state = states[spec.unit]
                sample = future.result()
                turn = state.instance["turns"][spec.turn - 1]
                scored = g1.score_response(sample.raw_text, turn["gold"])
                finish = getattr(sample, "finish_reason", None)
                row = {"protocol_id": PROTOCOL_ID, "fingerprint": fingerprint,
                       "model": state.unit["model"],
                       "provider_model": models[state.unit["model"]]["provider_model"],
                       "dialogue_id": state.unit["dialogue_id"],
                       "template": state.unit["template"], "scenario": state.unit["scenario"],
                       "turn": spec.turn, "regime": turn["regime"], "level": turn["level"],
                       "family": turn["family"], "arm": spec.arm, "role": spec.role,
                       "call_index": spec.index, "gold": turn["gold"],
                       "messages_sha256": request_hash,
                       "prompt_chars": sum(len(m["content"]) for m in messages),
                       "call_seed": seed, "raw_text": sample.raw_text, **scored,
                       "ok": bool(sample.ok), "n_tokens": int(sample.n_tokens),
                       "max_tokens": int(max_tokens), "finish_reason": finish,
                       "truncated": int(sample.n_tokens) >= int(max_tokens)
                       or finish == "length",
                       "latency_s": sample.latency_s}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                if not row["ok"]:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and tripped is None:
                        tripped = (f"{consecutive_failures} consecutive failed calls; last: "
                                   f"{sample.raw_text[:300]}. Completed calls are saved; "
                                   "rerun the same command to resume.")
                    continue
                consecutive_failures = 0
                saved[call_key(state.unit, spec)] = row
                completed += 1
                if spec.is_primary:
                    state.primaries[spec.turn] = sample.raw_text
                    if row["truncated"]:
                        state.truncated_primaries.add(spec.turn)
                    if spec.turn < turns_total[spec.unit]:
                        open_turn(spec.unit, spec.turn + 1)
                if callback is not None:
                    callback(completed)
            submit_ready()

    if tripped is not None:
        raise CircuitBreakerOpen(tripped)
    if arms is not None:
        expected = {key for key in expected if key[3] in arms}
    missing = len(expected) - len([key for key in saved if key in expected])
    result = {"expected": len(expected), "completed": len(expected) - missing,
              "missing": missing}
    if arms is not None:
        result["arms"] = list(arms)
    return result


# ── budget ──────────────────────────────────────────────────────────────────

def estimate_unit_budget(instance: Mapping[str, Any], k: int, l_paraphrases: int, control: int,
                         assumed_reply_chars: int = 1000) -> dict[str, float]:
    """Character/4 proxy; replies assumed ``assumed_reply_chars`` long."""
    overhead = len(SYSTEM_PROMPT) + len(RESPONSE_CONTRACT) + 80
    history_chars = 0
    input_chars = 0.0
    calls = 0
    max_context = 0
    for turn in instance["turns"]:
        base = overhead + history_chars + 40
        user = len(turn["text"])
        paraphrase = sum(len(p) for p in turn["paraphrases"][:l_paraphrases])
        input_chars += k * (base + user) + l_paraphrases * base + paraphrase
        input_chars += control * (overhead + len(turn["state_summary_before"]) + user)
        calls += k + l_paraphrases + control
        max_context = max(max_context, base + user)
        history_chars += user + assumed_reply_chars + 40
    return {"calls": calls, "input_tokens": input_chars / CHARS_PER_TOKEN,
            "max_context_tokens": max_context / CHARS_PER_TOKEN}


# ── user-only history arm (CDR_STREAM_USER_ONLY_PROTOCOL.md) ─────────────────

USER_ONLY_ARM = "user_only"


def user_only_messages(instance: Mapping[str, Any], turn: int) -> list[dict[str, str]]:
    """Dialogue-arm format with every prior user turn and no assistant replies."""
    history = [{"role": "user", "content": spec["text"]}
               for spec in instance["turns"][:turn - 1]]
    return dialogue_messages(history, instance["turns"][turn - 1]["text"])


def acquire_user_only(units: Sequence[Mapping[str, Any]], instances: Sequence[Mapping[str, Any]],
                      clients: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]],
                      checkpoint: Path, fingerprint: str, max_tokens: int, *,
                      workers: int = 1, tokens_per_minute: int = 600_000, callback=None,
                      clock=time.monotonic, sleep=time.sleep) -> dict[str, Any]:
    """Resumable acquisition of the user-only arm (one primary per turn).

    No call depends on a model output, so every turn of every unit is queued at
    once. Rows use the stream-row layout with ``arm == "user_only"``.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    saved, _ = load_checkpoint(checkpoint, fingerprint)
    plan = [(i, t) for i, instance in enumerate(instances)
            for t in range(1, len(instance["turns"]) + 1)]
    expected = {(units[i]["model"], units[i]["dialogue_id"], t, USER_ONLY_ARM, "primary", 0)
                for i, t in plan}
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen user-only plan")
    queue = deque((i, t) for i, t in plan
                  if (units[i]["model"], units[i]["dialogue_id"], t, USER_ONLY_ARM,
                      "primary", 0) not in saved)
    limiter = TokenRateLimiter(tokens_per_minute, clock=clock)
    completed, consecutive_failures, tripped = len(saved), 0, None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (checkpoint.open("a", encoding="utf-8") as stream,
          ThreadPoolExecutor(max_workers=workers) as pool):
        in_flight: dict[Any, tuple[int, int, str, int]] = {}

        def submit_ready() -> None:
            while tripped is None and len(in_flight) < workers and queue:
                i, t = queue[0]
                messages = user_only_messages(instances[i], t)
                if not limiter.try_acquire(estimate_tokens(messages)):
                    return
                queue.popleft()
                key = (units[i]["model"], units[i]["dialogue_id"], t, USER_ONLY_ARM, "primary", 0)
                seed = cs._derive_seed(fingerprint, *key) % (2 ** 31)
                future = pool.submit(clients[units[i]["model"]].generate_messages, messages,
                                     seed=seed, gold="")
                in_flight[future] = (i, t, g1.messages_sha256(messages), seed,
                                     sum(len(m["content"]) for m in messages))

        submit_ready()
        while in_flight or (queue and tripped is None):
            if not in_flight:
                sleep(max(0.05, min(5.0, limiter.seconds_until_free())))
                submit_ready()
                continue
            done, _ = wait(in_flight, timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                i, t, request_hash, seed, chars = in_flight.pop(future)
                sample = future.result()
                unit, turn = units[i], instances[i]["turns"][t - 1]
                finish = getattr(sample, "finish_reason", None)
                row = {"protocol_id": "CDR-Stream-v1-user-only", "fingerprint": fingerprint,
                       "model": unit["model"],
                       "provider_model": models[unit["model"]]["provider_model"],
                       "dialogue_id": unit["dialogue_id"], "template": unit["template"],
                       "scenario": unit["scenario"], "turn": t, "regime": turn["regime"],
                       "level": turn["level"], "family": turn["family"],
                       "arm": USER_ONLY_ARM, "role": "primary", "call_index": 0,
                       "gold": turn["gold"], "messages_sha256": request_hash,
                       "prompt_chars": chars, "call_seed": seed, "raw_text": sample.raw_text,
                       **g1.score_response(sample.raw_text, turn["gold"]),
                       "ok": bool(sample.ok), "n_tokens": int(sample.n_tokens),
                       "max_tokens": int(max_tokens), "finish_reason": finish,
                       "truncated": int(sample.n_tokens) >= int(max_tokens)
                       or finish == "length", "latency_s": sample.latency_s}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                if not row["ok"]:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and tripped is None:
                        tripped = (f"{consecutive_failures} consecutive failed calls; last: "
                                   f"{sample.raw_text[:300]}. Completed calls are saved; "
                                   "rerun the same command to resume.")
                    continue
                consecutive_failures = 0
                saved[row_key(row)] = row
                completed += 1
                if callback is not None:
                    callback(completed)
            submit_ready()
    if tripped is not None:
        raise CircuitBreakerOpen(tripped)
    missing = len(expected) - len([k for k in saved if k in expected])
    return {"expected": len(expected), "completed": len(expected) - missing, "missing": missing}
