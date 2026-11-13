"""LHE-RQ-v1 acquisition: resumable packed-history conversations with K/L calls.

A *unit* is one (model, conversation). At turn t the runner issues the primary
(call 0), K-1 same-context repeats and L reworded-turn calls, all with the
packed history of turns 1..t-1. Only the primary reply enters later history.
Turns of one conversation are sequential; conversations and models run in
parallel; primaries are scheduled first. Every call is written raw before it
is used; successful calls are never repeated; failed calls are retried on the
next run; a circuit breaker stops repeated failures. Checkpoint handling, the
token-rate limiter and the breaker are shared with the CDR-Stream runner.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reasoning_monitor import lhe_early_warning as lhe
from reasoning_monitor.cdr_stream_runner import (
    MAX_CONSECUTIVE_FAILURES,
    CircuitBreakerOpen,
    TokenRateLimiter,
    estimate_tokens,
    load_checkpoint,
)

EMPTY_PRIMARY_PLACEHOLDER = "[EMPTY_PRIMARY_RESPONSE]"
TRUNCATED_PRIMARY_PLACEHOLDER = "[TRUNCATED_PRIMARY_RESPONSE]"


@dataclass(frozen=True)
class Call:
    unit: int
    turn: int
    role: str      # "primary", "repeat", "paraphrase"
    index: int     # repeat 1..K-1; paraphrase wording 1..L

    @property
    def is_primary(self) -> bool:
        return self.role == "primary"


def turn_calls(unit: int, turn: int, k_samples: int, l_paraphrases: int) -> list[Call]:
    calls = [Call(unit, turn, "primary", 0)]
    calls += [Call(unit, turn, "repeat", i) for i in range(1, k_samples)]
    calls += [Call(unit, turn, "paraphrase", i) for i in range(1, l_paraphrases + 1)]
    return calls


def call_key(unit: Mapping[str, Any], call: Call) -> tuple:
    # Same tuple layout as cdr_stream_runner.row_key, so load_checkpoint is shared.
    return (unit["model"], unit["dialogue_id"], call.turn, "dialogue", call.role, call.index)


def derive_seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def messages_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    return hashlib.sha256(json.dumps(list(messages), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass
class UnitState:
    unit: Mapping[str, Any]
    system: str
    primaries: dict[int, str] = field(default_factory=dict)
    truncated: set[int] = field(default_factory=set)

    def history_before(self, turn: int) -> list[dict[str, str]]:
        history = []
        for t in range(1, turn):
            keys = self.unit["conversation"]["turns"][t - 1]["keys"]
            history.append({"role": "user", "content": lhe.turn_text(keys)})
            reply = self.primaries[t]
            if t in self.truncated:
                reply = TRUNCATED_PRIMARY_PLACEHOLDER
            history.append({"role": "assistant",
                            "content": reply if reply.strip() else EMPTY_PRIMARY_PLACEHOLDER})
        return history

    def messages(self, call: Call) -> list[dict[str, str]]:
        keys = self.unit["conversation"]["turns"][call.turn - 1]["keys"]
        variant = call.index if call.role == "paraphrase" else 0
        return lhe.pack_messages(self.system, self.history_before(call.turn),
                                 lhe.turn_text(keys, variant), self.unit.get("mode", "cot"))


def build_units(models: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]],
                ids: Sequence[int], horizon: int) -> list[dict[str, Any]]:
    by_id = {int(s["id"]): s for s in samples}
    units = []
    for model in models:
        for cid in ids:
            conversation = lhe.build_conversation(by_id[cid], int(model["k"]), horizon)
            units.append({"model": model["name"], "provider_model": model["provider_model"],
                          "k": int(model["k"]), "mode": model.get("mode", "cot"),
                          "dialogue_id": f"lhe-{cid:03d}",
                          "conversation": conversation})
    return units


def acquire(units: Sequence[Mapping[str, Any]], dictionary: Mapping[str, int],
            clients: Mapping[str, Any], checkpoint: Path, fingerprint: str, max_tokens: int, *,
            k_samples: int, l_paraphrases: int, workers: int = 1,
            tokens_per_minute: int = 600_000, callback: Callable[[int], None] | None = None,
            clock=time.monotonic, sleep=time.sleep) -> dict[str, int]:
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    saved, _ = load_checkpoint(checkpoint, fingerprint)
    states = [UnitState(u, lhe.system_prompt(dictionary, u["k"], u.get("mode", "cot")))
              for u in units]
    horizon = [len(u["conversation"]["turns"]) for u in units]
    expected = {call_key(s.unit, c) for i, s in enumerate(states)
                for t in range(1, horizon[i] + 1)
                for c in turn_calls(i, t, k_samples, l_paraphrases)}
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen plan")

    primary_q: deque[Call] = deque()
    other_q: deque[Call] = deque()

    def open_turn(i: int, turn: int) -> None:
        for call in turn_calls(i, turn, k_samples, l_paraphrases):
            if call_key(states[i].unit, call) not in saved:
                (primary_q if call.is_primary else other_q).append(call)

    for i, state in enumerate(states):          # rebuild histories from saved primaries
        turn = 1
        while turn <= horizon[i]:
            open_turn(i, turn)
            key = call_key(state.unit, Call(i, turn, "primary", 0))
            if key not in saved:
                break
            state.primaries[turn] = saved[key]["raw_text"]
            if saved[key].get("truncated"):
                state.truncated.add(turn)
            turn += 1

    limiter = TokenRateLimiter(tokens_per_minute, clock=clock)
    failures, tripped = 0, None
    completed = len(saved)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (checkpoint.open("a", encoding="utf-8") as stream,
          ThreadPoolExecutor(max_workers=workers) as pool):
        in_flight: dict[Any, tuple[Call, str, int, int]] = {}

        def submit_ready() -> None:
            while tripped is None and len(in_flight) < workers and (primary_q or other_q):
                queue = primary_q if primary_q else other_q
                call = queue[0]
                state = states[call.unit]
                messages = state.messages(call)
                if not limiter.try_acquire(estimate_tokens(messages)):
                    return
                queue.popleft()
                seed = derive_seed(fingerprint, *call_key(state.unit, call))
                future = pool.submit(clients[state.unit["model"]].generate_messages,
                                     messages, seed=seed, gold="")
                in_flight[future] = (call, messages_sha256(messages),
                                     sum(len(m["content"]) for m in messages), seed)

        submit_ready()
        while in_flight or ((primary_q or other_q) and tripped is None):
            if not in_flight:
                sleep(max(0.05, min(5.0, limiter.seconds_until_free())))
                submit_ready()
                continue
            done, _ = wait(in_flight, timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                call, request_hash, prompt_chars, seed = in_flight.pop(future)
                state = states[call.unit]
                sample = future.result()
                spec = state.unit["conversation"]["turns"][call.turn - 1]
                finish = getattr(sample, "finish_reason", None)
                answer, strict = lhe.parse_answer(sample.raw_text)
                row = {"protocol_id": lhe.PROTOCOL_ID, "fingerprint": fingerprint,
                       "model": state.unit["model"],
                       "provider_model": state.unit["provider_model"],
                       "dialogue_id": state.unit["dialogue_id"], "k": state.unit["k"],
                       "mode": state.unit.get("mode", "cot"),
                       "turn": call.turn, "arm": "dialogue", "role": call.role,
                       "call_index": call.index, "keys": spec["keys"],
                       "true_sum": spec["true_sum"], "delta": spec["delta"],
                       "messages_sha256": request_hash, "prompt_chars": prompt_chars,
                       "call_seed": seed, "raw_text": sample.raw_text,
                       "answer": answer, "strict": strict, "ok": bool(sample.ok),
                       "n_tokens": int(sample.n_tokens), "max_tokens": int(max_tokens),
                       "finish_reason": finish,
                       "truncated": int(sample.n_tokens) >= int(max_tokens)
                       or finish == "length", "latency_s": sample.latency_s}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                if not row["ok"]:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES and tripped is None:
                        tripped = (f"{failures} consecutive failed calls; last: "
                                   f"{sample.raw_text[:300]}. Completed calls are saved; "
                                   "rerun the same command to resume.")
                    continue
                failures = 0
                saved[call_key(state.unit, call)] = row
                completed += 1
                if call.is_primary:
                    state.primaries[call.turn] = sample.raw_text
                    if row["truncated"]:
                        state.truncated.add(call.turn)
                    if call.turn < horizon[call.unit]:
                        open_turn(call.unit, call.turn + 1)
                if callback is not None:
                    callback(completed)
            submit_ready()

    if tripped is not None:
        raise CircuitBreakerOpen(tripped)
    missing = len(expected) - len([key for key in saved if key in expected])
    return {"expected": len(expected), "completed": len(expected) - missing, "missing": missing}
