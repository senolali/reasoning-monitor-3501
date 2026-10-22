import copy
import json
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from reasoning_monitor import cdr_stream_g1 as g1
from reasoning_monitor import cdr_stream_runner as runner
from reasoning_monitor.data import cdr_stream as cs

MAPPINGS = {"models": {
    "a": {"retained": True, "mappings": {"budget_ledger": {"E": 1, "M": 4, "H": 6},
                                         "milestone_chains": {"E": 1, "M": 3, "H": 8}}},
    "b": {"retained": True, "mappings": {"inventory_logistics": {"E": 2, "M": 4, "H": 8}}},
}}
MODELS = {"a": {"name": "a", "provider_model": "pa"}, "b": {"name": "b", "provider_model": "pb"}}


@pytest.fixture(scope="module")
def bank():
    return cs.generate_bank(version="v2")


def small_units(bank, turns=4):
    units = runner.select_units(bank, MAPPINGS, ["a", "b"], split="calibration",
                                scenarios=("S1",), per_model=1, indices=[0],
                                templates={"a": "milestone_chains", "b": "inventory_logistics"})
    skeletons = {d["dialogue_id"]: d for d in bank["dialogues"]}
    instances = []
    for unit in units:
        instance = copy.deepcopy(cs.instantiate(skeletons[unit["dialogue_id"]],
                                                unit["mapping"], "v2"))
        instance["turns"] = instance["turns"][:turns]
        instances.append(instance)
    return units, instances


class RecordingClient:
    def __init__(self, fail_after=None, empty_primary_turn1=False, truncate_turn1=False):
        self.lock = threading.Lock()
        self.calls = []
        self.fail_after = fail_after
        self.empty_primary_turn1 = empty_primary_turn1
        self.truncate_turn1 = truncate_turn1

    def generate_messages(self, messages, seed=0, gold=""):
        with self.lock:
            self.calls.append((seed, messages))
            n = len(self.calls)
        if self.fail_after is not None and n > self.fail_after:
            return SimpleNamespace(raw_text="[CALL_FAILED] HTTP 503", ok=False, n_tokens=0,
                                   finish_reason=None, latency_s=0.0)
        content = messages[-1]["content"]
        if self.truncate_turn1 and "Conversation so far: none." in content:
            return SimpleNamespace(raw_text="loop loop loop", ok=True, n_tokens=2048,
                                   finish_reason="stop", latency_s=0.0)
        if self.empty_primary_turn1 and "Conversation so far: none." in content:
            return SimpleNamespace(raw_text="", ok=True, n_tokens=1, finish_reason="stop",
                                   latency_s=0.0)
        return SimpleNamespace(raw_text=f"1. step\nFINAL_ANSWER: {seed % 997}", ok=True,
                               n_tokens=12, finish_reason="stop", latency_s=0.0)


def run(units, instances, client, path, workers=1, **kwargs):
    return runner.acquire_streams(units, instances, {"a": client, "b": client}, MODELS,
                                  path, "fp", 2048, k=3, l_paraphrases=2, control=1,
                                  workers=workers, **kwargs)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_history_contains_only_primary_replies(bank, tmp_path):
    units, instances = small_units(bank)
    client = RecordingClient()
    result = run(units, instances, client, tmp_path / "r.jsonl")
    assert result == {"expected": 2 * 4 * 6, "completed": 48, "missing": 0}
    recorded = rows(tmp_path / "r.jsonl")
    primaries = {(r["dialogue_id"], r["turn"]): r["raw_text"] for r in recorded
                 if r["arm"] == "dialogue" and r["role"] == "primary"}
    non_primary = {r["raw_text"] for r in recorded if r["role"] != "primary"} - set(
        primaries.values())
    for seed, messages in client.calls:
        content = messages[-1]["content"]
        if not content.startswith("Conversation so far (oldest first)"):
            continue
        packed = json.loads(content.split("as JSON:\n", 1)[1].split("\n\nNew message:", 1)[0])
        assistant = [m["content"] for m in packed if m["role"] == "assistant"]
        assert all(text not in non_primary for text in assistant)
        assert all(text in primaries.values() for text in assistant)
    # every recorded request hash matches a rebuild from recorded primaries
    states = [runner.UnitState(u, i) for u, i in zip(units, instances, strict=True)]
    for state in states:
        for (did, turn), text in primaries.items():
            if did == state.unit["dialogue_id"]:
                state.primaries[turn] = text
    index = {u["dialogue_id"]: n for n, u in enumerate(units)}
    for r in recorded:
        spec = runner.CallSpec(index[r["dialogue_id"]], r["turn"], r["arm"], r["role"],
                               r["call_index"])
        rebuilt = runner.build_messages(states[spec.unit], spec)
        assert g1.messages_sha256(rebuilt) == r["messages_sha256"]


def test_control_arm_uses_frozen_g1_prompt(bank):
    units, instances = small_units(bank, turns=2)
    state = runner.UnitState(units[0], instances[0])
    spec = runner.CallSpec(0, 2, "control", "primary", 0)
    turn = instances[0]["turns"][1]
    assert runner.build_messages(state, spec) == g1.control_messages(
        turn["state_summary_before"], turn["text"])


def test_resume_after_breaker_repeats_no_successful_call(bank, tmp_path):
    units, instances = small_units(bank)
    path = tmp_path / "r.jsonl"
    with pytest.raises(runner.CircuitBreakerOpen):
        run(units, instances, RecordingClient(fail_after=15), path)
    saved, failures = runner.load_checkpoint(path, "fp")
    assert len(saved) == 15 and failures >= runner.MAX_CONSECUTIVE_FAILURES
    good = RecordingClient()
    result = run(units, instances, good, path)
    assert result["missing"] == 0 and len(good.calls) == 48 - 15
    saved, _ = runner.load_checkpoint(path, "fp")
    assert len(saved) == 48


def test_empty_primary_uses_placeholder_in_later_history(bank, tmp_path):
    units, instances = small_units(bank, turns=2)
    client = RecordingClient(empty_primary_turn1=True)
    run(units, instances, client, tmp_path / "r.jsonl")
    later = [m[-1]["content"] for _, m in client.calls
             if m[-1]["content"].startswith("Conversation so far (oldest first)")]
    assert later and all(runner.EMPTY_PRIMARY_PLACEHOLDER in c for c in later)


def test_parallel_run_matches_sequential(bank, tmp_path):
    units, instances = small_units(bank)
    run(units, instances, RecordingClient(), tmp_path / "s.jsonl", workers=1)
    run(units, instances, RecordingClient(), tmp_path / "p.jsonl", workers=8)

    def key(r):
        return (r["dialogue_id"], r["turn"], r["arm"], r["role"], r["call_index"],
                r["messages_sha256"], r["raw_text"])

    assert sorted(map(key, rows(tmp_path / "s.jsonl"))) == sorted(
        map(key, rows(tmp_path / "p.jsonl")))


def test_rate_limiter_window():
    now = [0.0]
    limiter = runner.TokenRateLimiter(1000, clock=lambda: now[0])
    assert limiter.try_acquire(600)
    assert not limiter.try_acquire(600)
    now[0] = 61.0
    assert limiter.try_acquire(600)
    assert runner.TokenRateLimiter(10).try_acquire(50)  # oversize call allowed when idle


def test_select_units_balances_cells(bank):
    core = ("S0-E", "S0-H", "S1", "S2", "S3")
    units = runner.select_units(bank, MAPPINGS, ["a"], split="holdout", scenarios=core,
                                per_model=20)
    assert Counter(u["model"] for u in units) == {"a": 20}
    a_cells = Counter((u["scenario"], u["template"]) for u in units)
    assert set(a_cells.values()) == {2} and len(a_cells) == 10  # 2 templates x 5 scenarios
    one_template = runner.select_units(bank, MAPPINGS, ["b"], split="holdout", scenarios=core,
                                       per_model=15)
    assert len(one_template) == 15  # all three holdout dialogues per cell
    with pytest.raises(ValueError):
        runner.select_units(bank, MAPPINGS, ["b"], split="holdout",
                            scenarios=("S1",), per_model=30)


def test_truncated_primary_masked_only_when_opted_in(bank, tmp_path):
    units, instances = small_units(bank, turns=3)
    masked = RecordingClient(truncate_turn1=True)
    run(units, instances, masked, tmp_path / "m.jsonl", mask_truncated_primaries=True)
    later = [m[-1]["content"] for _, m in masked.calls
             if m[-1]["content"].startswith("Conversation so far (oldest first)")]
    assert later and all(runner.TRUNCATED_PRIMARY_PLACEHOLDER in c for c in later)
    assert all("loop loop loop" not in c for c in later)
    # resume rebuilds the same masked histories from the checkpoint
    saved, _ = runner.load_checkpoint(tmp_path / "m.jsonl", "fp")
    assert any(r["truncated"] for r in saved.values())
    default = RecordingClient(truncate_turn1=True)
    run(units, instances, default, tmp_path / "d.jsonl")
    later = [m[-1]["content"] for _, m in default.calls
             if m[-1]["content"].startswith("Conversation so far (oldest first)")]
    assert later and all("loop loop loop" in c for c in later)


def test_arm_filter_acquires_only_control_then_resume_completes(bank, tmp_path):
    units, instances = small_units(bank)
    path = tmp_path / "r.jsonl"
    first = RecordingClient()
    result = run(units, instances, first, path, arms=("control",))
    assert result["expected"] == 2 * 4 and result["missing"] == 0
    assert {r["arm"] for r in rows(path)} == {"control"}
    assert len(first.calls) == 8
    # A later unfiltered run fills every other call and repeats no control call.
    second = RecordingClient()
    result = run(units, instances, second, path)
    assert result["missing"] == 0 and len(second.calls) == 48 - 8
    keys = [(r["dialogue_id"], r["turn"], r["arm"], r["role"], r["call_index"])
            for r in rows(path)]
    assert len(keys) == len(set(keys)) == 48
