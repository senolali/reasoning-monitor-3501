from types import SimpleNamespace

import pytest

from reasoning_monitor import cdr_stream_g1 as g1
from reasoning_monitor.data import cdr_stream as cs


def test_isotonic_fit_is_non_increasing():
    fitted = g1.isotonic_decreasing([0.9, 0.95, 0.7, 0.75, 0.4], [30] * 5)
    assert all(a >= b for a, b in zip(fitted, fitted[1:], strict=False))
    assert fitted[0] == pytest.approx(0.925)


def test_trend_test_detects_decrease_only():
    levels = [1, 2, 3, 4]
    assert g1.cochran_armitage_decreasing(levels, [29, 24, 18, 10], [30] * 4) < 0.001
    assert g1.cochran_armitage_decreasing(levels, [10, 18, 24, 29], [30] * 4) > 0.99


def test_mapping_respects_bands_order_and_gaps():
    fitted = {1: 0.97, 2: 0.86, 3: 0.80, 4: 0.71, 5: 0.62, 6: 0.51, 7: 0.40, 8: 0.30}
    assert g1.select_mapping(fitted) == {"E": 2, "M": 4, "H": 6}
    ceiling = {level: 0.95 for level in range(1, 9)}
    assert g1.select_mapping(ceiling) is None


def _rows(cq_by_level, contingency=False):
    rows = []
    for level, cq in cq_by_level.items():
        correct = round(cq * 30)
        for i in range(30):
            rows.append({"level": level, "correct": i < correct, "contract_valid": True,
                         "contingency": level > 8 if contingency else False})
    return rows


def test_cell_statuses_mapped_g1b_and_excluded():
    mapped = g1.assess_cell(_rows({1: .97, 2: .87, 3: .8, 4: .7, 5: .6, 6: .5, 7: .4,
                                   8: .3}), False)
    assert mapped["status"] == "mapped"
    ceiling = g1.assess_cell(_rows({1: 1, 2: 1, 3: .97, 4: .93, 5: .9, 6: .87, 7: .83,
                                    8: .8}), False)
    assert ceiling["status"] == "needs_G1b"
    flat = g1.assess_cell(_rows({level: .5 for level in range(1, 9)}), False)
    assert flat["status"] == "excluded"
    extended = g1.assess_cell(_rows({1: 1, 2: 1, 3: .97, 4: .93, 5: .9, 6: .87, 7: .8,
                                     8: .77, 9: .6, 10: .5}, contingency=True), True)
    assert extended["status"] == "mapped" and extended["mapping"]["H"] >= 9


def test_items_are_valid_for_extreme_levels():
    items = [g1.make_item(t, level, family, 0) for t in cs.TEMPLATES
             for level in (1, 8, 10) for family in cs.FAMILIES]
    problems = [p for p in g1.validate_items(items) if p != "unequal cell sizes"]
    assert problems == []
    assert {i["contingency"] for i in items if i["level"] == 10} == {True}


def test_score_response_uses_marker_only():
    assert g1.score_response("1. add\nFINAL_ANSWER: 42", 42)["correct"]
    assert not g1.score_response("The answer is 42.", 42)["contract_valid"]


class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate_messages(self, messages, seed=0, gold=""):
        self.calls += 1
        return SimpleNamespace(raw_text="1. compute\nFINAL_ANSWER: 7", ok=True, n_tokens=20,
                               finish_reason="stop", latency_s=0.01)


def test_acquisition_resumes_without_new_calls(tmp_path):
    items = [g1.make_item("budget_ledger", level, "SUM", 0) for level in (1, 2, 9)]
    for item in items:
        item["contingency"] = item["level"] > 8
    models = [{"name": "m", "provider_model": "p"}]
    plan = g1.planned_calls(items, models, order_seed=1)
    assert len(plan) == 2  # contingency item excluded without a G1b cell
    client = FakeClient()
    checkpoint = tmp_path / "records.jsonl"
    rows = g1.acquire(plan, {"m": client}, checkpoint, "fp", 2048)
    assert client.calls == 2 and all(r["contract_valid"] for r in rows)
    g1.acquire(plan, {"m": client}, checkpoint, "fp", 2048)
    assert client.calls == 2
    with pytest.raises(ValueError):
        g1.acquire(plan, {"m": client}, checkpoint, "other", 2048)
    plan_b = g1.planned_calls(items, models, ["m|budget_ledger"], order_seed=1)
    g1.acquire(plan_b, {"m": client}, checkpoint, "fp", 2048)
    assert client.calls == 3


class FlakyClient:
    """Fails the first `fail_first` calls, then succeeds."""

    def __init__(self, fail_first):
        self.calls = 0
        self.fail_first = fail_first

    def generate_messages(self, messages, seed=0, gold=""):
        self.calls += 1
        if self.calls <= self.fail_first:
            return SimpleNamespace(raw_text="[CALL_FAILED] HTTP 503", ok=False, n_tokens=0,
                                   finish_reason=None, latency_s=0.0)
        return SimpleNamespace(raw_text="1. x\nFINAL_ANSWER: 1", ok=True, n_tokens=10,
                               finish_reason="stop", latency_s=0.0)


def _small_plan():
    items = [g1.make_item("lab_protocol", level, "MAX", 0) for level in (1, 2, 3)]
    return g1.planned_calls(items, [{"name": "m", "provider_model": "p"}], order_seed=2)


def test_failed_calls_are_retried_on_resume_not_cached(tmp_path):
    plan = _small_plan()
    checkpoint = tmp_path / "records.jsonl"
    client = FlakyClient(fail_first=1)
    first = g1.acquire(plan, {"m": client}, checkpoint, "fp", 2048)
    assert len(first) == 2  # one call failed and is not counted as completed
    saved, failures = g1.load_checkpoint(checkpoint, "fp")
    assert len(saved) == 2 and failures == 1
    second = g1.acquire(plan, {"m": client}, checkpoint, "fp", 2048)
    assert len(second) == 3 and client.calls == 4  # only the failed call was retried


def test_circuit_breaker_stops_after_consecutive_failures(tmp_path):
    items = [g1.make_item("lab_protocol", 1, "SUM", i) for i in range(5)]
    items += [g1.make_item("lab_protocol", 2, "SUM", i) for i in range(5)]
    items += [g1.make_item("lab_protocol", 3, "SUM", i) for i in range(2)]
    plan = g1.planned_calls(items, [{"name": "m", "provider_model": "p"}], order_seed=3)
    client = FlakyClient(fail_first=100)
    with pytest.raises(g1.CircuitBreakerOpen):
        g1.acquire(plan, {"m": client}, tmp_path / "r.jsonl", "fp", 2048)
    assert client.calls == g1.MAX_CONSECUTIVE_FAILURES


class ThreadSafeClient:
    """Answers with the item's level so rows can be checked after parallel runs."""

    def __init__(self, fail_all=False):
        import threading

        self.lock = threading.Lock()
        self.calls = 0
        self.seeds = []
        self.fail_all = fail_all

    def generate_messages(self, messages, seed=0, gold=""):
        import time

        with self.lock:
            self.calls += 1
            self.seeds.append(seed)
        time.sleep(0.002)
        if self.fail_all:
            return SimpleNamespace(raw_text="[CALL_FAILED] HTTP 503", ok=False, n_tokens=0,
                                   finish_reason=None, latency_s=0.0)
        return SimpleNamespace(raw_text="1. x\nFINAL_ANSWER: 3", ok=True, n_tokens=10,
                               finish_reason="stop", latency_s=0.0)


def _plan(n_items=24):
    items = [g1.make_item("inventory_logistics", 1 + i % 8, cs.FAMILIES[i % 6], i // 8)
             for i in range(n_items)]
    return g1.planned_calls(items, [{"name": "m", "provider_model": "p"}], order_seed=4)


def test_parallel_acquisition_matches_sequential(tmp_path):
    plan = _plan()
    sequential = ThreadSafeClient()
    parallel = ThreadSafeClient()
    rows_seq = g1.acquire(plan, {"m": sequential}, tmp_path / "s.jsonl", "fp", 2048)
    rows_par = g1.acquire(plan, {"m": parallel}, tmp_path / "p.jsonl", "fp", 2048,
                          workers=8)
    strip = [{k: v for k, v in r.items() if k != "latency_s"} for r in rows_seq]
    assert strip == [{k: v for k, v in r.items() if k != "latency_s"} for r in rows_par]
    assert sorted(sequential.seeds) == sorted(parallel.seeds)
    lines = (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(plan)  # one intact JSON line per call
    g1.acquire(plan, {"m": parallel}, tmp_path / "p.jsonl", "fp", 2048, workers=8)
    assert parallel.calls == len(plan)  # resume makes no new call


def test_parallel_circuit_breaker_stops_quickly(tmp_path):
    client = ThreadSafeClient(fail_all=True)
    with pytest.raises(g1.CircuitBreakerOpen):
        g1.acquire(_plan(40), {"m": client}, tmp_path / "r.jsonl", "fp", 2048, workers=8)
    assert g1.MAX_CONSECUTIVE_FAILURES <= client.calls < g1.MAX_CONSECUTIVE_FAILURES + 8
    saved, failures = g1.load_checkpoint(tmp_path / "r.jsonl", "fp")
    assert not saved and failures == client.calls


def test_partial_final_line_is_moved_aside(tmp_path):
    plan = _small_plan()
    checkpoint = tmp_path / "records.jsonl"
    g1.acquire(plan[:2], {"m": FakeClient()}, checkpoint, "fp", 2048)
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write('{"model": "m", "item_id": "trunc')
    saved, _ = g1.load_checkpoint(checkpoint, "fp")
    assert len(saved) == 2
    assert checkpoint.read_text(encoding="utf-8").endswith("\n")
    assert (tmp_path / "records.partial.jsonl").exists()


def test_v1_instrument_is_unchanged_by_versioning():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "configs/selections/cdr_stream_v1_g1_items.json")
                        .read_text(encoding="utf-8"))["items_sha256"]
    assert g1.items_sha256(g1.generate_items()) == frozen
    assert cs.bank_sha256(cs.generate_bank()) == (
        "e1e045b26de43b36e85c60b9a9b38cc1df4a4310e2d04013e03a09a21ed0994c")


def test_v2_ladder_changes_one_axis_per_step_and_has_two_terms_minimum():
    ladder = cs.LADDERS["v2"]
    primary = cs.PRIMARY_LEVELS["v2"]
    for low, high in zip(primary, primary[1:], strict=False):
        changes = [b - a for a, b in zip(ladder[low], ladder[high], strict=True)]
        assert sum(c != 0 for c in changes) == 1 and all(c >= 0 for c in changes)
    assert min(spec[0] for spec in ladder.values()) == 2
    assert set(cs.CONTINGENCY["v2"]) == {13, 14}


def test_v2_schedule_record_states_finish_days_and_items_validate():
    spec = g1.SPECS["v2"]
    items = [g1.make_item(t, level, family, 0, spec) for t in cs.TEMPLATES
             for level in (1, 12, 14) for family in cs.FAMILIES]
    assert [p for p in g1.validate_items(items) if p != "unequal cell sizes"] == []
    schedule = [i for i in items if i["template"] == "schedule_constraints"]
    assert all("finishes on day" in i["state_summary"] for i in schedule)
    assert all(i["axes"]["query_terms"] >= 2 for i in items)


def test_v2_stream_instantiation_passes_g0():
    skeleton = next(d for d in cs.generate_bank()["dialogues"]
                    if d["template"] == "schedule_constraints" and d["scenario"] == "S1")
    instance = cs.instantiate(skeleton, {"E": 2, "M": 6, "H": 11}, "v2")
    problems, _ = cs.validate_instance(instance)
    assert problems == [] and instance["version"] == "v2"


def test_smoke_subset_is_reused_by_the_full_run(tmp_path):
    spec = g1.SPECS["v2"]
    items = [g1.make_item("budget_ledger", level, family, i, spec)
             for level in (1, 2, 13) for family in ("SUM", "MAX") for i in range(2)]
    models = [{"name": "a", "provider_model": "pa"}, {"name": "b", "provider_model": "pb"}]
    plan = g1.planned_calls(items, models, order_seed=5)
    smoke = g1.smoke_subset(plan, per_cell=1)
    assert len(smoke) == 2 * 2  # 2 models x 2 primary levels; contingency excluded
    assert set(map(lambda x: (x[0]["name"], x[1]["item_id"]), smoke)) <= set(
        map(lambda x: (x[0]["name"], x[1]["item_id"]), plan))
    client = ThreadSafeClient()
    clients = {"a": client, "b": client}
    checkpoint = tmp_path / "r.jsonl"
    rows = g1.acquire(smoke, clients, checkpoint, "fp", 2048, workers=4)
    summary = g1.operational_summary(rows)
    assert "correct" not in json_dump(summary)
    g1.acquire(plan, clients, checkpoint, "fp", 2048, workers=4)
    assert client.calls == len(plan)  # smoke calls were not repeated


def json_dump(value):
    import json

    return json.dumps(value)


def test_milestone_chains_are_shallow_and_state_finish_days():
    from reasoning_monitor.data.cdr_stream_worlds import get_world

    world = get_world("milestone_chains", "v2")
    import random as _random

    state = world.init(_random.Random(3))
    assert all(len(p) <= 1 for p in state["prereqs"].values())
    depth = {}
    for task in world.TASKS:
        prereq = state["prereqs"][task]
        depth[task] = depth[prereq[0]] + 1 if prereq else 0
    assert max(depth.values()) == 2
    assert "finishes on day" in world.summary(state)
    items = [g1.make_item("milestone_chains", level, family, 0, g1.SPECS["v2ext"])
             for level in (1, 12, 14) for family in cs.FAMILIES]
    assert [p for p in g1.validate_items(items) if p != "unequal cell sizes"] == []


def test_v2_items_unchanged_and_shared_with_extension():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "configs/selections/cdr_stream_v1_g1v2_items.json")
                        .read_text(encoding="utf-8"))
    assert g1.items_sha256(g1.generate_items(g1.SPECS["v2"])) == frozen["items_sha256"]
    shared = {i["item_id"]: i for i in frozen["items"] if i["template"] == "lab_protocol"
              and i["level"] == 3}
    ext = {i["item_id"]: g1.make_item("lab_protocol", 3, i["family"],
                                      int(i["item_id"].split("/")[-1]), g1.SPECS["v2ext"])
           for i in shared.values()}
    assert ext == shared


def test_planned_calls_respect_model_template_filter():
    items = [g1.make_item(t, 1, "SUM", 0, g1.SPECS["v2ext"]) for t in g1.FINAL_TEMPLATES]
    models = [{"name": "c", "provider_model": "pc", "templates": list(g1.FINAL_TEMPLATES)},
              {"name": "g", "provider_model": "pg", "templates": ["milestone_chains"]}]
    plan = g1.planned_calls(items, models, order_seed=1)
    assert sum(m["name"] == "c" for m, _ in plan) == 4
    assert [i["template"] for m, i in plan if m["name"] == "g"] == ["milestone_chains"]


def test_combine_mappings_uses_latest_run_per_cell():
    v2 = {"g1_id": "v2", "cells": {
        "g|budget_ledger": {"status": "mapped", "mapping": {"E": 1, "M": 4, "H": 6}},
        "g|inventory_logistics": {"status": "excluded", "mapping": None},
        "g|lab_protocol": {"status": "mapped", "mapping": {"E": 3, "M": 4, "H": 5}},
        "g|schedule_constraints": {"status": "excluded", "mapping": None}},
        "models": {"g": {"operational_pass": True}}}
    ext = {"g1_id": "ext", "cells": {
        "g|milestone_chains": {"status": "excluded", "mapping": None},
        "g|budget_ledger": {"status": "not_run"}},
        "models": {"g": {"operational_pass": True}}}
    combined = g1.combine_mappings([v2, ext])
    g = combined["models"]["g"]
    assert g["retained"] and g["templates_excluded"] == 2
    assert set(g["mappings"]) == {"budget_ledger", "lab_protocol"}
    ext["cells"]["g|milestone_chains"]["status"] = "needs_G1b"
    assert not g1.combine_mappings([v2, ext])["models"]["g"]["retained"]


def test_declared_exclusion_counts_as_excluded_not_pending():
    run = {"g1_id": "frontier", "cells": {
        "c|budget_ledger": {"status": "mapped", "mapping": {"E": 3, "M": 6, "H": 9}},
        "c|inventory_logistics": {"status": "mapped", "mapping": {"E": 2, "M": 5, "H": 8}},
        "c|lab_protocol": {"status": "not_run"},
        "c|milestone_chains": {"status": "excluded", "mapping": None}},
        "models": {"c": {"operational_pass": True}}}
    assert not g1.combine_mappings([run])["models"]["c"]["retained"]
    combined = g1.combine_mappings(
        [run], declared_exclusions={"c": {"lab_protocol": "gateway classifier"}})
    model = combined["models"]["c"]
    assert model["retained"] and model["templates_excluded"] == 2
    assert model["statuses"]["lab_protocol"] == "excluded"


def test_assess_g1_drops_invalid_model():
    records = []
    for template in cs.TEMPLATES:
        for row in _rows({1: .97, 2: .87, 3: .8, 4: .7, 5: .6, 6: .5, 7: .4, 8: .3}):
            records.append({**row, "model": "good", "template": template,
                            "truncated": False})
            records.append({**row, "model": "bad", "template": template,
                            "contract_valid": False, "truncated": False})
    report = g1.assess_g1(records, ["good", "bad"])
    assert report["models"]["good"]["retained"]
    assert not report["models"]["bad"]["retained"]
