import random
from collections import Counter

import pytest

from reasoning_monitor.data import cdr_stream as cs
from reasoning_monitor.data.cdr_stream_worlds import WORLDS


@pytest.fixture(scope="module")
def bank():
    return cs.generate_bank()


def test_bank_is_deterministic_and_sized(bank):
    assert cs.bank_sha256(bank) == cs.bank_sha256(cs.generate_bank())
    counts = Counter(d["scenario"] for d in bank["dialogues"])
    assert counts["S1"] == 4 * cs.CORE_DIALOGUES_PER_CELL
    assert counts["S6"] == 4 * cs.SECONDARY_DIALOGUES_PER_CELL
    assert all(len(d["turns"]) == cs.T_TURNS for d in bank["dialogues"])


def test_skeletons_pass_structural_checks(bank):
    problems = [p for d in bank["dialogues"] for p in cs.validate_skeleton(d)]
    assert problems == []


@pytest.mark.parametrize("scenario,expected", [
    ("S1", {1: "E", 33: "E", 34: "M", 67: "M", 68: "H", 100: "H"}),
    ("S1-reverse", {1: "H", 34: "M", 68: "E"}),
    ("S2", {50: "E", 51: "H"}),
    ("S3", {33: "E", 34: "H", 66: "H", 67: "E"}),
])
def test_regime_schedules_match_design(scenario, expected):
    regimes = cs.regime_schedule(scenario)
    for t, regime in expected.items():
        assert regimes[t - 1] == regime


def test_ramp_moves_monotonically_from_easy_to_hard():
    regimes = cs.regime_schedule("S1-ramp")
    order = {"E": 0, "M": 1, "H": 2}
    first_half = [order[r] for r in regimes[:20]]
    last_half = [order[r] for r in regimes[-20:]]
    assert sum(first_half) < sum(last_half)
    assert regimes[0] == "E" and regimes[-1] == "H"


def test_slots_follow_the_mixture_proportions():
    slots = cs.mixture_slots(["E"] * 100, random.Random(1))
    counts = Counter("X" if s.startswith("X") else s for s in slots)
    assert counts == Counter({"C": 50, "-1": 20, "+1": 20, "X": 10})
    assert all(s == "X:M" for s in slots if s.startswith("X"))


def test_level_mapping_and_clipping():
    mapping = {"E": 1, "M": 4, "H": 8}
    assert cs.level_for({"regime": "E", "slot": "-1"}, mapping) == 1
    assert cs.level_for({"regime": "H", "slot": "+1"}, mapping) == 8
    assert cs.level_for({"regime": "E", "slot": "X:M"}, mapping) == 4
    with pytest.raises(ValueError):
        cs.validate_mapping({"E": 3, "M": 3, "H": 5})


def test_schedule_world_longest_path():
    world = WORLDS["schedule_constraints"]
    state = {"duration": {t: 1 for t in world.TASKS},
             "prereqs": {t: [] for t in world.TASKS}}
    state["duration"]["Survey"] = 3
    state["prereqs"]["Design"] = ["Survey"]
    state["prereqs"]["Prototype"] = ["Survey", "Design"]
    finish = world.values(state)
    assert finish["Survey"] == 3 and finish["Design"] == 4 and finish["Prototype"] == 5
    extended = world.apply(state, {"kind": "conditional", "task": "Survey", "n": 2,
                                   "threshold": 4, "other": "Design", "k": 3,
                                   "numbers": [2, 4, 3]})
    assert world.values(extended)["Design"] == 5 + 4  # Survey 5, Design 1+3


def test_budget_conditional_top_up():
    world = WORLDS["budget_ledger"]
    state = {"lines": {line: {"allocated": 1000, "spent": 0} for line in world.LINES}}
    event = {"kind": "conditional", "line": "Travel", "n": 900, "threshold": 200,
             "top_up": 300, "numbers": [900, 200, 300]}
    assert world.values(world.apply(state, event))["Travel"] == 400


def test_query_evaluation_families():
    world = WORLDS["inventory_logistics"]
    state = {"stock": {k: 10 for k in world.init(random.Random(0))["stock"]}}
    keys = list(state["stock"])
    state["stock"][keys[0]] = 30
    query = {"family": "SHARE", "A": [keys[0]], "B": [keys[1]], "x": None, "hyp": None,
             "filter": False}
    assert cs.evaluate_query(world, state, query) == 75
    query = {"family": "SUM", "A": keys[:3], "B": [], "x": 15, "hyp": None, "filter": True}
    assert cs.evaluate_query(world, state, query) == 30


@pytest.mark.parametrize("template", cs.TEMPLATES)
def test_instances_pass_g0_for_extreme_mappings(bank, template):
    skeleton = next(d for d in bank["dialogues"]
                    if d["template"] == template and d["scenario"] == "S2")
    for mapping in ({"E": 1, "M": 2, "H": 3}, {"E": 6, "M": 7, "H": 8}):
        instance = cs.instantiate(skeleton, mapping)
        problems, stats = cs.validate_instance(instance)
        assert problems == []
        assert stats["paraphrase_similarity_mean"] < 0.8
        assert instance["turns"][0]["text"] != instance["turns"][0]["paraphrases"][0]


def test_noise_keeps_numbers_and_labels():
    rng = random.Random(3)
    text = "Transfer 15 valves from Izmir to Konya, then report the stock carefully."
    noisy = cs.add_noise(text, 3, rng, {"valves", "izmir", "konya"})
    assert "15" in noisy and "valves" in noisy and "izmir" in noisy
    assert "," not in noisy


def test_budget_estimate_counts_calls(bank):
    skeleton = cs.package_a_dialogues(bank)[0]
    estimate = cs.estimate_dialogue_tokens(cs.instantiate(skeleton, cs.REFERENCE_MAPPING),
                                           3, 2, 1)
    assert estimate["calls"] == cs.T_TURNS * 6
    assert len(cs.package_a_dialogues(bank)) == 5 * 4 * 2
