"""CDR-Stream: deterministic long dialogue streams with per-turn exact gold.

Design: CDR_STREAM_DESIGN.md. The bank stores model-agnostic *skeletons*: the
regime (E/M/H) of each turn, its mixture slot, query family, surface variant and
surface levels. A concrete dialogue is produced by ``instantiate`` for a given
level mapping {E: e, M: m, H: h}. The mapping is chosen per (model, template)
after the single-turn calibration gate G1, so the same skeleton gives matched
expected accuracy across models. Nothing here calls a model.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasoning_monitor.data.cdr_stream_worlds import N_VARIANTS, WORLDS, get_world, join_labels

PROTOCOL_ID = "CDR-Stream-v1"
GENERATOR_VERSION = "cdr-stream-draft-1"
BANK_STATUS = "draft-not-frozen"
T_TURNS = 100
LEVELS = tuple(range(1, 9))
FAMILIES = ("SUM", "DIFF", "COUNT_ABOVE", "MAX", "WHATIF", "SHARE")
TEMPLATES = tuple(WORLDS)
REGIMES = ("E", "M", "H")
REFERENCE_MAPPING = {"E": 2, "M": 4, "H": 6}  # review / budget only, not a G1 result

# level -> (query terms, events, distractors, event complexity)
LEVEL_SPECS: dict[int, tuple[int, int, int, int]] = {
    1: (1, 1, 0, 0), 2: (2, 1, 0, 0), 3: (3, 1, 1, 0), 4: (3, 2, 1, 1),
    5: (4, 2, 2, 1), 6: (5, 2, 2, 1), 7: (6, 3, 3, 2), 8: (8, 3, 4, 2),
    # Contingency levels, used only through the frozen G1b ceiling rule.
    9: (9, 4, 5, 2), 10: (10, 5, 6, 2),
}
CONTINGENCY_LEVELS = (9, 10)
MAX_LEVEL = CONTINGENCY_LEVELS[-1]
# SUM queries add a threshold filter when event complexity is 2 (v1: levels >= 7).

# v2 ladder (G1 v2): one axis changes per step, minimum two query terms so that
# MAX/COUNT_ABOVE are never asked about a single entity.
LADDER_V2: dict[int, tuple[int, int, int, int]] = {
    1: (2, 1, 0, 0), 2: (2, 1, 1, 0), 3: (2, 2, 1, 0), 4: (3, 2, 1, 0),
    5: (3, 2, 1, 1), 6: (3, 2, 2, 1), 7: (4, 2, 2, 1), 8: (4, 3, 2, 1),
    9: (5, 3, 2, 1), 10: (5, 3, 2, 2), 11: (6, 3, 2, 2), 12: (6, 3, 3, 2),
    # Contingency levels, used only through the frozen G1b ceiling rule.
    13: (8, 3, 4, 2), 14: (10, 4, 5, 2),
}
LADDERS: dict[str, dict[int, tuple[int, int, int, int]]] = {"v1": LEVEL_SPECS,
                                                            "v2": LADDER_V2}
PRIMARY_LEVELS: dict[str, tuple[int, ...]] = {"v1": LEVELS, "v2": tuple(range(1, 13))}
CONTINGENCY: dict[str, tuple[int, ...]] = {"v1": CONTINGENCY_LEVELS, "v2": (13, 14)}

# Mixture slots per 10-turn block: 50% centre, 20% each neighbour, 10% cross.
SLOT_BLOCK = ("C",) * 5 + ("-1",) * 2 + ("+1",) * 2 + ("X",)
CROSS = {"E": ("M",), "H": ("M",), "M": ("E", "H")}

SCENARIOS: dict[str, dict[str, Any]] = {
    "S0-E": {"core": True}, "S0-H": {"core": True},
    "S1": {"core": True}, "S2": {"core": True}, "S3": {"core": True},
    "S0-M": {"core": False}, "S1-reverse": {"core": False},
    "S1-ramp": {"core": False}, "S4": {"core": False}, "S5": {"core": False},
    "S6": {"core": False},
}
CORE_DIALOGUES_PER_CELL = 5   # 0 calibration, 1 development, 2-4 holdout
SECONDARY_DIALOGUES_PER_CELL = 3  # 0 calibration, 1 development, 2 holdout
S6_SPIKES = (15, 30, 50)

FILLERS = (
    "Thanks for keeping track of all this.",
    "I am writing this between two meetings, so bear with me.",
    "As usual, the team wants the numbers to be exact.",
    "Nothing else unusual happened today.",
    "Please keep using the same conventions as before.",
    "We double-checked the paperwork this morning.",
    "The weather made everyone a little slow today.",
    "I appreciate the careful bookkeeping so far.",
    "Let me know if anything looks inconsistent.",
    "This update comes from the operations team.",
    "We will review everything again at the end of the week.",
    "Some colleagues are on leave, so updates may be brief.",
)


def _derive_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


# ── skeleton schedules ──────────────────────────────────────────────────────

def regime_schedule(scenario: str, turns: int = T_TURNS) -> list[str]:
    def steps(*spec):
        out = []
        for regime, start, end in spec:
            out += [regime] * (end - start + 1)
        return out

    if scenario in {"S0-E", "S4", "S5"}:
        return ["E"] * turns
    if scenario == "S0-M":
        return ["M"] * turns
    if scenario == "S0-H":
        return ["H"] * turns
    if scenario in {"S1", "S6"}:
        return steps(("E", 1, 33), ("M", 34, 67), ("H", 68, turns))
    if scenario == "S1-reverse":
        return steps(("H", 1, 33), ("M", 34, 67), ("E", 68, turns))
    if scenario == "S2":
        return steps(("E", 1, 50), ("H", 51, turns))
    if scenario == "S3":
        return steps(("E", 1, 33), ("H", 34, 66), ("E", 67, turns))
    if scenario == "S1-ramp":
        # Deterministic systematic sampling of piecewise-linear E->M->H weights.
        assigned = Counter()
        cumulative = Counter()
        out = []
        half = turns // 2
        for t in range(1, turns + 1):
            if t <= half:
                w = {"E": 1 - (t - 1) / (half - 1), "M": (t - 1) / (half - 1), "H": 0.0}
            else:
                u = (t - half - 1) / (turns - half - 1)
                w = {"E": 0.0, "M": 1 - u, "H": u}
            for r in REGIMES:
                cumulative[r] += w[r]
            choice = max(REGIMES, key=lambda r: (cumulative[r] - assigned[r], -REGIMES.index(r)))
            assigned[choice] += 1
            out.append(choice)
        return out
    raise ValueError(f"unknown scenario {scenario!r}")


def surface_schedule(scenario: str, turns: int = T_TURNS) -> tuple[list[int], list[int]]:
    ramp = [0 if t <= 33 else min(3, round(3 * (t - 33) / 34)) for t in range(1, turns + 1)]
    flat = [0] * turns
    if scenario == "S4":
        return ramp, flat
    if scenario == "S5":
        return flat, ramp
    return flat, flat


def mixture_slots(regimes: Sequence[str], rng: random.Random) -> list[str]:
    """Stratified slots per run of equal regime; cross slots name their regime."""
    slots: list[str] = []
    cross_toggle = itertools.cycle((0, 1))
    for _, run in itertools.groupby(regimes):
        run = list(run)
        regime = run[0]
        pool: list[str] = []
        while len(pool) < len(run):
            block = list(SLOT_BLOCK)
            rng.shuffle(block)
            pool += block
        for slot in pool[:len(run)]:
            if slot == "X":
                options = CROSS[regime]
                slot = "X:" + options[next(cross_toggle) % len(options)]
            slots.append(slot)
    return slots


def family_schedule(turns: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    while len(out) < turns:
        block = list(FAMILIES)
        rng.shuffle(block)
        if out and block[0] == out[-1]:
            block[0], block[-1] = block[-1], block[0]
        out += block
    return out[:turns]


def split_for(index: int, core: bool) -> str:
    if index == 0:
        return "calibration"
    if index == 1:
        return "development"
    return "holdout"


def make_skeleton(template: str, scenario: str, index: int, base_seed: int) -> dict[str, Any]:
    core = SCENARIOS[scenario]["core"]
    dialogue_id = f"{scenario}/{template}/{index:02d}"
    seed = _derive_seed(base_seed, dialogue_id)
    rng = random.Random(seed)
    regimes = regime_schedule(scenario)
    slots = mixture_slots(regimes, rng)
    if scenario == "S6":
        for spike in S6_SPIKES:
            regimes[spike - 1] = "H"
            slots[spike - 1] = "C"
    families = family_schedule(len(regimes), rng)
    verbosity, noise = surface_schedule(scenario)
    turns = [{"t": t, "regime": regimes[t - 1], "slot": slots[t - 1],
              "family": families[t - 1], "variant": rng.randrange(N_VARIANTS),
              "verbosity": verbosity[t - 1], "noise": noise[t - 1]}
             for t in range(1, len(regimes) + 1)]
    return {"dialogue_id": dialogue_id, "template": template, "scenario": scenario,
            "core": core, "split": split_for(index, core), "index": index,
            "seed": seed, "turns": turns}


# Final template set after G1 (schedule_constraints replaced by milestone_chains).
TEMPLATES_V2 = ("budget_ledger", "inventory_logistics", "lab_protocol", "milestone_chains")


def generate_bank(base_seed: int = 20261003, version: str = "v1") -> dict[str, Any]:
    """Skeleton bank. ``v1`` is the reviewed draft; ``v2`` uses the final
    templates and the G1 v2 ladder. Skeletons of templates present in both
    versions are identical (same seed and dialogue IDs)."""
    templates = TEMPLATES if version == "v1" else TEMPLATES_V2
    dialogues = []
    for scenario, spec in SCENARIOS.items():
        per_cell = CORE_DIALOGUES_PER_CELL if spec["core"] else SECONDARY_DIALOGUES_PER_CELL
        for template in templates:
            for index in range(per_cell):
                dialogues.append(make_skeleton(template, scenario, index, base_seed))
    levels = PRIMARY_LEVELS[version]
    bank = {"protocol_id": PROTOCOL_ID, "generator_version": GENERATOR_VERSION,
            "status": BANK_STATUS, "base_seed": base_seed, "turns_per_dialogue": T_TURNS,
            "levels": list(levels),
            "level_specs": {str(k): LADDERS[version][k] for k in levels},
            "families": list(FAMILIES), "templates": list(templates),
            "scenarios": SCENARIOS, "dialogues": dialogues}
    if version != "v1":  # v1 keys stay exactly as frozen in the draft bank
        bank["ladder_version"] = version
        bank["status"] = "frozen-for-pilot"
    return bank


def bank_sha256(bank: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(bank, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def write_bank(path: str | Path, bank: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(bank, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ── instantiation ───────────────────────────────────────────────────────────

def level_for(turn: dict[str, Any], mapping: dict[str, int], version: str = "v1") -> int:
    slot = turn["slot"]
    if slot.startswith("X:"):
        return mapping[slot[2:]]
    centre = mapping[turn["regime"]]
    if slot == "-1":
        return max(LEVELS[0], centre - 1)
    if slot == "+1":
        return min(max(PRIMARY_LEVELS[version][-1], mapping["H"]), centre + 1)
    return centre


def validate_mapping(mapping: dict[str, int], version: str = "v1") -> None:
    top = max(LADDERS[version])
    if not (LEVELS[0] <= mapping["E"] < mapping["M"] < mapping["H"] <= top):
        raise ValueError(f"mapping must satisfy 1 <= E < M < H <= {top}: {mapping}")


ENTITY_FIELDS = ("line", "source", "target", "key", "c", "task", "other", "prereq")


def event_entities(event: dict[str, Any]) -> list[str]:
    return [event[f] for f in ENTITY_FIELDS if f in event]


def _threshold(values: list[int], rng: random.Random) -> int:
    ordered = sorted(set(values))
    if len(ordered) >= 2:
        k = rng.randrange(1, len(ordered))
        low, high = ordered[k - 1], ordered[k]
        if high - low >= 2:
            return rng.randrange(low + 1, high)
    base = ordered[0]
    return max(0, base + rng.choice([-1, 1]) * rng.randrange(3, 40))


def make_query(world, state, family, spec, changed, rng) -> dict[str, Any]:
    """``spec`` is a ladder entry (query terms, events, distractors, complexity)."""
    n_terms = spec[0]
    entities = world.entities(state)
    values = world.values(state)

    def pick(count, must=()):
        chosen = [e for e in must if e in entities][:count]
        preferred = [e for e in changed if e not in chosen]
        rng.shuffle(preferred)
        if preferred and len(chosen) < count:
            chosen.append(preferred[0])
        rest = [e for e in entities if e not in chosen]
        rng.shuffle(rest)
        chosen += rest[:count - len(chosen)]
        return chosen

    query: dict[str, Any] = {"family": family, "A": [], "B": [], "x": None, "hyp": None,
                             "filter": False}
    if family in {"DIFF", "SHARE"}:
        total = max(2, n_terms)
        both = pick(total)
        size_a = math.ceil(total / 2)
        a, b = both[:size_a], both[size_a:]
        if family == "DIFF" and sum(values[e] for e in a) < sum(values[e] for e in b):
            a, b = b, a
        if family == "SHARE" and sum(values[e] for e in a + b) == 0:
            raise RuntimeError("zero share denominator")
        query.update(A=a, B=b)
    elif family == "WHATIF":
        hyp = None
        for _ in range(30):
            hyp = world.make_event(state, world.hypothetical_kind, rng)
            if hyp is not None:
                break
        if hyp is None:
            raise RuntimeError("no hypothetical event possible")
        query.update(A=pick(n_terms, must=event_entities(hyp)), hyp=hyp)
    else:
        query["A"] = pick(n_terms)
        if family == "COUNT_ABOVE" or (family == "SUM" and spec[3] >= 2):
            query["x"] = _threshold([values[e] for e in query["A"]], rng)
            query["filter"] = family == "SUM"
    numbers = ([query["x"]] if query["x"] is not None else []) + (
        list(query["hyp"]["numbers"]) if query["hyp"] else [])
    query["numbers"] = numbers
    return query


def evaluate_query(world, state, query) -> int:
    values = world.values(state)
    family = query["family"]
    a = query["A"]
    if family == "SUM":
        return sum(values[e] for e in a if not query["filter"] or values[e] > query["x"])
    if family == "DIFF":
        return sum(values[e] for e in a) - sum(values[e] for e in query["B"])
    if family == "COUNT_ABOVE":
        return sum(values[e] > query["x"] for e in a)
    if family == "MAX":
        return max(values[e] for e in a)
    if family == "WHATIF":
        hyp_values = world.values(world.apply(state, query["hyp"]))
        return sum(hyp_values[e] for e in a)
    if family == "SHARE":
        num = sum(values[e] for e in a)
        return (100 * num) // (num + sum(values[e] for e in query["B"]))
    raise ValueError(family)


def realize_query(world, query, variant) -> str:
    vn = world.value_noun
    la = join_labels([world.label(e) for e in query["A"]])
    lb = join_labels([world.label(e) for e in query["B"]]) if query["B"] else ""
    x = query["x"]
    family = query["family"]
    if family == "SUM" and query["filter"]:
        return (f"Considering only those of {la} whose {vn} is greater than {x}, what is "
                f"their total {vn}?",
                f"Among {la}, add up the {vn} of the ones above {x}. What is the result?",
                f"What is the combined {vn} of the items among {la} that exceed {x}?",
                f"Take {la}, keep those with a {vn} over {x}, and give their total.")[variant]
    if family == "SUM":
        return (f"What is the total {vn} of {la}?",
                f"Add up the {vn} of {la}. What do you get?",
                f"Taken together, what is the combined {vn} of {la}?",
                f"Give the sum of the {vn} values for {la}.")[variant]
    if family == "DIFF":
        return (f"By how much does the total {vn} of {la} exceed the total {vn} of {lb}?",
                f"What is the combined {vn} of {la} minus the combined {vn} of {lb}?",
                f"How much larger is the total {vn} of {la} than that of {lb}?",
                f"Subtract the total {vn} of {lb} from the total {vn} of {la}. What is "
                "left?")[variant]
    if family == "COUNT_ABOVE":
        return (f"How many of {la} have a {vn} greater than {x}?",
                f"Among {la}, count those whose {vn} is above {x}.",
                f"For how many of {la} does the {vn} exceed {x}?",
                f"Of {la}, how many are over {x} in {vn}?")[variant]
    if family == "MAX":
        return (f"What is the largest {vn} among {la}?",
                f"Among {la}, what is the highest {vn}?",
                f"Which value is the maximum {vn} across {la}? Give the number.",
                f"Report the greatest {vn} found among {la}.")[variant]
    if family == "WHATIF":
        clause = world.hypothetical_clause(query["hyp"], variant)
        return (f"Suppose, hypothetically and without recording it, that {clause}. What "
                f"would the total {vn} of {la} then be?",
                f"Without changing the records: if {clause}, what would {la} total in "
                f"{vn}?",
                f"Imagine (do not record it) that {clause}. Give the resulting combined "
                f"{vn} of {la}.",
                f"Hypothetically, if {clause}, what would be the sum of the {vn} of {la}? "
                "This is not a real change.")[variant]
    if family == "SHARE":
        group = join_labels([world.label(e) for e in query["A"] + query["B"]])
        return (f"Consider the group {group}. What percentage of the group's combined {vn} "
                f"comes from {la}? Round down to a whole number.",
                f"Within the group made of {group}, what share of the total {vn} belongs "
                f"to {la}, as a whole-number percentage rounded down?",
                f"Treat the combined {vn} of {group} as the whole. What percentage of it "
                f"is contributed by {la}? Round down.",
                f"Looking only at {group}: what percent (rounded down) of their joint {vn} "
                f"is accounted for by {la}?")[variant]
    raise ValueError(family)


def add_verbosity(text: str, level: int, rng: random.Random) -> str:
    if level <= 0:
        return text
    return " ".join(rng.sample(FILLERS, k=level)) + " " + text


def add_noise(text: str, level: int, rng: random.Random, protected: set[str]) -> str:
    if level <= 0:
        return text
    pieces = re.split(r"(\s+)", text)
    out = []
    for piece in pieces:
        core = piece.strip(".,;:?!()")
        if (len(core) >= 4 and core.isalpha() and core.lower() not in protected
                and rng.random() < 0.05 * level):
            i = rng.randrange(1, len(core) - 2)
            swapped = core[:i] + core[i + 1] + core[i] + core[i + 2:]
            piece = piece.replace(core, swapped, 1)
        out.append(piece)
    noisy = "".join(out)
    if level >= 2:
        noisy = noisy.lower()
    if level >= 3:
        noisy = noisy.replace(",", "")
    return noisy


def _generate_events(world, state, n_events, complexity, rng):
    kinds: list[str] = []
    if complexity >= 1:
        kinds.append(rng.choice(world.complex_kinds(complexity)))
        if complexity >= 2 and n_events >= 2:
            kinds.append("correction")
    while len(kinds) < n_events:
        kinds.append(rng.choice(world.simple_kinds()))
    rng.shuffle(kinds)
    events = []
    for kind in kinds:
        event = None
        candidates = [kind] + [k for k in world.simple_kinds() if k != kind]
        for attempt in range(40):
            event = world.make_event(state, candidates[min(attempt // 10, len(candidates) - 1)],
                                     rng)
            if event is not None:
                break
        if event is None:
            continue
        events.append(event)
        state = world.apply(state, event)
    return events, state


def instantiate(skeleton: dict[str, Any], mapping: dict[str, int],
                version: str = "v1") -> dict[str, Any]:
    validate_mapping(mapping, version)
    world = get_world(skeleton["template"], version)
    rng = random.Random(_derive_seed(skeleton["seed"], mapping["E"], mapping["M"],
                                     mapping["H"]))
    state = world.init(rng)
    entities = world.entities(state)
    last_mention = {e: 1 for e in entities}
    protected = {w.lower() for e in entities for w in world.label_words(e)}
    turns = []
    for spec in skeleton["turns"]:
        t = spec["t"]
        level = level_for(spec, mapping, version)
        ladder_spec = LADDERS[version][level]
        n_terms, n_events, n_distractors, complexity = ladder_spec
        state_before = state
        summary_before = world.summary(state_before)
        events, state = _generate_events(world, state, n_events, complexity, rng)
        changed = sorted({e for ev in events for e in event_entities(ev)})
        for entity in changed:
            last_mention[entity] = t
        distractors = []
        for _ in range(n_distractors):
            d = world.make_distractor(state, rng)
            d["numbers"] = [d["n"]]
            distractors.append(d)
        query = make_query(world, state, spec["family"], ladder_spec, changed, rng)
        gold = evaluate_query(world, state, query)
        # Distractors go to fixed positions after the events (events keep order).
        slots = sorted(rng.sample(range(len(events) + 1), k=min(len(distractors),
                                                                  len(events) + 1)))
        opening_text, opening_numbers = world.opening(state_before) if t == 1 else ("", [])
        texts = []
        for k in range(3):  # 0 = primary, 1..2 = paraphrases
            variant = (spec["variant"] + k) % N_VARIANTS
            sentences = [world.realize_event(ev, (variant + i) % N_VARIANTS)
                         for i, ev in enumerate(events)]
            for j, (position, d) in enumerate(zip(slots, distractors, strict=False)):
                sentences.insert(position + j, world.realize_distractor(
                    d, (variant + j) % N_VARIANTS))
            for d in distractors[len(slots):]:
                sentences.append(world.realize_distractor(d, variant))
            body = " ".join(sentences + [realize_query(world, query, variant)])
            if opening_text:
                body = opening_text + " " + body
            surface_rng = random.Random(_derive_seed(skeleton["seed"], t, k, "surface"))
            body = add_verbosity(body, spec["verbosity"], surface_rng)
            body = add_noise(body, spec["noise"], surface_rng, protected)
            texts.append(body)
        query_entities = query["A"] + query["B"]
        turns.append({
            "t": t, "regime": spec["regime"], "slot": spec["slot"], "level": level,
            "family": spec["family"], "verbosity": spec["verbosity"], "noise": spec["noise"],
            "axes": {"query_terms": len(query_entities), "events": len(events),
                     "events_planned": n_events, "distractors": len(distractors),
                     "complexity": complexity, "filter": query["filter"]},
            "dependency_distance": max(t - last_mention[e] for e in query_entities),
            "events": events, "distractors": distractors, "query": query, "gold": gold,
            "expected_numbers": opening_numbers + [n for ev in events for n in ev["numbers"]]
            + [n for d in distractors for n in d["numbers"]] + query["numbers"],
            "expected_labels": sorted({w.lower() for e in changed + query_entities
                                       for w in world.label_words(e)}),
            "text": texts[0], "paraphrases": texts[1:],
            "state_summary_before": summary_before,
            "values_after": world.values(state),
        })
    return {"dialogue_id": skeleton["dialogue_id"], "template": skeleton["template"],
            "scenario": skeleton["scenario"], "split": skeleton["split"],
            "mapping": dict(mapping), "version": version, "turns": turns}


# ── G0 validation ───────────────────────────────────────────────────────────

def _ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def validate_skeleton(skeleton: dict[str, Any]) -> list[str]:
    problems = []
    turns = skeleton["turns"]
    did = skeleton["dialogue_id"]
    if len(turns) != T_TURNS:
        problems.append(f"{did}: {len(turns)} turns")
    families = [t["family"] for t in turns]
    for i in range(1, len(families)):
        if families[i] == families[i - 1]:
            problems.append(f"{did}: family repeats at t={i + 1}")
    for start in range(0, len(families) - 19):
        counts = Counter(families[start:start + 20])
        if max(counts.values()) > 5:
            problems.append(f"{did}: family share > 25% in window starting t={start + 1}")
    expected = regime_schedule(skeleton["scenario"])
    if skeleton["scenario"] == "S6":
        for spike in S6_SPIKES:
            expected[spike - 1] = "H"
    if [t["regime"] for t in turns] != expected:
        problems.append(f"{did}: regime schedule mismatch")
    for regime, run in itertools.groupby(turns, key=lambda t: t["regime"]):
        run = list(run)
        if len(run) >= 30 and set(FAMILIES) - {t["family"] for t in run}:
            problems.append(f"{did}: family missing in a {regime} run")
        if len(run) >= 10 and skeleton["scenario"] != "S6":
            block = Counter(t["slot"][:1] if t["slot"].startswith("X") else t["slot"]
                            for t in run[:10])
            if block != Counter({"C": 5, "-1": 2, "+1": 2, "X": 1}):
                problems.append(f"{did}: slot stratification off in first block of a run")
    return problems


def validate_instance(instance: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems = []
    did = instance["dialogue_id"]
    gold_in_text = 0
    short_events = 0
    golds = Counter()
    similarity = []
    for turn in instance["turns"]:
        t = turn["t"]
        if not isinstance(turn["gold"], int):
            problems.append(f"{did} t={t}: non-integer gold")
        if any(v < 0 for v in turn["values_after"].values()):
            problems.append(f"{did} t={t}: negative value")
        if turn["query"]["family"] == "DIFF" and turn["gold"] < 0:
            problems.append(f"{did} t={t}: negative DIFF")
        expected = Counter(turn["expected_numbers"])
        for k, text in enumerate([turn["text"]] + turn["paraphrases"]):
            if Counter(_ints(text)) != expected:
                problems.append(f"{did} t={t} realization {k}: numbers differ from spec")
            lowered = text.lower()
            missing = [w for w in turn["expected_labels"] if w not in lowered]
            if missing:
                problems.append(f"{did} t={t} realization {k}: labels missing {missing}")
        for p in turn["paraphrases"]:
            ratio = difflib.SequenceMatcher(None, turn["text"], p).ratio()
            similarity.append(ratio)
            if ratio >= 0.95:
                problems.append(f"{did} t={t}: paraphrase too similar ({ratio:.2f})")
        spec = LADDERS[instance.get("version", "v1")][turn["level"]]
        if turn["axes"]["distractors"] != spec[2]:
            problems.append(f"{did} t={t}: distractor count mismatch")
        if turn["axes"]["events"] < turn["axes"]["events_planned"]:
            short_events += 1
        gold_in_text += turn["gold"] in _ints(turn["text"])
        golds[turn["gold"]] += 1
    n = len(instance["turns"])
    stats = {"gold_in_current_text_rate": gold_in_text / n,
             "max_single_gold_share": max(golds.values()) / n,
             "turns_with_fewer_events_than_planned": short_events,
             "paraphrase_similarity_mean": sum(similarity) / len(similarity),
             "paraphrase_similarity_max": max(similarity)}
    if short_events > 0.05 * n:
        problems.append(f"{did}: {short_events} turns had fewer events than planned")
    return problems, stats


def all_mappings() -> list[dict[str, int]]:
    return [{"E": e, "M": m, "H": h} for e, m, h in itertools.combinations(LEVELS, 3)]


# ── budget estimate (character/4 token proxy, declared) ─────────────────────

SYSTEM_CHARS = 320
CONTRACT_CHARS = 360
ASSUMED_REPLY_CHARS = 600   # about 150 tokens: eight short reasoning lines + marker
CHARS_PER_TOKEN = 4.0


def estimate_dialogue_tokens(instance: dict[str, Any], k: int, l_paraphrases: int,
                             control_calls: int) -> dict[str, float]:
    history = 0
    dialogue_in = control_in = 0.0
    calls = 0
    for turn in instance["turns"]:
        user = len(turn["text"])
        paraphrase_mean = (sum(len(p) for p in turn["paraphrases"][:l_paraphrases])
                           / max(1, l_paraphrases))
        base = SYSTEM_CHARS + CONTRACT_CHARS + history
        dialogue_in += k * (base + user) + l_paraphrases * (base + paraphrase_mean)
        control_in += control_calls * (SYSTEM_CHARS + CONTRACT_CHARS
                                       + len(turn["state_summary_before"]) + user)
        calls += k + l_paraphrases + control_calls
        history += user + ASSUMED_REPLY_CHARS
    output_tokens = calls * ASSUMED_REPLY_CHARS / CHARS_PER_TOKEN
    return {"calls": calls, "input_tokens": (dialogue_in + control_in) / CHARS_PER_TOKEN,
            "output_tokens_assumed": output_tokens,
            "final_turn_context_tokens": (SYSTEM_CHARS + CONTRACT_CHARS + history)
            / CHARS_PER_TOKEN}


def package_a_dialogues(bank: dict[str, Any], per_cell: int = 2) -> list[dict[str, Any]]:
    """Package A: core scenarios, holdout indices 2..(1+per_cell), four templates."""
    return [d for d in bank["dialogues"]
            if d["core"] and d["split"] == "holdout" and d["index"] < 2 + per_cell]


def copy_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(skeleton)
