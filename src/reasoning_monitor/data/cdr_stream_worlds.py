"""Template worlds for CDR-Stream: state, events, exact solvers and realizers.

Each world keeps an integer-valued state. ``values(state)`` gives the quantity
that queries ask about, one integer per entity. Events are plain dictionaries
(the semantic specification); ``apply`` is the exact solver step. Realizers turn
a specification into English text in one of four surface variants. Every number
a realization prints is listed in the event's ``numbers`` field, so the G0
validator can check that no variant drops or alters a number.

See CDR_STREAM_DESIGN.md. Nothing here calls a model.
"""

from __future__ import annotations

import copy
import random
from typing import Any

N_VARIANTS = 4


def join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


class World:
    """Interface shared by all templates."""

    name = "base"
    domain = "base"
    value_noun = "value"
    entity_noun = "entities"
    hypothetical_kind = "none"

    def init(self, rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    def entities(self, state: dict[str, Any]) -> list[str]:
        raise NotImplementedError

    def label(self, entity: str) -> str:
        return entity

    def label_words(self, entity: str) -> list[str]:
        return self.label(entity).split()

    def values(self, state: dict[str, Any]) -> dict[str, int]:
        raise NotImplementedError

    def make_event(self, state: dict[str, Any], kind: str, rng: random.Random
                   ) -> dict[str, Any] | None:
        raise NotImplementedError

    def simple_kinds(self) -> tuple[str, ...]:
        raise NotImplementedError

    def complex_kinds(self, complexity: int) -> tuple[str, ...]:
        return ("correction",) if complexity == 1 else ("conditional",)

    def apply(self, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def realize_event(self, event: dict[str, Any], variant: int) -> str:
        raise NotImplementedError

    def hypothetical_clause(self, event: dict[str, Any], variant: int) -> str:
        raise NotImplementedError

    def make_distractor(self, state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    def realize_distractor(self, distractor: dict[str, Any], variant: int) -> str:
        raise NotImplementedError

    def opening(self, state: dict[str, Any]) -> tuple[str, list[int]]:
        raise NotImplementedError

    def summary(self, state: dict[str, Any]) -> str:
        raise NotImplementedError


# ── budget ledger ───────────────────────────────────────────────────────────

class BudgetLedger(World):
    name = "budget_ledger"
    domain = "arithmetic/accounting"
    value_noun = "remaining amount"
    entity_noun = "budget lines"
    hypothetical_kind = "transfer"
    LINES = ("Travel", "Equipment", "Software", "Training", "Catering", "Printing",
             "Consulting", "Maintenance", "Outreach", "Licenses")

    def init(self, rng):
        lines = {}
        for line in self.LINES:
            allocated = rng.randrange(1200, 4000)
            lines[line] = {"allocated": allocated,
                           "spent": rng.randrange(100, allocated // 2)}
        return {"lines": lines}

    def entities(self, state):
        return list(self.LINES)

    def label(self, entity):
        return f"the {entity} line"

    def label_words(self, entity):
        return [entity]

    def values(self, state):
        return {k: v["allocated"] - v["spent"] for k, v in state["lines"].items()}

    def simple_kinds(self):
        return ("spend", "refund", "transfer")

    def make_event(self, state, kind, rng):
        values = self.values(state)
        lines = state["lines"]
        line = rng.choice(self.LINES)
        if kind == "spend":
            if values[line] < 40:
                return None
            n = rng.randrange(20, min(400, values[line]))
            return {"kind": kind, "line": line, "n": n, "numbers": [n]}
        if kind == "refund":
            if lines[line]["spent"] < 40:
                return None
            n = rng.randrange(10, min(250, lines[line]["spent"]))
            return {"kind": kind, "line": line, "n": n, "numbers": [n]}
        if kind == "transfer":
            other = rng.choice([x for x in self.LINES if x != line])
            if values[line] < 60:
                return None
            n = rng.randrange(20, min(500, values[line] // 2 + 21))
            return {"kind": kind, "source": line, "target": other, "n": n, "numbers": [n]}
        if kind == "correction":
            recorded = rng.randrange(40, 400)
            actual = recorded + rng.choice([-1, 1]) * rng.randrange(5, 35)
            if actual <= 0 or lines[line]["spent"] - recorded + actual < 0 \
                    or values[line] - (actual - recorded) < 0:
                return None
            return {"kind": kind, "line": line, "recorded": recorded, "actual": actual,
                    "numbers": [recorded, actual]}
        if kind == "conditional":
            if values[line] < 120:
                return None
            n = rng.randrange(40, min(500, values[line] - 40))
            after = values[line] - n
            threshold = max(10, after + rng.choice([-1, 1]) * rng.randrange(20, 200))
            top_up = rng.randrange(50, 400)
            return {"kind": kind, "line": line, "n": n, "threshold": threshold,
                    "top_up": top_up, "numbers": [n, threshold, top_up]}
        raise ValueError(kind)

    def apply(self, state, event):
        state = copy.deepcopy(state)
        lines = state["lines"]
        kind = event["kind"]
        if kind == "spend":
            lines[event["line"]]["spent"] += event["n"]
        elif kind == "refund":
            lines[event["line"]]["spent"] -= event["n"]
        elif kind == "transfer":
            lines[event["source"]]["allocated"] -= event["n"]
            lines[event["target"]]["allocated"] += event["n"]
        elif kind == "correction":
            lines[event["line"]]["spent"] += event["actual"] - event["recorded"]
        elif kind == "conditional":
            line = lines[event["line"]]
            line["spent"] += event["n"]
            if line["allocated"] - line["spent"] < event["threshold"]:
                line["allocated"] += event["top_up"]
        else:
            raise ValueError(kind)
        return state

    def realize_event(self, e, variant):
        k = e["kind"]
        if k == "spend":
            return (f"Record an expense of {e['n']} on the {e['line']} line.",
                    f"The {e['line']} line was charged {e['n']}.",
                    f"A payment of {e['n']} went out from the {e['line']} line.",
                    f"Please log {e['n']} as spent under {e['line']}.")[variant]
        if k == "refund":
            return (f"The {e['line']} line received a refund of {e['n']}.",
                    f"Reverse {e['n']} of earlier spending on the {e['line']} line.",
                    f"A refund of {e['n']} came back to {e['line']}.",
                    f"{e['n']} previously spent under {e['line']} was refunded.")[variant]
        if k == "transfer":
            return (f"Move {e['n']} of allocation from the {e['source']} line to the "
                    f"{e['target']} line.",
                    f"Reallocate {e['n']} from {e['source']} to {e['target']}.",
                    f"The {e['target']} line gets {e['n']} of extra allocation taken from "
                    f"the {e['source']} line.",
                    f"Shift {e['n']} of budget allocation away from {e['source']} into "
                    f"{e['target']}.")[variant]
        if k == "correction":
            return (f"Correction: an earlier expense on the {e['line']} line was entered as "
                    f"{e['recorded']} but was actually {e['actual']}.",
                    f"One {e['line']} expense was mistyped: it is {e['actual']}, not "
                    f"{e['recorded']}.",
                    f"Please fix a {e['line']} entry recorded as {e['recorded']}; the true "
                    f"amount was {e['actual']}.",
                    f"Audit note: a {e['line']} charge of {e['recorded']} should have been "
                    f"{e['actual']}.")[variant]
        if k == "conditional":
            return (f"Record an expense of {e['n']} on the {e['line']} line. If its remaining "
                    f"amount is then below {e['threshold']}, a top-up of {e['top_up']} is "
                    f"added to its allocation.",
                    f"The {e['line']} line spends {e['n']}; whenever that leaves less than "
                    f"{e['threshold']} remaining, its allocation automatically grows by "
                    f"{e['top_up']}.",
                    f"Charge {e['n']} to {e['line']}. Rule: a remaining amount under "
                    f"{e['threshold']} triggers an allocation top-up of {e['top_up']}.",
                    f"Log {e['n']} spent on {e['line']}, then apply the safeguard: below "
                    f"{e['threshold']} remaining means {e['top_up']} more allocation."
                    )[variant]
        raise ValueError(k)

    def hypothetical_clause(self, e, variant):
        return (f"{e['n']} of allocation were moved from the {e['source']} line to the "
                f"{e['target']} line",
                f"{e['source']} gave {e['n']} of its allocation to {e['target']}",
                f"the {e['target']} line received {e['n']} of allocation from "
                f"{e['source']}",
                f"we shifted {e['n']} of budget from {e['source']} into "
                f"{e['target']}")[variant]

    def make_distractor(self, state, rng):
        return {"line": rng.choice(self.LINES), "n": rng.randrange(50, 900)}

    def realize_distractor(self, d, variant):
        return (f"A quote of {d['n']} for the {d['line']} line arrived but has not been "
                "approved.",
                f"Someone asked whether {d['n']} could be spent on {d['line']} next year; "
                "nothing was recorded.",
                f"The {d['line']} coordinator mentioned an estimate of {d['n']}; it is only "
                "a forecast.",
                f"For reference only: a proposal of {d['n']} for {d['line']} is still under "
                "discussion.")[variant]

    def opening(self, state):
        parts, numbers = [], []
        for line in self.LINES:
            v = state["lines"][line]
            parts.append(f"{line}: allocated {v['allocated']}, spent {v['spent']}")
            numbers += [v["allocated"], v["spent"]]
        text = ("We are tracking a project budget. The remaining amount of a line is its "
                "allocation minus its spending. Current lines: " + "; ".join(parts) + ".")
        return text, numbers

    def summary(self, state):
        parts = [f"{k}: allocated {v['allocated']}, spent {v['spent']}"
                 for k, v in state["lines"].items()]
        return ("Verified budget state (remaining = allocated - spent): "
                + "; ".join(parts) + ".")


# ── inventory logistics ─────────────────────────────────────────────────────

class InventoryLogistics(World):
    name = "inventory_logistics"
    domain = "multi-entity quantitative"
    value_noun = "stock"
    entity_noun = "stock positions"
    hypothetical_kind = "move"
    WAREHOUSES = ("Izmir", "Mersin", "Konya")
    ITEMS = ("valves", "pumps", "cables", "sensors")

    def init(self, rng):
        return {"stock": {f"{i}@{w}": rng.randrange(40, 320)
                          for w in self.WAREHOUSES for i in self.ITEMS}}

    def entities(self, state):
        return list(state["stock"])

    def label(self, entity):
        item, warehouse = entity.split("@")
        return f"{item} at {warehouse}"

    def label_words(self, entity):
        return entity.split("@")

    def values(self, state):
        return dict(state["stock"])

    def simple_kinds(self):
        return ("receive", "ship", "move")

    def make_event(self, state, kind, rng):
        stock = state["stock"]
        key = rng.choice(list(stock))
        item, warehouse = key.split("@")
        if kind == "receive":
            n = rng.randrange(5, 90)
            return {"kind": kind, "key": key, "n": n, "numbers": [n]}
        if kind == "ship":
            if stock[key] < 10:
                return None
            n = rng.randrange(5, min(90, stock[key]))
            return {"kind": kind, "key": key, "n": n, "numbers": [n]}
        if kind == "move":
            if stock[key] < 10:
                return None
            other = f"{item}@{rng.choice([w for w in self.WAREHOUSES if w != warehouse])}"
            n = rng.randrange(5, min(80, stock[key]))
            return {"kind": kind, "key": key, "target": other, "n": n, "numbers": [n]}
        if kind == "correction":
            recorded = rng.randrange(10, 90)
            actual = recorded + rng.choice([-1, 1]) * rng.randrange(2, 9)
            if actual <= 0 or stock[key] + actual - recorded < 0:
                return None
            return {"kind": kind, "key": key, "recorded": recorded, "actual": actual,
                    "numbers": [recorded, actual]}
        if kind == "conditional":
            if stock[key] < 30:
                return None
            n = rng.randrange(5, min(90, stock[key] - 10))
            after = stock[key] - n
            threshold = max(5, after + rng.choice([-1, 1]) * rng.randrange(5, 60))
            reorder = rng.randrange(20, 120)
            return {"kind": kind, "key": key, "n": n, "threshold": threshold,
                    "reorder": reorder, "numbers": [n, threshold, reorder]}
        raise ValueError(kind)

    def apply(self, state, event):
        state = copy.deepcopy(state)
        stock = state["stock"]
        kind = event["kind"]
        if kind == "receive":
            stock[event["key"]] += event["n"]
        elif kind == "ship":
            stock[event["key"]] -= event["n"]
        elif kind == "move":
            stock[event["key"]] -= event["n"]
            stock[event["target"]] += event["n"]
        elif kind == "correction":
            stock[event["key"]] += event["actual"] - event["recorded"]
        elif kind == "conditional":
            stock[event["key"]] -= event["n"]
            if stock[event["key"]] < event["threshold"]:
                stock[event["key"]] += event["reorder"]
        else:
            raise ValueError(kind)
        return state

    def realize_event(self, e, variant):
        k = e["kind"]
        item, wh = e["key"].split("@")
        if k == "receive":
            return (f"A delivery of {e['n']} {item} arrived at {wh}.",
                    f"{wh} received {e['n']} {item}.",
                    f"Add {e['n']} {item} to the {wh} stock.",
                    f"Incoming shipment: {e['n']} {item} for the {wh} warehouse.")[variant]
        if k == "ship":
            return (f"{wh} shipped {e['n']} {item} to a customer.",
                    f"Remove {e['n']} {item} from {wh}; they were sold.",
                    f"An order took {e['n']} {item} out of the {wh} warehouse.",
                    f"Dispatch from {wh}: {e['n']} {item}.")[variant]
        if k == "move":
            _, target = e["target"].split("@")
            return (f"Transfer {e['n']} {item} from {wh} to {target}.",
                    f"{e['n']} {item} were moved from the {wh} warehouse to {target}.",
                    f"{target} took over {e['n']} {item} from {wh}.",
                    f"Rebalance: {e['n']} {item} go from {wh} to {target}.")[variant]
        if k == "correction":
            return (f"Correction: an earlier delivery of {item} at {wh} was logged as "
                    f"{e['recorded']} units but was actually {e['actual']}.",
                    f"A {wh} receipt of {item} said {e['recorded']}; the real count was "
                    f"{e['actual']}.",
                    f"Fix the {wh} {item} delivery: {e['actual']} arrived, not "
                    f"{e['recorded']}.",
                    f"Audit: one {item} delivery at {wh} was {e['actual']} units, wrongly "
                    f"entered as {e['recorded']}.")[variant]
        if k == "conditional":
            return (f"{wh} shipped {e['n']} {item}. If its {item} stock is then below "
                    f"{e['threshold']}, an automatic reorder of {e['reorder']} arrives.",
                    f"Ship {e['n']} {item} from {wh}; the reorder rule adds {e['reorder']} "
                    f"whenever stock falls under {e['threshold']}.",
                    f"An order removes {e['n']} {item} at {wh}. Stock below "
                    f"{e['threshold']} triggers a delivery of {e['reorder']}.",
                    f"Dispatch {e['n']} {item} from {wh}, then: under {e['threshold']} "
                    f"left means {e['reorder']} are restocked.")[variant]
        raise ValueError(k)

    def hypothetical_clause(self, e, variant):
        item, wh = e["key"].split("@")
        _, target = e["target"].split("@")
        return (f"{e['n']} {item} were transferred from {wh} to {target}",
                f"{wh} sent {e['n']} {item} to {target}",
                f"{target} received {e['n']} {item} from {wh}",
                f"we moved {e['n']} {item} out of {wh} into {target}")[variant]

    def make_distractor(self, state, rng):
        return {"key": rng.choice(list(state["stock"])), "n": rng.randrange(10, 200)}

    def realize_distractor(self, d, variant):
        item, wh = d["key"].split("@")
        return (f"A customer asked whether {d['n']} {item} could be reserved at {wh}; no "
                "reservation was made.",
                f"The {wh} manager estimates demand of {d['n']} {item} next month.",
                f"A supplier offered {d['n']} {item} for {wh}; the offer was not accepted.",
                f"Note only: last year {wh} held about {d['n']} {item} at this time."
                )[variant]

    def opening(self, state):
        parts, numbers = [], []
        for wh in self.WAREHOUSES:
            items = [f"{state['stock'][f'{i}@{wh}']} {i}" for i in self.ITEMS]
            numbers += [state["stock"][f"{i}@{wh}"] for i in self.ITEMS]
            parts.append(f"{wh} holds " + join_labels(items))
        return ("We are tracking warehouse stock. " + "; ".join(parts) + ".", numbers)

    def summary(self, state):
        parts = [f"{self.label(k)}: {v}" for k, v in state["stock"].items()]
        return "Verified stock levels: " + "; ".join(parts) + "."


# ── lab protocol ────────────────────────────────────────────────────────────

class LabProtocol(World):
    name = "lab_protocol"
    domain = "science/volumes"
    value_noun = "volume in mL"
    entity_noun = "containers"
    hypothetical_kind = "pour"
    CONTAINERS = ("buffer flask", "ethanol bottle", "saline flask", "reagent vial",
                  "culture tube", "acid beaker", "base beaker", "stock solution",
                  "wash bottle", "sample tube")

    def init(self, rng):
        return {"ml": {c: rng.randrange(60, 480) for c in self.CONTAINERS}}

    def entities(self, state):
        return list(self.CONTAINERS)

    def label(self, entity):
        return f"the {entity}"

    def label_words(self, entity):
        return entity.split()

    def values(self, state):
        return dict(state["ml"])

    def simple_kinds(self):
        return ("add", "use", "pour")

    def complex_kinds(self, complexity):
        return ("correction",) if complexity == 1 else ("conditional", "dilute")

    def make_event(self, state, kind, rng):
        ml = state["ml"]
        c = rng.choice(self.CONTAINERS)
        if kind == "add":
            n = rng.randrange(5, 120)
            return {"kind": kind, "c": c, "n": n, "numbers": [n]}
        if kind == "use":
            if ml[c] < 10:
                return None
            n = rng.randrange(5, min(120, ml[c]))
            return {"kind": kind, "c": c, "n": n, "numbers": [n]}
        if kind == "pour":
            if ml[c] < 10:
                return None
            other = rng.choice([x for x in self.CONTAINERS if x != c])
            n = rng.randrange(5, min(100, ml[c]))
            return {"kind": kind, "c": c, "target": other, "n": n, "numbers": [n]}
        if kind == "correction":
            recorded = rng.randrange(10, 100)
            actual = recorded + rng.choice([-1, 1]) * rng.randrange(2, 9)
            if actual <= 0 or ml[c] - actual + recorded < 0:
                return None
            # A pipetting error: the volume withdrawn was `actual`, not `recorded`.
            return {"kind": kind, "c": c, "recorded": recorded, "actual": actual,
                    "numbers": [recorded, actual]}
        if kind == "conditional":
            if ml[c] < 40:
                return None
            n = rng.randrange(5, min(120, ml[c] - 20))
            after = ml[c] - n
            threshold = max(5, after + rng.choice([-1, 1]) * rng.randrange(5, 80))
            refill = rng.randrange(20, 150)
            return {"kind": kind, "c": c, "n": n, "threshold": threshold,
                    "refill": refill, "numbers": [n, threshold, refill]}
        if kind == "dilute":
            if ml[c] > 600:
                return None
            return {"kind": kind, "c": c, "numbers": []}
        raise ValueError(kind)

    def apply(self, state, event):
        state = copy.deepcopy(state)
        ml = state["ml"]
        kind = event["kind"]
        if kind == "add":
            ml[event["c"]] += event["n"]
        elif kind == "use":
            ml[event["c"]] -= event["n"]
        elif kind == "pour":
            ml[event["c"]] -= event["n"]
            ml[event["target"]] += event["n"]
        elif kind == "correction":
            ml[event["c"]] -= event["actual"] - event["recorded"]
        elif kind == "conditional":
            ml[event["c"]] -= event["n"]
            if ml[event["c"]] < event["threshold"]:
                ml[event["c"]] += event["refill"]
        elif kind == "dilute":
            ml[event["c"]] *= 2
        else:
            raise ValueError(kind)
        return state

    def realize_event(self, e, variant):
        k = e["kind"]
        c = e["c"]
        if k == "add":
            return (f"Add {e['n']} mL of solution to the {c}.",
                    f"The {c} was topped up with {e['n']} mL.",
                    f"{e['n']} mL more went into the {c}.",
                    f"Pipette {e['n']} mL into the {c}.")[variant]
        if k == "use":
            return (f"{e['n']} mL were withdrawn from the {c} for an assay.",
                    f"Use {e['n']} mL from the {c}.",
                    f"The experiment consumed {e['n']} mL of the {c}.",
                    f"Take {e['n']} mL out of the {c}.")[variant]
        if k == "pour":
            return (f"Pour {e['n']} mL from the {c} into the {e['target']}.",
                    f"{e['n']} mL were transferred from the {c} to the {e['target']}.",
                    f"The {e['target']} received {e['n']} mL taken from the {c}.",
                    f"Move {e['n']} mL out of the {c} and into the {e['target']}.")[variant]
        if k == "correction":
            return (f"Correction: an earlier withdrawal from the {c} was logged as "
                    f"{e['recorded']} mL but was actually {e['actual']} mL.",
                    f"A pipetting log for the {c} said {e['recorded']} mL were removed; "
                    f"the true amount was {e['actual']} mL.",
                    f"Fix one {c} entry: {e['actual']} mL were taken out, not "
                    f"{e['recorded']} mL.",
                    f"Audit: a withdrawal of {e['actual']} mL from the {c} was recorded as "
                    f"{e['recorded']} mL.")[variant]
        if k == "conditional":
            return (f"Withdraw {e['n']} mL from the {c}. If it then holds less than "
                    f"{e['threshold']} mL, refill it with {e['refill']} mL.",
                    f"Use {e['n']} mL of the {c}; the protocol adds {e['refill']} mL "
                    f"whenever it falls below {e['threshold']} mL.",
                    f"Take {e['n']} mL from the {c}. Below {e['threshold']} mL triggers a "
                    f"{e['refill']} mL refill.",
                    f"Remove {e['n']} mL from the {c}, then: under {e['threshold']} mL "
                    f"means {e['refill']} mL are added.")[variant]
        if k == "dilute":
            return (f"Dilute the {c} with an equal volume of water, which doubles its volume.",
                    f"Add an equal volume of water to the {c} (its volume doubles).",
                    f"Perform a twofold dilution of the {c}; the volume doubles.",
                    f"Mix the {c} with the same volume of water, doubling it.")[variant]
        raise ValueError(k)

    def hypothetical_clause(self, e, variant):
        return (f"{e['n']} mL were poured from the {e['c']} into the {e['target']}",
                f"the {e['target']} received {e['n']} mL from the {e['c']}",
                f"we transferred {e['n']} mL out of the {e['c']} into the {e['target']}",
                f"{e['n']} mL moved from the {e['c']} to the {e['target']}")[variant]

    def make_distractor(self, state, rng):
        return {"c": rng.choice(self.CONTAINERS), "n": rng.randrange(20, 700)}

    def realize_distractor(self, d, variant):
        return (f"A colleague noted that the {d['c']} would ideally hold {d['n']} mL; this "
                "is only a note.",
                f"The {d['c']} has a capacity mark at {d['n']} mL, which is not its "
                "content.",
                f"Last week the {d['c']} was planned at {d['n']} mL, but plans changed.",
                f"For comparison only, a reference protocol uses {d['n']} mL in the "
                f"{d['c']}.")[variant]

    def opening(self, state):
        parts = [f"the {c} holds {state['ml'][c]} mL" for c in self.CONTAINERS]
        return ("We are running a lab protocol and tracking volumes. Currently "
                + "; ".join(parts) + ".", [state["ml"][c] for c in self.CONTAINERS])

    def summary(self, state):
        parts = [f"{c}: {v} mL" for c, v in state["ml"].items()]
        return "Verified container volumes: " + "; ".join(parts) + "."


# ── schedule constraints (logic) ────────────────────────────────────────────

class ScheduleConstraints(World):
    name = "schedule_constraints"
    domain = "logic/precedence"
    value_noun = "earliest finish day"
    entity_noun = "tasks"
    hypothetical_kind = "extend"
    TASKS = ("Survey", "Design", "Prototype", "Testing", "Review", "Procurement",
             "Assembly", "Training", "Rollout", "Audit")

    def init(self, rng):
        durations = {t: rng.randrange(2, 9) for t in self.TASKS}
        prereqs = {t: [] for t in self.TASKS}
        for index, task in enumerate(self.TASKS[1:], start=1):
            for earlier in rng.sample(self.TASKS[:index], k=min(index, rng.choice([1, 1, 2]))):
                prereqs[task].append(earlier)
        return {"duration": durations, "prereqs": {k: sorted(v, key=self.TASKS.index)
                                                   for k, v in prereqs.items()}}

    def entities(self, state):
        return list(self.TASKS)

    def label(self, entity):
        return entity

    def values(self, state):
        finish: dict[str, int] = {}
        for task in self.TASKS:  # topological by construction
            start = max((finish[p] for p in state["prereqs"][task]), default=0)
            finish[task] = start + state["duration"][task]
        return finish

    def simple_kinds(self):
        return ("extend", "shorten", "depend")

    def make_event(self, state, kind, rng):
        task = rng.choice(self.TASKS)
        duration = state["duration"][task]
        if kind == "extend":
            n = rng.randrange(1, 5)
            return {"kind": kind, "task": task, "n": n, "numbers": [n]}
        if kind == "shorten":
            if duration < 3:
                return None
            n = rng.randrange(1, duration - 1)
            return {"kind": kind, "task": task, "n": n, "numbers": [n]}
        if kind == "depend":
            index = self.TASKS.index(task)
            options = [t for t in self.TASKS[:index] if t not in state["prereqs"][task]]
            if not options:
                return None
            return {"kind": kind, "task": task, "prereq": rng.choice(options), "numbers": []}
        if kind == "correction":
            actual = duration + rng.choice([-1, 1]) * rng.randrange(1, 4)
            if actual < 1:
                return None
            return {"kind": kind, "task": task, "recorded": duration, "actual": actual,
                    "numbers": [duration, actual]}
        if kind == "conditional":
            other = rng.choice([t for t in self.TASKS[self.TASKS.index(task) + 1:]] or [None])
            if other is None:
                return None
            n = rng.randrange(1, 4)
            after = self.values(self.apply(state, {"kind": "extend", "task": task,
                                                   "n": n}))[task]
            threshold = max(1, after + rng.choice([-1, 1]) * rng.randrange(1, 5))
            k = rng.randrange(1, 4)
            return {"kind": kind, "task": task, "n": n, "threshold": threshold,
                    "other": other, "k": k, "numbers": [n, threshold, k]}
        raise ValueError(kind)

    def apply(self, state, event):
        state = copy.deepcopy(state)
        kind = event["kind"]
        if kind == "extend":
            state["duration"][event["task"]] += event["n"]
        elif kind == "shorten":
            state["duration"][event["task"]] -= event["n"]
        elif kind == "depend":
            prereqs = state["prereqs"][event["task"]]
            prereqs.append(event["prereq"])
            prereqs.sort(key=self.TASKS.index)
        elif kind == "correction":
            state["duration"][event["task"]] = event["actual"]
        elif kind == "conditional":
            state["duration"][event["task"]] += event["n"]
            if self.values(state)[event["task"]] > event["threshold"]:
                state["duration"][event["other"]] += event["k"]
        else:
            raise ValueError(kind)
        return state

    def realize_event(self, e, variant):
        k = e["kind"]
        t = e["task"]
        if k == "extend":
            return (f"{t} will take {e['n']} more days than planned.",
                    f"Extend {t} by {e['n']} days.",
                    f"The duration of {t} grows by {e['n']} days.",
                    f"Add {e['n']} days to {t}.")[variant]
        if k == "shorten":
            return (f"{t} can be done {e['n']} days faster.",
                    f"Shorten {t} by {e['n']} days.",
                    f"The duration of {t} drops by {e['n']} days.",
                    f"Remove {e['n']} days from {t}.")[variant]
        if k == "depend":
            return (f"{t} now cannot start until {e['prereq']} has finished.",
                    f"Add a dependency: {e['prereq']} must finish before {t} starts.",
                    f"New constraint: {t} waits for {e['prereq']}.",
                    f"{e['prereq']} has become a prerequisite of {t}.")[variant]
        if k == "correction":
            return (f"Correction: {t} was recorded as taking {e['recorded']} days but "
                    f"actually takes {e['actual']} days.",
                    f"The {t} estimate of {e['recorded']} days was wrong; it is "
                    f"{e['actual']} days.",
                    f"Fix {t}: {e['actual']} days, not {e['recorded']}.",
                    f"Revised: {t} needs {e['actual']} days instead of the recorded "
                    f"{e['recorded']}.")[variant]
        if k == "conditional":
            return (f"Extend {t} by {e['n']} days. If {t} then finishes after day "
                    f"{e['threshold']}, {e['other']} also takes {e['k']} more days.",
                    f"{t} grows by {e['n']} days; a finish later than day "
                    f"{e['threshold']} adds {e['k']} days to {e['other']}.",
                    f"Add {e['n']} days to {t}. Rule: finishing after day {e['threshold']} "
                    f"delays {e['other']} by {e['k']} days of extra work.",
                    f"{t} takes {e['n']} days longer, and if its finish passes day "
                    f"{e['threshold']}, {e['other']} needs {e['k']} more days."
                    )[variant]
        raise ValueError(k)

    def hypothetical_clause(self, e, variant):
        return (f"{e['task']} took {e['n']} more days",
                f"we extended {e['task']} by {e['n']} days",
                f"{e['n']} days were added to {e['task']}",
                f"the duration of {e['task']} grew by {e['n']} days")[variant]

    def make_distractor(self, state, rng):
        return {"task": rng.choice(self.TASKS), "n": rng.randrange(2, 15)}

    def realize_distractor(self, d, variant):
        return (f"Someone suggested {d['task']} might take {d['n']} days, but the plan is "
                "unchanged.",
                f"A similar project once spent {d['n']} days on {d['task']}; that is only "
                "background.",
                f"The {d['task']} lead would like {d['n']} days, which was not approved.",
                f"For reference, {d['task']} was budgeted at {d['n']} days in an old draft."
                )[variant]

    def opening(self, state):
        parts, numbers = [], [0]  # "day 0" in the opening
        for task in self.TASKS:
            prereqs = state["prereqs"][task]
            dep = (" after " + join_labels(prereqs)) if prereqs else " with no prerequisite"
            parts.append(f"{task} takes {state['duration'][task]} days{dep}")
            numbers.append(state["duration"][task])
        return ("We are planning a project schedule starting on day 0. A task starts when "
                "all its prerequisites have finished; its finish day is its start day plus "
                "its duration. " + "; ".join(parts) + ".", numbers)

    def summary(self, state):
        parts = []
        for task in self.TASKS:
            prereqs = state["prereqs"][task]
            dep = ("after " + join_labels(prereqs)) if prereqs else "no prerequisite"
            parts.append(f"{task}: {state['duration'][task]} days, {dep}")
        return ("Verified schedule (start day 0; a task starts when all prerequisites "
                "finish): " + "; ".join(parts) + ".")


WORLDS: dict[str, World] = {w.name: w for w in (BudgetLedger(), InventoryLogistics(),
                                                 LabProtocol(), ScheduleConstraints())}


class ScheduleConstraintsV2(ScheduleConstraints):
    """G1 v2: the record also states each task's verified earliest finish day.

    In v1 even a level-1 query needed a full longest-path computation over the
    10-task graph (G1 v1 diagnosis). Here the model only has to propagate the
    effect of the current turn's events through the dependency graph.
    """

    def opening(self, state):
        text, numbers = super().opening(state)
        finish = self.values(state)
        listed = "; ".join(f"{task} finishes on day {finish[task]}" for task in self.TASKS)
        return (text + " Current earliest finish days: " + listed + ".",
                numbers + [finish[task] for task in self.TASKS])

    def summary(self, state):
        finish = self.values(state)
        listed = "; ".join(f"{task}: finishes on day {finish[task]}" for task in self.TASKS)
        return super().summary(state) + " Current earliest finish days: " + listed + "."


class MilestoneChains(ScheduleConstraintsV2):
    """Shallow precedence logic (investigator option 2b, after G1 v1/v2).

    Ten milestones form four independent chains of lengths 3, 3, 2 and 2; every
    milestone has at most one prerequisite, so a change propagates at most two
    steps. The record states each milestone's earliest finish day.
    """

    name = "milestone_chains"
    domain = "logic/precedence (shallow chains)"
    value_noun = "earliest finish day"
    entity_noun = "milestones"
    hypothetical_kind = "extend"
    TASKS = ("Kickoff", "Draft", "Approval", "Sampling", "Analysis", "Report",
             "Ordering", "Delivery", "Briefing", "Launch")
    CHAIN_LENGTHS = (3, 3, 2, 2)

    def init(self, rng):
        durations = {t: rng.randrange(2, 9) for t in self.TASKS}
        prereqs: dict[str, list[str]] = {t: [] for t in self.TASKS}
        start = 0
        for length in self.CHAIN_LENGTHS:
            chain = self.TASKS[start:start + length]
            for earlier, later in zip(chain, chain[1:], strict=False):
                prereqs[later] = [earlier]
            start += length
        return {"duration": durations, "prereqs": prereqs}

    def simple_kinds(self):
        return ("extend", "shorten")


WORLDS_V2: dict[str, World] = {**WORLDS, "schedule_constraints": ScheduleConstraintsV2(),
                               "milestone_chains": MilestoneChains()}


def get_world(name: str, version: str = "v1") -> World:
    if version == "v1":
        return WORLDS[name]
    if version == "v2":
        return WORLDS_V2[name]
    raise ValueError(f"unknown instrument version {version!r}")
