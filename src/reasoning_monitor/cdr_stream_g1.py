"""CDR-Stream G1: single-turn difficulty calibration (items, prompt, frozen rule).

G1 measures single-turn CQ at every ladder level, in exactly the control-arm
format (verified state summary + one user turn), for every (model, template).
A frozen rule then maps each (model, template) to easy/medium/hard centre levels
inside accuracy bands, or excludes it. Protocol: CDR_STREAM_G1_PROTOCOL.md.

Items come from a separate calibration namespace: fresh states reached by seeded
random walks, never from bank dialogues. Nothing here calls a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_monitor.data import cdr_stream as cs
from reasoning_monitor.data.cdr_stream_worlds import N_VARIANTS
from reasoning_monitor.lost_conversation_semantic_rescore import semantic_parse

G1_ID = "CDR-Stream-v1-G1"
G1_SEED = 20261005
ITEMS_PER_CELL = 30          # 5 per query family
PRIMARY_LEVELS = cs.LEVELS   # 1..8
CONTINGENCY_LEVELS = cs.CONTINGENCY_LEVELS  # 9..10, G1b only
MAX_WALK_STEPS = 99


@dataclass(frozen=True)
class G1Spec:
    """One frozen calibration instrument: id, item seed and ladder/world version."""

    g1_id: str
    seed: int
    version: str
    templates: tuple[str, ...] | None = None

    @property
    def template_list(self) -> tuple[str, ...]:
        return self.templates if self.templates is not None else cs.TEMPLATES

    @property
    def primary_levels(self) -> tuple[int, ...]:
        return cs.PRIMARY_LEVELS[self.version]

    @property
    def contingency_levels(self) -> tuple[int, ...]:
        return cs.CONTINGENCY[self.version]


# Template set of the stream study after investigator option 2b: the v2
# schedule template is replaced by the shallower milestone_chains logic template.
FINAL_TEMPLATES = ("budget_ledger", "inventory_logistics", "lab_protocol",
                   "milestone_chains")

SPECS = {"v1": G1Spec(G1_ID, G1_SEED, "v1"),
         "v2": G1Spec("CDR-Stream-v1-G1v2", 20261007, "v2"),
         # Same seed and ladder as v2: items of the three shared templates are
         # regenerated bit-identically (checked against the frozen v2 file).
         "v2ext": G1Spec("CDR-Stream-v1-G1v2ext", 20261007, "v2", FINAL_TEMPLATES),
         # Third core model after Gemma 4 hit the ceiling: same frozen items.
         "v2gemma3": G1Spec("CDR-Stream-v1-G1v2gemma3", 20261007, "v2",
                            FINAL_TEMPLATES),
         # Llama replacement after every Llama-4 endpoint failed K diversity in
         # stream probes: Llama-3 70B on the same frozen items.
         "v2llama3_70b": G1Spec("CDR-Stream-v1-G1v2llama3-70b", 20261007, "v2",
                                FINAL_TEMPLATES),
         # Frontier external validation (Claude Sonnet 5): same frozen items.
         "v2frontier": G1Spec("CDR-Stream-v1-G1v2frontier", 20261007, "v2",
                              FINAL_TEMPLATES)}
V1 = SPECS["v1"]

TARGETS = {"E": 0.85, "M": 0.70, "H": 0.50}
BANDS = {"E": (0.75, 0.92), "M": (0.60, 0.78), "H": (0.35, 0.65)}
MIN_ADJACENT_GAP = 0.10
MIN_EASY_HARD_GAP = 0.25
G1B_TRIGGER = BANDS["H"][1]          # fitted CQ at level 8 above this -> run G1b
TREND_ALPHA = 0.01
MIN_CONTRACT_VALIDITY = 0.90
MAX_TEMPLATE_EXCLUSIONS = 2          # a model excluded on more than 2 of 4 is dropped

# Frozen prompt, shared with the CDR-Stream control arm.
SYSTEM_PROMPT = ("You keep exact track of an evolving record for a user. Use only the "
                 "information provided in the message. Compute every quantity exactly.")
RESPONSE_CONTRACT = ("RESPONSE CONTRACT: Give at most eight short numbered reasoning lines. "
                     "Then end with exactly one final line in the form "
                     "FINAL_ANSWER: <integer>")


def control_messages(state_summary: str, turn_text: str) -> list[dict[str, str]]:
    content = (f"Current verified record:\n{state_summary}\n\n"
               f"New message:\n{turn_text}\n\n{RESPONSE_CONTRACT}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + RESPONSE_CONTRACT).encode()).hexdigest()


# ── items ───────────────────────────────────────────────────────────────────

def _walk_state(world, rng: random.Random) -> dict[str, Any]:
    state = world.init(rng)
    for _ in range(rng.randrange(0, MAX_WALK_STEPS + 1)):
        kinds = world.simple_kinds() + world.complex_kinds(1) + world.complex_kinds(2)
        for _attempt in range(10):
            event = world.make_event(state, rng.choice(kinds), rng)
            if event is not None:
                state = world.apply(state, event)
                break
    return state


def make_item(template: str, level: int, family: str, index: int,
              spec: G1Spec = V1) -> dict[str, Any]:
    world = cs.get_world(template, spec.version)
    rng = random.Random(cs._derive_seed(spec.seed, template, level, family, index))
    state = _walk_state(world, rng)
    summary = world.summary(state)
    ladder_spec = cs.LADDERS[spec.version][level]
    _, n_events, n_distractors, complexity = ladder_spec
    events, after = cs._generate_events(world, state, n_events, complexity, rng)
    changed = sorted({e for ev in events for e in cs.event_entities(ev)})
    distractors = []
    for _ in range(n_distractors):
        d = world.make_distractor(after, rng)
        d["numbers"] = [d["n"]]
        distractors.append(d)
    query = cs.make_query(world, after, family, ladder_spec, changed, rng)
    gold = cs.evaluate_query(world, after, query)
    variant = rng.randrange(N_VARIANTS)
    sentences = [world.realize_event(ev, (variant + i) % N_VARIANTS)
                 for i, ev in enumerate(events)]
    for j, d in enumerate(distractors):
        sentences.insert(rng.randrange(len(sentences) + 1),
                         world.realize_distractor(d, (variant + j) % N_VARIANTS))
    text = " ".join(sentences + [cs.realize_query(world, query, variant)])
    expected = ([n for ev in events for n in ev["numbers"]]
                + [n for d in distractors for n in d["numbers"]] + query["numbers"])
    return {"item_id": f"{template}/L{level:02d}/{family}/{index:02d}",
            "template": template, "level": level, "family": family,
            "contingency": level in spec.contingency_levels,
            "state_summary": summary, "turn_text": text, "gold": gold,
            "expected_numbers": expected,
            "axes": {"query_terms": len(query["A"]) + len(query["B"]),
                     "events": len(events), "events_planned": n_events,
                     "distractors": len(distractors), "complexity": complexity,
                     "filter": query["filter"]}}


def generate_items(spec: G1Spec = V1) -> list[dict[str, Any]]:
    items = []
    per_family = ITEMS_PER_CELL // len(cs.FAMILIES)
    for template in spec.template_list:
        for level in spec.primary_levels + spec.contingency_levels:
            for family in cs.FAMILIES:
                for index in range(per_family):
                    items.append(make_item(template, level, family, index, spec))
    return items


def items_sha256(items: Sequence[Mapping]) -> str:
    return hashlib.sha256(json.dumps(list(items), sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def validate_items(items: Sequence[Mapping]) -> list[str]:
    import re

    problems = []
    seen = set()
    for item in items:
        if item["item_id"] in seen:
            problems.append(f"duplicate {item['item_id']}")
        seen.add(item["item_id"])
        found = sorted(int(x) for x in re.findall(r"\d+", item["turn_text"]))
        if found != sorted(item["expected_numbers"]):
            problems.append(f"{item['item_id']}: numbers differ from spec")
        if not isinstance(item["gold"], int) or item["gold"] < 0:
            problems.append(f"{item['item_id']}: invalid gold")
        if item["axes"]["events"] < item["axes"]["events_planned"]:
            problems.append(f"{item['item_id']}: fewer events than planned")
    cells = defaultdict(int)
    for item in items:
        cells[(item["template"], item["level"])] += 1
    if any(n != ITEMS_PER_CELL for n in cells.values()):
        problems.append("unequal cell sizes")
    return problems


# ── frozen assessment rule ──────────────────────────────────────────────────

def isotonic_decreasing(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Pool-adjacent-violators fit constrained to be non-increasing."""
    blocks = [[v, w, 1] for v, w in zip(values, weights, strict=True)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = []
    for value, _, count in blocks:
        fitted += [value] * count
    return fitted


def cochran_armitage_decreasing(levels: Sequence[int], correct: Sequence[int],
                                totals: Sequence[int]) -> float:
    """One-sided p-value for a decreasing success trend (normal approximation)."""
    n = sum(totals)
    p = sum(correct) / n
    if p in (0.0, 1.0):
        return 1.0
    mean_level = sum(t * x for t, x in zip(totals, levels, strict=True)) / n
    statistic = sum(x * (c - t * p)
                    for x, c, t in zip(levels, correct, totals, strict=True))
    variance = p * (1 - p) * sum(t * (x - mean_level) ** 2
                                 for x, t in zip(levels, totals, strict=True))
    if variance <= 0:
        return 1.0
    z = statistic / math.sqrt(variance)
    return 0.5 * math.erfc(-z / math.sqrt(2))  # P(Z <= z): small when decreasing


def select_mapping(fitted: Mapping[int, float]) -> dict[str, int] | None:
    """Ordered (e, m, h) inside bands with required gaps, closest to targets."""
    levels = sorted(fitted)
    best, best_score = None, None
    for e in levels:
        for m in levels:
            for h in levels:
                if not e < m < h:
                    continue
                f = {"E": fitted[e], "M": fitted[m], "H": fitted[h]}
                if any(not BANDS[r][0] <= f[r] <= BANDS[r][1] for r in f):
                    continue
                if (f["E"] - f["M"] < MIN_ADJACENT_GAP or f["M"] - f["H"] < MIN_ADJACENT_GAP
                        or f["E"] - f["H"] < MIN_EASY_HARD_GAP):
                    continue
                score = sum((f[r] - TARGETS[r]) ** 2 for r in f)
                key = (round(score, 12), e, m, h)
                if best_score is None or key < best_score:
                    best, best_score = {"E": e, "M": m, "H": h}, key
    return best


def assess_cell(rows: Sequence[Mapping], include_contingency: bool,
                spec: G1Spec = V1) -> dict[str, Any]:
    by_level: dict[int, list[int]] = defaultdict(list)
    valid = 0
    for row in rows:
        if row["contingency"] and not include_contingency:
            continue
        by_level[row["level"]].append(int(row["correct"]))
        valid += int(row["contract_valid"])
    levels = sorted(by_level)
    totals = [len(by_level[x]) for x in levels]
    correct = [sum(by_level[x]) for x in levels]
    raw = [c / t for c, t in zip(correct, totals, strict=True)]
    fitted_list = isotonic_decreasing(raw, totals)
    fitted = dict(zip(levels, fitted_list, strict=True))
    p_trend = cochran_armitage_decreasing(levels, correct, totals)
    mapping = select_mapping(fitted) if p_trend <= TREND_ALPHA else None
    if mapping is not None:
        status = "mapped"
    elif (not include_contingency and p_trend <= TREND_ALPHA
          and fitted[max(spec.primary_levels)] > G1B_TRIGGER):
        status = "needs_G1b"
    else:
        status = "excluded"
    return {"levels": levels, "n": totals, "correct": correct, "raw_cq": raw,
            "fitted_cq": fitted_list, "trend_p_one_sided": p_trend,
            "contract_validity": valid / max(1, sum(totals)),
            "mapping": mapping, "status": status}


def assess_g1(records: Sequence[Mapping], models: Sequence[str],
              spec: G1Spec = V1) -> dict[str, Any]:
    """Apply the frozen G1 rule. Invalid or missing answers count as incorrect."""
    by_cell: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in records:
        by_cell[(row["model"], row["template"])].append(row)
    cells: dict[str, dict[str, Any]] = {}
    models_report = {}
    for model in models:
        excluded = 0
        model_rows = [r for r in records if r["model"] == model]
        validity = (sum(r["contract_valid"] for r in model_rows) / len(model_rows)
                    if model_rows else 0.0)
        truncated = sum(r["truncated"] for r in model_rows)
        for template in spec.template_list:
            rows = by_cell.get((model, template), [])
            has_contingency = any(r["contingency"] for r in rows)
            result = assess_cell(rows, include_contingency=has_contingency,
                                 spec=spec) if rows else {
                "status": "not_run"}
            cells[f"{model}|{template}"] = result
            excluded += result["status"] == "excluded"
        operational = validity >= MIN_CONTRACT_VALIDITY and truncated == 0
        models_report[model] = {
            "contract_validity": validity, "truncated": truncated,
            "operational_pass": operational, "templates_excluded": excluded,
            "retained": operational and excluded <= MAX_TEMPLATE_EXCLUSIONS
            and not any(cells[f"{model}|{t}"]["status"] == "needs_G1b"
                        for t in spec.template_list),
        }
    return {"g1_id": spec.g1_id, "version": spec.version, "targets": TARGETS,
            "bands": BANDS,
            "rule": {"min_adjacent_gap": MIN_ADJACENT_GAP,
                     "min_easy_hard_gap": MIN_EASY_HARD_GAP, "trend_alpha": TREND_ALPHA,
                     "g1b_trigger_fitted_level8_above": G1B_TRIGGER,
                     "min_contract_validity": MIN_CONTRACT_VALIDITY,
                     "max_template_exclusions": MAX_TEMPLATE_EXCLUSIONS},
            "cells": cells, "models": models_report,
            "needs_G1b": sorted(k for k, v in cells.items() if v["status"] == "needs_G1b")}


def messages_sha256(messages: Sequence[Mapping]) -> str:
    return hashlib.sha256(json.dumps(list(messages), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def planned_calls(items: Sequence[Mapping], models: Sequence[Mapping],
                  g1b_cells: Sequence[str] = (), order_seed: int = 0
                  ) -> list[tuple[Mapping, Mapping]]:
    """Primary items for every model, plus contingency items for G1b cells.

    The order is shuffled per model with a fixed seed so that any drift in the
    gateway during the run is spread over levels instead of confounding them.
    """
    plan = []
    for model in models:
        allowed = model.get("templates")
        selected = [item for item in items
                    if (allowed is None or item["template"] in allowed)
                    and (not item["contingency"]
                         or f"{model['name']}|{item['template']}" in g1b_cells)]
        order = list(selected)
        random.Random(cs._derive_seed(order_seed, model["name"])).shuffle(order)
        plan += [(model, item) for item in order]
    return plan


MAX_CONSECUTIVE_FAILURES = 10


class CircuitBreakerOpen(RuntimeError):
    """Too many consecutive failed calls; stop and resume later."""


def load_checkpoint(checkpoint: Path, fingerprint: str) -> tuple[dict, int]:
    """Return {key: latest successful row} and the number of failed attempts.

    Failed calls (ok=False, e.g. exhausted retries) stay in the file for audit
    but do not count as completed, so they are retried on the next run. A
    partially written final line (interrupted write) is moved aside.
    """
    successes: dict[tuple[str, str], dict] = {}
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
            raise ValueError("Checkpoint belongs to a different frozen G1 setup")
        if row["ok"]:
            successes[(row["model"], row["item_id"])] = row
        else:
            failures += 1
    return successes, failures


def combine_mappings(assessments: Sequence[Mapping],
                     templates: Sequence[str] = FINAL_TEMPLATES,
                     declared_exclusions: Mapping[str, Mapping[str, str]] | None = None
                     ) -> dict[str, Any]:
    """Final per-(model, template) mapping across calibration runs.

    Later assessments take precedence for cells they actually ran. Retention
    applies the unchanged frozen rule over the final template set; a model must
    be operationally clean in every run that measured it.
    """
    cells: dict[str, Mapping] = {}
    operational: dict[str, bool] = {}
    for assessment in assessments:
        for key, cell in assessment["cells"].items():
            model, template = key.split("|")
            if template in templates and cell.get("status") != "not_run":
                cells[key] = {**cell, "source": assessment["g1_id"]}
        for model, report in assessment["models"].items():
            ran = any(k.startswith(model + "|") and v.get("status") != "not_run"
                      for k, v in assessment["cells"].items())
            if ran:
                operational[model] = operational.get(model, True) and report["operational_pass"]
    # Exclusions declared before any accuracy was seen (e.g. a gateway
    # classifier truncating one template) count as excluded, not pending.
    for model, reasons in (declared_exclusions or {}).items():
        for template, reason in reasons.items():
            if f"{model}|{template}" not in cells:
                cells[f"{model}|{template}"] = {"status": "excluded", "mapping": None,
                                                "source": "declared", "reason": reason}
    models = {}
    for model, ok in operational.items():
        statuses = {t: cells.get(f"{model}|{t}", {"status": "not_run"})["status"]
                    for t in templates}
        excluded = sum(status == "excluded" for status in statuses.values())
        models[model] = {
            "operational_pass": ok, "statuses": statuses, "templates_excluded": excluded,
            "pending": [t for t, status in statuses.items()
                        if status in {"needs_G1b", "not_run"}],
            "retained": ok and excluded <= MAX_TEMPLATE_EXCLUSIONS
            and all(status in {"mapped", "excluded"} for status in statuses.values()),
            "mappings": {t: cells[f"{model}|{t}"]["mapping"] for t in templates
                         if statuses[t] == "mapped"},
        }
    return {"templates": list(templates), "models": models}


def smoke_subset(plan: Sequence[tuple[Mapping, Mapping]], per_cell: int = 1
                 ) -> list[tuple[Mapping, Mapping]]:
    """First ``per_cell`` primary calls per (model, template, level) in plan order.

    The subset uses the same keys, messages and seeds as the full plan, so a
    later full run reuses these records instead of repeating them.
    """
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    subset = []
    for model, item in plan:
        if item["contingency"]:
            continue
        cell = (model["name"], item["template"], item["level"])
        if counts[cell] < per_cell:
            counts[cell] += 1
            subset.append((model, item))
    return subset


def operational_summary(records: Sequence[Mapping]) -> dict[str, Any]:
    """Technical health only; deliberately reports no accuracy."""
    by_model: dict[str, list[Mapping]] = defaultdict(list)
    for row in records:
        by_model[row["model"]].append(row)
    return {model: {"calls": len(rows),
                    "contract_validity": sum(r["contract_valid"] for r in rows) / len(rows),
                    "truncated": sum(r["truncated"] for r in rows),
                    "max_output_tokens_seen": max(r["n_tokens"] for r in rows),
                    "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2]}
            for model, rows in by_model.items()}


def acquire(plan: Sequence[tuple[Mapping, Mapping]], clients: Mapping[str, Any],
            checkpoint: Path, fingerprint: str, max_tokens: int, callback=None,
            workers: int = 1, g1_id: str = G1_ID) -> list[dict[str, Any]]:
    """Resumable acquisition: every call is written raw before scoring is used.

    Successful calls are never repeated. Failed calls are recorded and retried
    on the next run. Raises CircuitBreakerOpen after MAX_CONSECUTIVE_FAILURES.

    With ``workers > 1`` the API calls run in a bounded thread pool. Only the
    calling thread writes the checkpoint, failure counting and the breaker, so
    the file format and resume rules are identical to sequential execution.
    Seeds depend on the call key, never on completion order.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if workers < 1:
        raise ValueError("workers must be >= 1")
    saved, _ = load_checkpoint(checkpoint, fingerprint)
    expected = {(model["name"], item["item_id"]) for model, item in plan}
    if set(saved) - expected:
        raise ValueError("Checkpoint contains calls outside the frozen plan")
    pending = []
    for model, item in plan:
        key = (model["name"], item["item_id"])
        messages = control_messages(item["state_summary"], item["turn_text"])
        request_hash = messages_sha256(messages)
        if key in saved:
            if saved[key]["messages_sha256"] != request_hash:
                raise ValueError(f"Changed request on resume: {key}")
            continue
        pending.append((model, item, key, messages, request_hash,
                        cs._derive_seed(fingerprint, *key) % (2 ** 31)))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def call(task):
        model, item, _key, messages, _hash, seed = task
        return clients[model["name"]].generate_messages(messages, seed=seed, gold="")

    consecutive_failures = 0
    tripped_message = None
    queue = iter(pending)
    with (checkpoint.open("a", encoding="utf-8") as stream,
          ThreadPoolExecutor(max_workers=workers) as pool):
        in_flight = {}

        def top_up():
            while tripped_message is None and len(in_flight) < workers:
                task = next(queue, None)
                if task is None:
                    return
                in_flight[pool.submit(call, task)] = task

        top_up()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                model, item, key, messages, request_hash, seed = in_flight.pop(future)
                sample = future.result()
                scored = score_response(sample.raw_text, item["gold"])
                finish = getattr(sample, "finish_reason", None)
                row = {"protocol_id": g1_id, "fingerprint": fingerprint,
                       "model": model["name"], "provider_model": model["provider_model"],
                       "item_id": item["item_id"], "template": item["template"],
                       "level": item["level"], "family": item["family"],
                       "contingency": item["contingency"], "gold": item["gold"],
                       "messages": messages, "messages_sha256": request_hash,
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
                    if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                            and tripped_message is None):
                        tripped_message = (
                            f"{consecutive_failures} consecutive failed calls; last: "
                            f"{sample.raw_text[:300]}. Completed calls are saved; rerun "
                            "the same command to resume.")
                    continue
                consecutive_failures = 0
                saved[key] = row
                if callback is not None:
                    callback(len(saved))
            top_up()
    if tripped_message is not None:
        raise CircuitBreakerOpen(tripped_message)
    return [saved[key] for key in (
        (model["name"], item["item_id"]) for model, item in plan) if key in saved]


def score_response(raw_text: str, gold: int) -> dict[str, Any]:
    parsed = semantic_parse(raw_text)
    correct = parsed["kind"] == "answer_attempt" and parsed["answer"] == str(gold)
    return {"answer": parsed["answer"], "response_kind": parsed["kind"],
            "contract_valid": bool(parsed["contract_valid"]), "correct": bool(correct)}


def plot_g1(assessment: Mapping, output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(assessment["models"])
    templates = sorted({key.split("|")[1] for key in assessment["cells"]},
                       key=lambda t: (t not in FINAL_TEMPLATES, t))
    fig, axes = plt.subplots(len(models), len(templates), figsize=(15, 3.4 * len(models)),
                             squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    colors = {"E": "#009E73", "M": "#E69F00", "H": "#D55E00"}
    for r, model in enumerate(models):
        for c, template in enumerate(templates):
            axis = axes[r, c]
            cell = assessment["cells"].get(f"{model}|{template}", {})
            for regime, (low, high) in BANDS.items():
                axis.axhspan(low, high, color=colors[regime], alpha=0.08)
            if "levels" in cell:
                axis.plot(cell["levels"], cell["raw_cq"], "o", color="#555555",
                          markersize=6, label="raw CQ")
                axis.plot(cell["levels"], cell["fitted_cq"], "-", color="#0072B2",
                          linewidth=2, label="isotonic fit")
                for regime, level in (cell.get("mapping") or {}).items():
                    axis.axvline(level, color=colors[regime], linestyle="--", linewidth=1.2)
                    axis.text(level, 1.02, regime, color=colors[regime], ha="center",
                              fontsize=8)
                axis.text(0.02, 0.04, f"{cell['status']}  n/level={cell['n'][0]}  "
                          f"trend p={cell['trend_p_one_sided']:.2g}",
                          transform=axis.transAxes, fontsize=7)
            axis.set_ylim(-0.02, 1.08)
            axis.grid(alpha=0.2)
            if r == 0:
                axis.set_title(template)
            if c == 0:
                axis.set_ylabel(f"{model}\nsingle-turn CQ")
            if r == len(models) - 1:
                axis.set_xlabel("Difficulty level")
    axes[0, 0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.1))
    fig.suptitle(f"{assessment['g1_id']} difficulty calibration (state-summary arm, K=1). "
                 "Shaded: E/M/H target bands; dashed: frozen-rule mapping", fontsize=10)
    names = []
    for extension in ("png", "pdf"):
        path = output / f"g1_calibration.{extension}"
        fig.savefig(path, dpi=160)
        names.append(path.name)
    plt.close(fig)
    return names
