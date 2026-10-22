"""Deterministic, exact-scored longitudinal dialogue streams for CDR-v1."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

PROTOCOL_ID = "CDR-v1"
GENERATOR_VERSION = "controlled-dialogue-v1"
TEMPLATES = ("research_budget", "software_incident", "study_cohort")
CONDITIONS = ("self_history", "oracle_history", "state_summary")


def _split_for_index(index: int) -> str:
    if index < 4:
        return "calibration"
    if index < 7:
        return "development"
    return "holdout"


def solve_dialogue_state(template: str, state: dict[str, Any]) -> int:
    if template == "research_budget":
        return int(state["authorized"] - sum(state["spent"].values()))
    if template == "software_incident":
        return int(5 * state["critical"] + 2 * state["major"] + state["minor"])
    if template == "study_cohort":
        return int(
            sum(
                site["enrolled"] - site["excluded"] - site["withdrawn"]
                for site in state["sites"].values()
            )
        )
    raise ValueError(f"unknown dialogue template: {template}")


def state_summary(template: str, state: dict[str, Any]) -> str:
    if template == "research_budget":
        spent = ", ".join(
            f"{name}={value}" for name, value in sorted(state["spent"].items())
        )
        return (
            f"Authorized budget={state['authorized']}. Current spending: {spent}. "
            "Remaining funds equal authorized budget minus total spending."
        )
    if template == "software_incident":
        return (
            f"Open incidents: critical={state['critical']}, major={state['major']}, "
            f"minor={state['minor']}. Risk score=5*critical+2*major+minor."
        )
    if template == "study_cohort":
        sites = "; ".join(
            f"{name}: enrolled={value['enrolled']}, excluded={value['excluded']}, "
            f"withdrawn={value['withdrawn']}"
            for name, value in sorted(state["sites"].items())
        )
        return (
            f"Current cohort by site: {sites}. Analyzable participants equal "
            "enrolled minus excluded minus withdrawn, summed across sites."
        )
    raise ValueError(f"unknown dialogue template: {template}")


def _turn(
    *,
    dialogue_id: str,
    template: str,
    split: str,
    number: int,
    before: dict[str, Any],
    after: dict[str, Any],
    prompt: str,
    paraphrases: tuple[str, str],
    update_count: int,
) -> dict[str, Any]:
    gold = solve_dialogue_state(template, after)
    return {
        "qid": f"{dialogue_id}-T{number:02d}",
        "dialogue_id": dialogue_id,
        "template": template,
        "split": split,
        "turn": number,
        "state_before": copy.deepcopy(before),
        "state_after": copy.deepcopy(after),
        "state_summary_before": state_summary(template, before),
        "user_prompt": prompt,
        "paraphrases": list(paraphrases),
        "gold_answer": str(gold),
        "oracle_response": (
            "Using the verified running state, the requested value is "
            f"{gold}.\nFINAL_ANSWER: {gold}"
        ),
        "intrinsic_updates": int(update_count),
    }


def _budget_dialogue(index: int, rng: random.Random) -> dict[str, Any]:
    dialogue_id = f"cdr-v1-budget-{index + 1:03d}"
    split = _split_for_index(index)
    state: dict[str, Any] = {"authorized": 0, "spent": {}}
    turns = []
    categories = ("equipment", "personnel", "travel")

    before = copy.deepcopy(state)
    authorized = rng.randrange(110_000, 160_001, 5_000)
    spent = {name: rng.randrange(6_000, 22_001, 500) for name in categories}
    state = {"authorized": authorized, "spent": spent}
    prompt = (
        "We are starting a continuing research-project budget review. The authorized "
        f"budget is {authorized}. Spending recorded so far is equipment {spent['equipment']}, "
        f"personnel {spent['personnel']}, and travel {spent['travel']}. What amount remains?"
    )
    turns.append(
        _turn(
            dialogue_id=dialogue_id,
            template="research_budget",
            split=split,
            number=1,
            before=before,
            after=state,
            prompt=prompt,
            paraphrases=(
                prompt.replace("What amount remains?", "How much of the budget remains?"),
                prompt.replace("Spending recorded so far is", "The current expenditures are"),
            ),
            update_count=4,
        )
    )

    for number in range(2, 13):
        before = copy.deepcopy(state)
        category = categories[(number + index) % len(categories)]
        if number in {5, 10}:
            amount = rng.randrange(4_000, 10_001, 500)
            state["authorized"] += amount
            prompt = (
                f"The sponsor has added {amount} to the authorized budget. Keep every "
                "earlier expenditure in the running record. What amount now remains?"
            )
            variants = (
                f"Increase the authorized budget by {amount}, without changing prior spending. "
                "What is the new remaining amount?",
                f"A supplemental award of {amount} has arrived. With the existing expense "
                "history unchanged, how much remains?",
            )
        elif number in {7, 11}:
            amount = min(
                state["spent"][category], rng.randrange(1_000, 4_001, 500)
            )
            state["spent"][category] -= amount
            prompt = (
                f"A refund of {amount} has been credited against {category} spending. "
                "Apply that correction to our running budget. What amount remains?"
            )
            variants = (
                f"Reduce the recorded {category} expenditure by {amount} because of a refund. "
                "Using the rest of our running record, how much remains?",
                f"Correct {category} spending downward by {amount}. What is the updated "
                "unspent budget?",
            )
        else:
            amount = rng.randrange(1_500, 7_001, 500)
            state["spent"][category] += amount
            prompt = (
                f"Record a new {category} expenditure of {amount} in the same project. "
                "Using the complete running budget, what amount remains?"
            )
            variants = (
                f"Add {amount} to {category} spending and retain all earlier entries. "
                "How much of the authorized budget is left?",
                f"The project has now spent another {amount} on {category}. What is the "
                "updated remaining amount?",
            )
        turns.append(
            _turn(
                dialogue_id=dialogue_id,
                template="research_budget",
                split=split,
                number=number,
                before=before,
                after=state,
                prompt=prompt,
                paraphrases=variants,
                update_count=1,
            )
        )
    return {"dialogue_id": dialogue_id, "template": "research_budget", "split": split, "turns": turns}


def _incident_dialogue(index: int, rng: random.Random) -> dict[str, Any]:
    dialogue_id = f"cdr-v1-incident-{index + 1:03d}"
    split = _split_for_index(index)
    state = {"critical": 0, "major": 0, "minor": 0}
    turns = []
    before = copy.deepcopy(state)
    state = {
        "critical": rng.randint(1, 3),
        "major": rng.randint(4, 8),
        "minor": rng.randint(6, 12),
    }
    prompt = (
        "We are opening a continuing software-incident review. There are currently "
        f"{state['critical']} critical, {state['major']} major, and {state['minor']} minor "
        "open incidents. Use risk score = 5*critical + 2*major + minor throughout this "
        "conversation. What is the current risk score?"
    )
    turns.append(
        _turn(
            dialogue_id=dialogue_id,
            template="software_incident",
            split=split,
            number=1,
            before=before,
            after=state,
            prompt=prompt,
            paraphrases=(
                prompt.replace("What is the current risk score?", "Calculate the present risk score."),
                prompt.replace("We are opening", "Begin"),
            ),
            update_count=3,
        )
    )

    severities = ("critical", "major", "minor")
    for number in range(2, 13):
        before = copy.deepcopy(state)
        severity = severities[(number + 2 * index) % len(severities)]
        if number in {4, 7, 10} and state[severity] > 0:
            amount = min(state[severity], rng.randint(1, 2))
            state[severity] -= amount
            prompt = (
                f"The response team has resolved {amount} {severity} incident(s). Update the "
                "same incident record and report the current risk score."
            )
            variants = (
                f"Remove {amount} resolved {severity} incident(s) from our running case. "
                "What is the updated risk score?",
                f"Close {amount} of the open {severity} incidents. Using the established "
                "scoring rule, calculate the new score.",
            )
        else:
            amount = rng.randint(1, 3)
            state[severity] += amount
            prompt = (
                f"Monitoring has identified {amount} additional {severity} incident(s). Add "
                "them to the continuing record and report the current risk score."
            )
            variants = (
                f"Increase the open {severity} count by {amount}. Keeping the earlier record, "
                "what is the new risk score?",
                f"There are {amount} new {severity} incident(s). Apply the same scoring rule "
                "to the updated running totals.",
            )
        turns.append(
            _turn(
                dialogue_id=dialogue_id,
                template="software_incident",
                split=split,
                number=number,
                before=before,
                after=state,
                prompt=prompt,
                paraphrases=variants,
                update_count=1,
            )
        )
    return {"dialogue_id": dialogue_id, "template": "software_incident", "split": split, "turns": turns}


def _cohort_dialogue(index: int, rng: random.Random) -> dict[str, Any]:
    dialogue_id = f"cdr-v1-cohort-{index + 1:03d}"
    split = _split_for_index(index)
    state: dict[str, Any] = {"sites": {}}
    turns = []
    sites = ("North", "Central", "South")
    before = copy.deepcopy(state)
    state = {
        "sites": {
            site: {
                "enrolled": rng.randint(35, 60),
                "excluded": rng.randint(2, 6),
                "withdrawn": rng.randint(1, 4),
            }
            for site in sites
        }
    }
    details = "; ".join(
        f"{site}: enrolled {values['enrolled']}, excluded {values['excluded']}, "
        f"withdrawn {values['withdrawn']}"
        for site, values in state["sites"].items()
    )
    prompt = (
        "We are beginning a continuing multicenter study review. For each site, analyzable "
        "participants equal enrolled minus excluded minus withdrawn. Current records are "
        f"{details}. How many participants are analyzable across all sites?"
    )
    turns.append(
        _turn(
            dialogue_id=dialogue_id,
            template="study_cohort",
            split=split,
            number=1,
            before=before,
            after=state,
            prompt=prompt,
            paraphrases=(
                prompt.replace("How many participants are analyzable across all sites?", "Find the total analyzable cohort."),
                prompt.replace("Current records are", "The present site records are"),
            ),
            update_count=9,
        )
    )

    fields = ("enrolled", "excluded", "withdrawn")
    for number in range(2, 13):
        before = copy.deepcopy(state)
        site = sites[(number + index) % len(sites)]
        field = fields[(number + 2 * index) % len(fields)]
        if field == "enrolled":
            amount = rng.randint(2, 7)
            state["sites"][site][field] += amount
            action = f"enrolled {amount} additional participants"
        elif number in {6, 11} and state["sites"][site][field] > 0:
            amount = min(state["sites"][site][field], rng.randint(1, 2))
            state["sites"][site][field] -= amount
            action = f"corrected the {field} count downward by {amount}"
        else:
            capacity = (
                state["sites"][site]["enrolled"]
                - state["sites"][site]["excluded"]
                - state["sites"][site]["withdrawn"]
            )
            amount = min(max(capacity, 1), rng.randint(1, 3))
            state["sites"][site][field] += amount
            action = f"recorded {amount} additional {field} participant(s)"
        prompt = (
            f"The {site} site has {action}. Apply this update to the continuing study "
            "record. How many participants are now analyzable across all sites?"
        )
        variants = (
            f"Update the {site} site: it has {action}. Using all earlier site records, "
            "what is the new total analyzable cohort?",
            f"For {site}, {action}. After incorporating that change, calculate the current "
            "number of analyzable participants across the study.",
        )
        turns.append(
            _turn(
                dialogue_id=dialogue_id,
                template="study_cohort",
                split=split,
                number=number,
                before=before,
                after=state,
                prompt=prompt,
                paraphrases=variants,
                update_count=1,
            )
        )
    return {"dialogue_id": dialogue_id, "template": "study_cohort", "split": split, "turns": turns}


def generate_dialogue_bank(
    seed: int = 20260927,
    dialogues_per_template: int = 10,
) -> dict[str, Any]:
    if dialogues_per_template != 10:
        raise ValueError("CDR-v1 freezes exactly 10 dialogues per template")
    rng = random.Random(seed)
    builders = (_budget_dialogue, _incident_dialogue, _cohort_dialogue)
    dialogues = [
        builder(index, rng)
        for builder in builders
        for index in range(dialogues_per_template)
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "conditions": list(CONDITIONS),
        "dialogues": dialogues,
    }


def validate_dialogue_bank(bank: dict[str, Any]) -> list[str]:
    problems = []
    if bank.get("protocol_id") != PROTOCOL_ID:
        problems.append("protocol_id")
    dialogues = bank.get("dialogues") or []
    if len(dialogues) != 30:
        problems.append("dialogue_count")
    ids = [dialogue.get("dialogue_id") for dialogue in dialogues]
    if len(ids) != len(set(ids)):
        problems.append("unique_dialogue_ids")
    for dialogue in dialogues:
        template = dialogue.get("template")
        turns = dialogue.get("turns") or []
        if template not in TEMPLATES or len(turns) != 12:
            problems.append(f"shape:{dialogue.get('dialogue_id')}")
            continue
        previous_after = None
        for expected_turn, turn in enumerate(turns, start=1):
            if int(turn.get("turn", -1)) != expected_turn:
                problems.append(f"turn_order:{turn.get('qid')}")
            if previous_after is not None and turn.get("state_before") != previous_after:
                problems.append(f"state_continuity:{turn.get('qid')}")
            if str(solve_dialogue_state(template, turn["state_after"])) != str(
                turn.get("gold_answer")
            ):
                problems.append(f"gold:{turn.get('qid')}")
            paraphrases = turn.get("paraphrases") or []
            if len(paraphrases) != 2 or len(set(paraphrases)) != 2:
                problems.append(f"paraphrases:{turn.get('qid')}")
            previous_after = turn.get("state_after")
    return problems


def dialogue_bank_sha256(bank: dict[str, Any]) -> str:
    canonical = json.dumps(bank, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_dialogue_bank(path: str | Path, bank: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
