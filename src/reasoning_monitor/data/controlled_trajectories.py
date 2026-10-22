"""Deterministic, solver-backed problem families for the RAT-v1 study."""

from __future__ import annotations

import hashlib
import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_monitor.types import Query

PROTOCOL_ID = "RAT-v1"
GENERATOR_VERSION = "controlled-trajectories-v1"
TEMPLATES = ("arithmetic_chain", "shortest_path", "precedence_schedule")
SPLIT_FAMILY_COUNTS = {"calibration": 8, "development": 6, "holdout": 6}
LEVEL_SIZES = {
    "arithmetic_chain": (2, 4, 7, 11, 16, 22),
    "shortest_path": (5, 7, 10, 14, 19, 25),
    "precedence_schedule": (5, 7, 10, 14, 19, 25),
}
OUTPUT_CONTRACT = (
    "Use at most eight concise numbered reasoning lines. Then put exactly one "
    "plain-text line in the form FINAL_ANSWER: <integer> last, outside LaTeX "
    "and code blocks. Do not write anything after that line."
)


@dataclass(frozen=True, slots=True)
class ControlledItem:
    qid: str
    family_id: str
    template: str
    split: str
    level: int
    prompt: str
    gold_answer: str
    paraphrases: tuple[str, ...]
    solution_steps: int
    structure: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "family_id": self.family_id,
            "template": self.template,
            "split": self.split,
            "level": self.level,
            "prompt": self.prompt,
            "gold_answer": self.gold_answer,
            "paraphrases": list(self.paraphrases),
            "solution_steps": self.solution_steps,
            "structure": self.structure,
        }


def _rng(seed: int, template: str, family_index: int) -> random.Random:
    return random.Random(f"{seed}:{template}:{family_index}:{GENERATOR_VERSION}")


def _with_contract(body: str) -> str:
    return f"{body.strip()}\n\nOutput requirements:\n{OUTPUT_CONTRACT}"


def _split_assignments(seed: int, families_per_template: int) -> dict[str, str]:
    if families_per_template != sum(SPLIT_FAMILY_COUNTS.values()):
        raise ValueError(
            "RAT-v1 requires exactly 20 families per template so its frozen "
            "8/6/6 family split remains unchanged"
        )
    assignments: dict[str, str] = {}
    for template_index, template in enumerate(TEMPLATES):
        indices = list(range(families_per_template))
        random.Random(seed + 1009 * (template_index + 1)).shuffle(indices)
        start = 0
        for split, count in SPLIT_FAMILY_COUNTS.items():
            for family_index in indices[start : start + count]:
                assignments[f"{template}:{family_index}"] = split
            start += count
    return assignments


def _apply_arithmetic(value: int, operation: dict[str, Any]) -> int:
    operand = int(operation["operand"])
    if operation["operator"] == "add":
        return value + operand
    if operation["operator"] == "subtract":
        return value - operand
    if operation["operator"] == "multiply":
        return value * operand
    raise ValueError(f"unknown arithmetic operator: {operation['operator']}")


def _arithmetic_family(
    seed: int,
    family_index: int,
    split: str,
    levels: int,
) -> list[ControlledItem]:
    rng = _rng(seed, "arithmetic_chain", family_index)
    start = rng.randint(12, 45)
    operations: list[dict[str, Any]] = []
    value = start
    operator_sequence: list[str] = []
    while len(operator_sequence) < LEVEL_SIZES["arithmetic_chain"][-1]:
        block = ["add", "multiply", "subtract"]
        rng.shuffle(block)
        operator_sequence.extend(block)
    for operator in operator_sequence[: LEVEL_SIZES["arithmetic_chain"][-1]]:
        if operator == "multiply" and abs(value) > 5000:
            operator = "add"
        operand = rng.randint(2, 9) if operator != "multiply" else rng.randint(2, 4)
        if operator == "subtract" and value - operand <= 0:
            operator = "add"
        operation = {"operator": operator, "operand": operand}
        value = _apply_arithmetic(value, operation)
        operations.append(operation)

    family_id = f"arithmetic-{family_index + 1:03d}"
    items: list[ControlledItem] = []
    for level, operation_count in enumerate(
        LEVEL_SIZES["arithmetic_chain"], start=1
    ):
        prefix = operations[:operation_count]
        answer = start
        for operation in prefix:
            answer = _apply_arithmetic(answer, operation)
        imperative = {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply by",
        }
        steps = [
            f"{index}. {imperative[operation['operator']]} {operation['operand']}"
            for index, operation in enumerate(prefix, start=1)
        ]
        compact = "; ".join(
            f"{imperative[operation['operator']]} {operation['operand']}"
            for operation in prefix
        )
        body = (
            f"Begin with the integer {start}. Apply these operations in order:\n"
            + "\n".join(steps)
            + "\nWhat integer do you obtain at the end?"
        )
        paraphrase_bodies = (
            f"Take {start} as the initial value. In sequence, {compact}. Report the final integer.",
            f"A running total starts at {start}. Perform, from first to last: {compact}. What is the ending total?",
            f"Evaluate this ordered calculation starting from {start}: {compact}. Give only the resulting integer in the required final line.",
        )
        items.append(
            ControlledItem(
                qid=f"rat-v1-{family_id}-L{level}",
                family_id=family_id,
                template="arithmetic_chain",
                split=split,
                level=level,
                prompt=_with_contract(body),
                gold_answer=str(answer),
                paraphrases=tuple(_with_contract(text) for text in paraphrase_bodies),
                solution_steps=operation_count,
                structure={
                    "start": start,
                    "operations": prefix,
                    "n_entities": 1,
                    "n_relations": operation_count,
                },
            )
        )
    return items


def _shortest_distance(n_nodes: int, edges: list[dict[str, int]]) -> int:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for edge in edges:
        left, right, weight = edge["left"], edge["right"], edge["weight"]
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    distances = [10**9] * n_nodes
    distances[0] = 0
    queue = [(0, 0)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distances[-1]


def _route_family(
    seed: int,
    family_index: int,
    split: str,
    levels: int,
) -> list[ControlledItem]:
    rng = _rng(seed, "shortest_path", family_index)
    max_nodes = LEVEL_SIZES["shortest_path"][-1]
    edges: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    backbone_distances = [0]

    def add_edge(left: int, right: int, weight: int) -> None:
        pair = tuple(sorted((left, right)))
        if pair not in seen:
            seen.add(pair)
            edges.append({"left": pair[0], "right": pair[1], "weight": weight})

    for node in range(1, max_nodes):
        backbone_weight = rng.randint(3, 9)
        add_edge(node - 1, node, backbone_weight)
        backbone_distances.append(backbone_distances[-1] + backbone_weight)
        if node >= 2:
            left = rng.randrange(0, node - 1)
            chain_cost = backbone_distances[node] - backbone_distances[left]
            add_edge(left, node, chain_cost + rng.randint(2, 7))
        if node >= 4:
            left = rng.randrange(0, node - 2)
            chain_cost = backbone_distances[node - 1] - backbone_distances[left]
            add_edge(left, node - 1, chain_cost + rng.randint(2, 7))

    labels = [chr(ord("A") + index) for index in range(max_nodes)]
    family_id = f"route-{family_index + 1:03d}"
    items: list[ControlledItem] = []
    for level, n_nodes in enumerate(LEVEL_SIZES["shortest_path"], start=1):
        prefix = [
            edge for edge in edges if edge["left"] < n_nodes and edge["right"] < n_nodes
        ]
        answer = _shortest_distance(n_nodes, prefix)
        roads = [
            f"{labels[edge['left']]}-{labels[edge['right']]}: {edge['weight']}"
            for edge in prefix
        ]
        road_lines = "\n".join(f"- {road}" for road in roads)
        road_compact = "; ".join(roads)
        target = labels[n_nodes - 1]
        body = (
            "The following roads are undirected, and each number is the travel cost:\n"
            f"{road_lines}\nWhat is the minimum total cost from A to {target}?"
        )
        paraphrase_bodies = (
            f"In an undirected network, the weighted roads are {road_compact}. Find the least possible travel cost from A to {target}.",
            f"Every listed connection can be used in either direction: {road_compact}. How small can the total cost of a route from A to {target} be?",
            f"Compute the shortest-path distance from A to {target} using these undirected weighted edges: {road_compact}.",
        )
        items.append(
            ControlledItem(
                qid=f"rat-v1-{family_id}-L{level}",
                family_id=family_id,
                template="shortest_path",
                split=split,
                level=level,
                prompt=_with_contract(body),
                gold_answer=str(answer),
                paraphrases=tuple(_with_contract(text) for text in paraphrase_bodies),
                solution_steps=n_nodes - 1,
                structure={
                    "n_nodes": n_nodes,
                    "edges": prefix,
                    "source": 0,
                    "target": n_nodes - 1,
                    "n_entities": n_nodes,
                    "n_relations": len(prefix),
                },
            )
        )
    return items


def _schedule_finish(tasks: list[dict[str, Any]]) -> tuple[int, list[int]]:
    finishes: list[int] = []
    for task in tasks:
        start = max((finishes[index] for index in task["prerequisites"]), default=0)
        finishes.append(start + int(task["duration"]))
    return max(finishes), finishes


def _schedule_family(
    seed: int,
    family_index: int,
    split: str,
    levels: int,
) -> list[ControlledItem]:
    rng = _rng(seed, "precedence_schedule", family_index)
    task_counts = LEVEL_SIZES["precedence_schedule"]
    max_tasks = task_counts[-1]
    level_boundaries = {count - 1 for count in task_counts}
    tasks: list[dict[str, Any]] = []
    finishes: list[int] = []
    for index in range(max_tasks):
        duration = rng.randint(1, 8)
        if index == 0:
            prerequisites: list[int] = []
        elif index in level_boundaries:
            critical = max(range(index), key=lambda task_index: finishes[task_index])
            prerequisites = [critical]
            if index >= 4:
                candidate = rng.randrange(0, index)
                if candidate not in prerequisites:
                    prerequisites.append(candidate)
                    prerequisites.sort()
        else:
            prerequisite_count = rng.randint(0, min(2, index))
            prerequisites = sorted(rng.sample(range(index), prerequisite_count))
        task = {"duration": duration, "prerequisites": prerequisites}
        tasks.append(task)
        start = max((finishes[item] for item in prerequisites), default=0)
        finishes.append(start + duration)

    labels = [chr(ord("A") + index) for index in range(max_tasks)]
    family_id = f"schedule-{family_index + 1:03d}"
    items: list[ControlledItem] = []
    for level, n_tasks in enumerate(task_counts, start=1):
        prefix = tasks[:n_tasks]
        answer, _ = _schedule_finish(prefix)
        descriptions: list[str] = []
        for index, task in enumerate(prefix):
            prerequisites = task["prerequisites"]
            duration_unit = "hour" if task["duration"] == 1 else "hours"
            if prerequisites:
                dependencies = ", ".join(labels[item] for item in prerequisites)
                description = (
                    f"Task {labels[index]} takes {task['duration']} {duration_unit} and starts "
                    f"after {dependencies}."
                )
            else:
                description = (
                    f"Task {labels[index]} takes {task['duration']} {duration_unit} and has no "
                    "prerequisites."
                )
            descriptions.append(description)
        lines = "\n".join(f"- {description}" for description in descriptions)
        compact = " ".join(descriptions)
        body = (
            "Unlimited workers are available, but every prerequisite must finish before "
            f"a task starts.\n{lines}\nWhat is the minimum number of hours needed to finish all tasks?"
        )
        paraphrase_bodies = (
            f"Assume any number of tasks may run in parallel once their prerequisites are complete. {compact} Find the earliest time when every task is finished.",
            f"Schedule these jobs with unlimited parallel capacity and the stated dependencies: {compact} What is the minimum makespan in hours?",
            f"Each task begins as soon as all of its prerequisite tasks are done, with unlimited workers. {compact} Determine the total completion time.",
        )
        items.append(
            ControlledItem(
                qid=f"rat-v1-{family_id}-L{level}",
                family_id=family_id,
                template="precedence_schedule",
                split=split,
                level=level,
                prompt=_with_contract(body),
                gold_answer=str(answer),
                paraphrases=tuple(_with_contract(text) for text in paraphrase_bodies),
                solution_steps=n_tasks,
                structure={
                    "tasks": prefix,
                    "n_entities": n_tasks,
                    "n_relations": sum(
                        len(task["prerequisites"]) for task in prefix
                    ),
                },
            )
        )
    return items


def generate_controlled_bank(
    seed: int = 20260926,
    families_per_template: int = 20,
    levels: int = 6,
) -> dict[str, Any]:
    """Generate the complete deterministic RAT-v1 item bank."""
    if levels != 6:
        raise ValueError("RAT-v1 freezes exactly six difficulty levels")
    assignments = _split_assignments(seed, families_per_template)
    builders = {
        "arithmetic_chain": _arithmetic_family,
        "shortest_path": _route_family,
        "precedence_schedule": _schedule_family,
    }
    items: list[ControlledItem] = []
    for template in TEMPLATES:
        for family_index in range(families_per_template):
            split = assignments[f"{template}:{family_index}"]
            items.extend(
                builders[template](seed, family_index, split, levels)
            )
    return {
        "protocol_id": PROTOCOL_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "families_per_template": families_per_template,
        "levels": levels,
        "templates": list(TEMPLATES),
        "level_sizes": {name: list(sizes) for name, sizes in LEVEL_SIZES.items()},
        "split_family_counts_per_template": SPLIT_FAMILY_COUNTS,
        "output_contract": OUTPUT_CONTRACT,
        "items": [item.to_dict() for item in items],
    }


def canonical_bank_text(bank: dict[str, Any]) -> str:
    return json.dumps(bank, indent=2, ensure_ascii=False) + "\n"


def bank_sha256(bank: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bank_text(bank).encode("utf-8")).hexdigest()


def write_controlled_bank(path: str | Path, bank: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_bank_text(bank))


def item_gold_answer(item: dict[str, Any]) -> str:
    template = str(item["template"])
    structure = item["structure"]
    if template == "arithmetic_chain":
        value = int(structure["start"])
        for operation in structure["operations"]:
            value = _apply_arithmetic(value, operation)
        return str(value)
    if template == "shortest_path":
        return str(_shortest_distance(int(structure["n_nodes"]), structure["edges"]))
    if template == "precedence_schedule":
        return str(_schedule_finish(structure["tasks"])[0])
    raise ValueError(f"unknown controlled template: {template}")


def bank_queries(bank: dict[str, Any], split: str) -> list[Query]:
    """Convert one frozen family split to monitor-compatible queries."""
    selected = [item for item in bank["items"] if item["split"] == split]
    selected.sort(key=lambda item: (item["template"], item["family_id"], item["level"]))
    levels = int(bank["levels"])
    return [
        Query(
            qid=str(item["qid"]),
            t=index,
            dataset="controlled_trajectory",
            domain=str(item["template"]),
            prompt=str(item["prompt"]),
            gold_answer=str(item["gold_answer"]),
            difficulty=f"level_{item['level']}",
            paraphrases=tuple(item["paraphrases"]),
            solution_steps=int(item["solution_steps"]),
            difficulty_rating=float(item["level"]),
            difficulty_quantile=(float(item["level"]) - 0.5) / levels,
            family_id=str(item["family_id"]),
            trajectory_level=int(item["level"]),
            trajectory_split=split,
        )
        for index, item in enumerate(selected, start=1)
    ]
