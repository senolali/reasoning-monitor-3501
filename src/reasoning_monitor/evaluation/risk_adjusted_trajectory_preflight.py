"""Call-free structural preflight for the independent RAT-v1 protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from reasoning_monitor.data.controlled_trajectories import (
    LEVEL_SIZES,
    OUTPUT_CONTRACT,
    TEMPLATES,
    bank_queries,
    canonical_bank_text,
    generate_controlled_bank,
    item_gold_answer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family_nesting_ok(items: list[dict[str, Any]]) -> bool:
    ordered = sorted(items, key=lambda item: int(item["level"]))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if int(current["level"]) != int(previous["level"]) + 1:
            return False
        template = str(current["template"])
        old = previous["structure"]
        new = current["structure"]
        if template == "arithmetic_chain":
            if (
                len(new["operations"]) <= len(old["operations"])
                or new["operations"][: len(old["operations"])] != old["operations"]
            ):
                return False
        elif template == "shortest_path":
            old_edges = {
                (edge["left"], edge["right"], edge["weight"])
                for edge in old["edges"]
            }
            new_edges = {
                (edge["left"], edge["right"], edge["weight"])
                for edge in new["edges"]
            }
            if int(new["n_nodes"]) <= int(old["n_nodes"]) or not old_edges <= new_edges:
                return False
        elif template == "precedence_schedule":
            if (
                len(new["tasks"]) <= len(old["tasks"])
                or new["tasks"][: len(old["tasks"])] != old["tasks"]
            ):
                return False
            if int(current["gold_answer"]) <= int(previous["gold_answer"]):
                return False
        else:
            return False
    return True


def assess_controlled_bank(bank: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    items = list(bank.get("items") or [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("family_id"))].append(item)

    qids = [str(item.get("qid")) for item in items]
    family_splits = {
        family_id: {str(item.get("split")) for item in rows}
        for family_id, rows in grouped.items()
    }
    template_families = {
        template: {
            family_id
            for family_id, rows in grouped.items()
            if rows and rows[0].get("template") == template
        }
        for template in TEMPLATES
    }
    split_families = Counter(
        next(iter(splits))
        for splits in family_splits.values()
        if len(splits) == 1
    )
    split_items = Counter(str(item.get("split")) for item in items)
    prompt_contract_ok = all(
        str(item.get("prompt", "")).endswith(OUTPUT_CONTRACT)
        and all(
            str(paraphrase).endswith(OUTPUT_CONTRACT)
            for paraphrase in item.get("paraphrases", [])
        )
        for item in items
    )
    paraphrases_ok = all(
        len(item.get("paraphrases", [])) == int(gate["paraphrases_per_item"])
        and len(set(item.get("paraphrases", []))) == int(gate["paraphrases_per_item"])
        and item.get("prompt") not in item.get("paraphrases", [])
        for item in items
    )
    gold_ok = all(str(item.get("gold_answer")) == item_gold_answer(item) for item in items)
    family_levels_ok = all(
        sorted(int(item["level"]) for item in rows)
        == list(range(1, int(gate["levels"]) + 1))
        for rows in grouped.values()
    )
    nesting_ok = all(_family_nesting_ok(rows) for rows in grouped.values())
    complexity_profiles_ok = all(
        [
            int(item["structure"]["n_relations"])
            if template == "arithmetic_chain"
            else int(item["structure"]["n_entities"])
            for item in sorted(rows, key=lambda row: int(row["level"]))
        ]
        == list(LEVEL_SIZES[template])
        for template in TEMPLATES
        for rows in [
            family_rows
            for family_rows in grouped.values()
            if family_rows and family_rows[0].get("template") == template
        ]
    )
    queries_ok = all(
        len(bank_queries(bank, split)) == expected
        for split, expected in gate["expected_items_by_split"].items()
    )

    checks = {
        "protocol_id": bank.get("protocol_id") == gate["protocol_id"],
        "generator_version": bank.get("generator_version")
        == gate["generator_version"],
        "item_count": len(items) == int(gate["expected_items"]),
        "family_count": len(grouped) == int(gate["expected_families"]),
        "unique_qids": len(qids) == len(set(qids)),
        "templates": list(bank.get("templates") or []) == list(TEMPLATES),
        "level_sizes": bank.get("level_sizes") == gate["expected_level_sizes"],
        "families_per_template": all(
            len(families) == int(gate["families_per_template"])
            for families in template_families.values()
        ),
        "family_split_integrity": all(len(splits) == 1 for splits in family_splits.values()),
        "split_family_counts": dict(split_families)
        == gate["expected_families_by_split"],
        "split_item_counts": dict(split_items) == gate["expected_items_by_split"],
        "family_levels": family_levels_ok,
        "nested_trajectories": nesting_ok,
        "complexity_profiles": complexity_profiles_ok,
        "solver_gold_answers": gold_ok,
        "prompt_contract": prompt_contract_ok,
        "genuine_paraphrase_surfaces": paraphrases_ok,
        "monitor_query_conversion": queries_ok,
    }
    return {
        "n_items": len(items),
        "n_families": len(grouped),
        "template_family_counts": {
            template: len(families) for template, families in template_families.items()
        },
        "split_family_counts": dict(split_families),
        "split_item_counts": dict(split_items),
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_preflight(
    root: str | Path,
    protocol_path: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    protocol_path = Path(protocol_path).resolve()
    gate_path = Path(gate_path).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    bank_path = (root / protocol["benchmark"]["bank_path"]).resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    regenerated = generate_controlled_bank(
        seed=int(protocol["benchmark"]["seed"]),
        families_per_template=int(protocol["benchmark"]["families_per_template"]),
        levels=int(protocol["benchmark"]["levels"]),
    )
    problems: list[str] = []
    bank_hash = _sha256(bank_path)
    if bank_hash != gate["expected_bank_sha256"]:
        problems.append("bank hash differs from the frozen gate")
    if canonical_bank_text(bank) != canonical_bank_text(regenerated):
        problems.append("saved bank differs from deterministic regeneration")
    protocol_hash = _sha256(protocol_path)
    if protocol_hash != gate["expected_protocol_sha256"]:
        problems.append("protocol hash differs from the frozen gate")
    source_hashes = {}
    for relative_path, expected_hash in gate.get("expected_source_sha256", {}).items():
        source_path = root / relative_path
        observed_hash = _sha256(source_path)
        source_hashes[relative_path] = observed_hash
        if observed_hash != expected_hash:
            problems.append(f"frozen source hash differs: {relative_path}")

    assessment = assess_controlled_bank(bank, gate)
    problems.extend(name for name, passed in assessment["checks"].items() if not passed)
    no_paid_calls = bool(
        protocol.get("status") == "DESIGN_ONLY_NO_MODEL_CALLS"
        and protocol["execution"]["paid_calls_authorized"] is False
    )
    if not no_paid_calls:
        problems.append("protocol does not explicitly block paid calls")
    return {
        "analysis": "rat_v1_structural_preflight",
        "protocol_id": protocol.get("protocol_id"),
        "no_model_calls": True,
        "bank_path": bank_path.relative_to(root).as_posix(),
        "bank_sha256": bank_hash,
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hashes,
        **assessment,
        "paid_calls_authorized": False,
        "problems": problems,
        "pass": assessment["pass"] and no_paid_calls and not problems,
    }
