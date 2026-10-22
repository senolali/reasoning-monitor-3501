from __future__ import annotations

import json

from reasoning_monitor.data.dialogue_streams import (
    TEMPLATES,
    dialogue_bank_sha256,
    generate_dialogue_bank,
    solve_dialogue_state,
    validate_dialogue_bank,
    write_dialogue_bank,
)


def test_dialogue_bank_is_deterministic_exact_and_family_split(tmp_path):
    first = generate_dialogue_bank()
    second = generate_dialogue_bank()

    assert dialogue_bank_sha256(first) == dialogue_bank_sha256(second)
    assert validate_dialogue_bank(first) == []
    assert len(first["dialogues"]) == 30
    assert {dialogue["template"] for dialogue in first["dialogues"]} == set(TEMPLATES)
    assert sum(len(dialogue["turns"]) for dialogue in first["dialogues"]) == 360
    assert all(
        str(solve_dialogue_state(dialogue["template"], turn["state_after"]))
        == turn["gold_answer"]
        for dialogue in first["dialogues"]
        for turn in dialogue["turns"]
    )

    path = tmp_path / "bank.json"
    write_dialogue_bank(path, first)
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_dialogue_bank_has_three_disjoint_family_level_splits():
    bank = generate_dialogue_bank()
    counts = {"calibration": 0, "development": 0, "holdout": 0}
    for dialogue in bank["dialogues"]:
        counts[dialogue["split"]] += 1
        assert {turn["split"] for turn in dialogue["turns"]} == {dialogue["split"]}

    assert counts == {"calibration": 12, "development": 9, "holdout": 9}
