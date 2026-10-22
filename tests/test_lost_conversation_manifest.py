import json
from pathlib import Path

import pytest

from reasoning_monitor.lost_conversation_manifest import (
    build_manifest,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "lost_conversation_v2_manifest.yaml"


def test_frozen_manifest_uses_three_models_disjoint_samples_and_no_calls():
    artifacts = build_manifest(ROOT, CONFIG)
    manifest = artifacts["execution_manifest.json"]
    assert manifest["no_model_calls"] is True
    assert manifest["execution_authorized"] is False
    assert manifest["execution_ready"] is False
    assert [row["name"] for row in manifest["models"]] == [
        "gpt4o_mini", "llama4_scout", "mistral_7b"
    ]
    assert len(manifest["integration_task_ids"]) == 4
    assert len(manifest["science_task_ids"]) == 48
    assert not set(manifest["integration_task_ids"]) & set(manifest["science_task_ids"])
    assert manifest["integration_is_effect_gate"] is False
    assert manifest["science_may_start_without_new_explicit_approval"] is False
    assert len(manifest["pending_before_integration_approval"]) == 5


def test_budget_uses_all_turns_models_conditions_and_k_l():
    manifest = build_manifest(ROOT, CONFIG)["execution_manifest.json"]
    for name in ("integration_budget", "science_budget", "total_budget"):
        budget = manifest[name]
        assert budget["logical_model_calls"] == budget["prefixes_per_model_condition"] * 3 * 2 * 7
        assert budget["maximum_output_tokens"] == budget["logical_model_calls"] * 2048


def test_pre_final_gold_dependent_metrics_remain_missing():
    observability = build_manifest(ROOT, CONFIG)["execution_manifest.json"][
        "metric_observability"
    ]["pre_final"]
    assert observability["CQ"].startswith("missing")
    assert observability["RS"].startswith("missing")
    assert observability["ES"].startswith("missing")
    assert observability["RQ6_mean"].startswith("missing")


def test_artifacts_are_immutable(tmp_path):
    artifacts = build_manifest(ROOT, CONFIG)
    write_artifacts(tmp_path, artifacts)
    write_artifacts(tmp_path, artifacts)
    (tmp_path / "assessment.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        write_artifacts(tmp_path, artifacts)
    assert json.loads((tmp_path / "execution_manifest.json").read_text())["protocol_id"] == "LIC-RQ-v2"
