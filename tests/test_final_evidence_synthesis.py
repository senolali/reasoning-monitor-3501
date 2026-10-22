import hashlib
from pathlib import Path

from reasoning_monitor.evaluation.final_evidence_synthesis import (
    synthesize_final_evidence,
    verify_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def test_section69_synthesis_reproduces_frozen_decisions(tmp_path):
    result = synthesize_final_evidence(
        ROOT,
        ROOT / "configs" / "final_evidence_sources.yaml",
        tmp_path,
    )

    assert result["synthesis_valid"] is True
    assert result["no_model_calls"] is True
    assert result["hypotheses"]["h1_non_cq_metrics_precede_cq"]["status"] == (
        "not_supported"
    )
    assert result["hypotheses"]["cq_tracks_difficulty_change"]["status"] == (
        "supported"
    )
    assert result["hypotheses"]["cs_ss_cross_model_generalizability"]["status"] == (
        "not_evaluable"
    )
    assert result["campaign_decision"]["full_k5_l3_campaign_authorized"] is False
    assert result["stop_rule"]["section70_authorized"] is False
    assert (tmp_path / "assessment.json").is_file()
    assert (tmp_path / "hypothesis_decisions.csv").is_file()
    assert (tmp_path / "FINAL_EVIDENCE_SYNTHESIS.md").is_file()


def test_source_verification_rejects_hash_mismatch(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}', encoding="utf-8")
    manifest = {
        "sources": [
            {
                "id": "section49_manifest",
                "path": "source.json",
                "sha256": hashlib.sha256(b"different").hexdigest(),
            }
        ]
    }

    documents, verification, problems = verify_sources(tmp_path, manifest)

    assert documents == {}
    assert verification[0]["matches"] is False
    assert any("hash mismatch" in problem for problem in problems)
