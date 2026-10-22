import copy
import hashlib
import json

import pytest

from reasoning_monitor.dbqr_prompt_manifest import validate_prompt_manifest


def _manifest():
    body = {
        "protocol_id": "DBQR-RQ-v1",
        "split": "training",
        "source_sha256": "source",
        "label_blind": True,
        "cases": [{
            "dialogue_id": "chat-1",
            "turn": 1,
            "qid": "chat-1/q1",
            "question": "Question?",
            "schema": {"schema_version": "test"},
            "table_member": "training/tables/chat-1/q1.pkl",
        }],
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def test_label_blind_manifest_validates():
    validate_prompt_manifest(_manifest())


def test_gold_or_program_field_is_rejected_even_with_recomputed_hash():
    manifest = _manifest()
    manifest["cases"][0]["answer"] = 42
    unsigned = copy.deepcopy(manifest)
    unsigned.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="forbidden"):
        validate_prompt_manifest(manifest)


def test_manifest_tampering_is_rejected():
    manifest = _manifest()
    manifest["cases"][0]["question"] = "Changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_prompt_manifest(manifest)
