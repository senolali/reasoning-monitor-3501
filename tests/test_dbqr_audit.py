import hashlib
import io
import json
import os
import pickle
import tarfile

import pytest

from reasoning_monitor.dbqr_audit import inspect_archive, pickle_globals, restricted_load


def fixture_archive(missing_table=False, duplicate_turn=False, unsafe=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for chat in range(1, 6):
            for turn in range(1, 11):
                row = dict(sectionID=chat, conversationID=chat,
                           questionID=1 if duplicate_turn else turn,
                           question="By how much?", answer=10.5,
                           program=["x = sum(table)"], queries={})
                for folder, suffix, content in (
                    ("questions", "json", json.dumps(row).encode()),
                    ("tables", "pkl", b"INTENTIONALLY NOT A VALID PICKLE"),
                ):
                    if missing_table and folder == "tables":
                        continue
                    member = tarfile.TarInfo(
                        f"practice/{folder}/chat-{chat}/question-{turn:02}.{suffix}")
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
        if unsafe:
            archive.addfile(tarfile.TarInfo("../outside"), io.BytesIO())
    blob = stream.getvalue()
    return blob, hashlib.sha256(blob).hexdigest()


def test_structural_pass_is_not_execution_approval():
    report = inspect_archive(*fixture_archive())
    assert report["structural_pass"]
    assert report["questions"] == 50
    assert report["es_eligible_upper_bound"] == 30
    assert not report["ready_for_model_run"]
    assert not report["execution_authorized"]
    assert all(not row["gold_verified"] for row in report["rows"])


@pytest.mark.parametrize("option", ["missing_table", "duplicate_turn"])
def test_bad_structure_does_not_pass(option):
    assert not inspect_archive(*fixture_archive(**{option: True}))["structural_pass"]


def test_hash_mismatch():
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_archive(fixture_archive()[0], "wrong")


def test_unsafe_archive_member():
    with pytest.raises(ValueError, match="Unexpected archive"):
        inspect_archive(*fixture_archive(unsafe=True))


def test_restricted_loader_accepts_plain_data_without_globals():
    raw = pickle.dumps({"safe": [1, 2, 3]}, protocol=4)
    value, globals_used = restricted_load(raw)
    assert value == {"safe": [1, 2, 3]}
    assert not globals_used


def test_restricted_loader_rejects_callable_global():
    raw = pickle.dumps(os.system, protocol=4)
    assert ("nt", "system") in pickle_globals(raw)
    with pytest.raises(pickle.UnpicklingError, match="Forbidden"):
        restricted_load(raw)
