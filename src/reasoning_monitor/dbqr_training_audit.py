"""Call-free DBQR training inventory and raw-context feasibility audit."""

from __future__ import annotations

import hashlib
import io
import json
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from reasoning_monitor.dbqr_audit import restricted_load

TRAINING_URL = "https://dbqr-qa.s3.us-east-1.amazonaws.com/training.tar.gz"
TRAINING_SHA256 = "76031bdb5d3aaf754d5fa8e14f1768e90913c96676aa07765457a9f6bdf681d8"
TRAINING_MAX_BYTES = 10_000_000
EXPANDED_MAX_BYTES = 200_000_000


def operation_signature(program: list[str]) -> tuple[str, ...]:
    """Return operation names only, excluding entities and numeric parameters."""
    signature = []
    for statement in program:
        match = re.search(r"(?:=\s*)?([A-Za-z_]\w*)\s*\(", statement)
        if not match:
            raise ValueError(f"Unrecognised reference statement: {statement}")
        signature.append(match.group(1))
    return tuple(signature)


def _safe_members(archive: tarfile.TarFile, root: str) -> dict[str, tarfile.TarInfo]:
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive entries")
    if sum(member.size for member in members) > EXPANDED_MAX_BYTES:
        raise ValueError("Excessive expanded archive size")
    files = {}
    for member in members:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                or not path.parts or path.parts[0] != root
                or not (member.isfile() or member.isdir())):
            raise ValueError("Unexpected archive member")
        if member.isfile():
            files[member.name] = member
    return files


def _question_rows(blob: bytes, expected_sha256: str, root: str,
                   expected_questions: int, expected_chats: int):
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > TRAINING_MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")
    rows = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        files = _safe_members(archive, root)
        for name, member in sorted(files.items()):
            path = PurePosixPath(name)
            if len(path.parts) != 4 or path.parts[1] != "questions" or path.suffix != ".json":
                continue
            row = json.load(archive.extractfile(member))
            required = {"sectionID", "conversationID", "questionID", "question",
                        "answer", "program", "queries"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f"Missing question fields: {name}")
            rows.append({"path": name, "chat": path.parts[2], "section": row["sectionID"],
                         "conversation": row["conversationID"], "turn": row["questionID"],
                         "answer_type": type(row["answer"]).__name__,
                         "operation_signature": operation_signature(row["program"])})
    chats = defaultdict(list)
    for row in rows:
        chats[row["chat"]].append(row)
    if len(rows) != expected_questions or len(chats) != expected_chats:
        raise ValueError("Unexpected question/chat count")
    if not all(sorted(row["turn"] for row in group) == list(range(1, 11))
               for group in chats.values()):
        raise ValueError("Every chat must contain turns 1..10 exactly once")
    return digest, rows, chats


def dialogue_signatures(blob: bytes, expected_sha256: str, root: str,
                        expected_questions: int, expected_chats: int) -> dict[str, tuple]:
    _, _, chats = _question_rows(blob, expected_sha256, root,
                                 expected_questions, expected_chats)
    return {chat: tuple(row["operation_signature"]
                        for row in sorted(group, key=lambda value: value["turn"]))
            for chat, group in chats.items()}


def inspect_training(training_blob: bytes, practice_blob: bytes | None = None) -> dict:
    digest, rows, chats = _question_rows(training_blob, TRAINING_SHA256, "training", 200, 20)
    turn_sizes = {}
    all_globals = set()
    with tarfile.open(fileobj=io.BytesIO(training_blob), mode="r:gz") as archive:
        files = _safe_members(archive, "training")
        for row in rows:
            question_path = PurePosixPath(row["path"])
            table_name = str(PurePosixPath("training", "tables", row["chat"],
                                           question_path.stem + ".pkl"))
            if table_name not in files:
                raise ValueError(f"Missing table: {table_name}")
            tables, globals_used = restricted_load(archive.extractfile(files[table_name]).read())
            all_globals.update(globals_used)
            rendered = "\n\n".join(f"TABLE {key}\n{frame.to_csv()}"
                                     for key, frame in tables.items())
            turn_sizes[(row["chat"], row["turn"])] = {
                "characters": len(rendered), "token_proxy": (len(rendered) + 3) // 4,
                "cells": sum(int(frame.size) for frame in tables.values()),
                "dataframes": len(tables),
            }
    dialogue_rows = []
    for chat, group in sorted(chats.items()):
        ordered = sorted(group, key=lambda value: value["turn"])
        sizes = [turn_sizes[(chat, row["turn"])] for row in ordered]
        dialogue_rows.append({
            "chat": chat, "section": ordered[0]["section"],
            "conversation": ordered[0]["conversation"],
            "total_token_proxy": sum(value["token_proxy"] for value in sizes),
            "maximum_turn_token_proxy": max(value["token_proxy"] for value in sizes),
            "total_cells": sum(value["cells"] for value in sizes),
            "turns_over_8192": sum(value["token_proxy"] > 8192 for value in sizes),
            "turns_over_32768": sum(value["token_proxy"] > 32768 for value in sizes),
            "turns_over_65536": sum(value["token_proxy"] > 65536 for value in sizes),
        })
    sections = {}
    for section in sorted({row["section"] for row in dialogue_rows}):
        group = [row for row in dialogue_rows if row["section"] == section]
        maxima = [row["maximum_turn_token_proxy"] for row in group]
        sections[str(section)] = {
            "dialogues": len(group), "turns": len(group) * 10,
            "median_maximum_turn_token_proxy": int(statistics.median(maxima)),
            "maximum_turn_token_proxy": max(maxima),
            "dialogues_with_all_turns_at_most_8192": sum(not row["turns_over_8192"]
                                                          for row in group),
            "turns_over_8192": sum(row["turns_over_8192"] for row in group),
            "turns_over_32768": sum(row["turns_over_32768"] for row in group),
            "turns_over_65536": sum(row["turns_over_65536"] for row in group),
        }
    signatures = {chat: tuple(row["operation_signature"]
                              for row in sorted(group, key=lambda value: value["turn"]))
                  for chat, group in chats.items()}
    signature_counts = Counter(signatures.values())
    turn_signature_counts = Counter(row["operation_signature"] for row in rows)
    operation_counts = Counter(operation for row in rows
                               for operation in row["operation_signature"])
    practice_overlap = []
    if practice_blob is not None:
        from reasoning_monitor.dbqr_audit import SOURCE_SHA256

        practice = dialogue_signatures(practice_blob, SOURCE_SHA256, "practice", 50, 5)
        practice_overlap = sorted(
            {chat for chat, signature in signatures.items() if signature in set(practice.values())})
    return {
        "protocol_id": "DBQR-RQ-v1", "stage": "training_context_feasibility_audit",
        "no_model_calls": True, "execution_authorized": False,
        "source_sha256": digest, "questions": len(rows), "dialogues": len(chats),
        "answer_types": dict(sorted(Counter(row["answer_type"] for row in rows).items())),
        "restricted_pickle_globals": [f"{module}.{name}"
                                      for module, name in sorted(all_globals)],
        "sections": sections, "dialogues_detail": dialogue_rows,
        "operation_counts": dict(operation_counts.most_common()),
        "unique_turn_operation_shapes": len(turn_signature_counts),
        "repeated_turn_operation_shape_counts": sorted(
            (count for count in turn_signature_counts.values() if count > 1), reverse=True),
        "exact_operation_template_groups": sorted(signature_counts.values(), reverse=True),
        "training_dialogues_matching_a_practice_operation_template": practice_overlap,
        "raw_all_dialogue_feasibility_pass": all(not row["turns_over_8192"]
                                                   for row in dialogue_rows),
        "ready_for_model_run": False,
        "note": ("Character/4 is a transparent sizing proxy, not a tokenizer result. "
                 "Thresholds are descriptive and do not authorize excluding sections."),
    }
