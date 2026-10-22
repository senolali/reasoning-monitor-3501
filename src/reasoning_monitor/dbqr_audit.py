"""Call-free, practice-only DBQR inventory. Never deserialize supplied pickles."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import pickle
import pickletools
import tarfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

SOURCE_URL = "https://dbqr-qa.s3.us-east-1.amazonaws.com/practice.tar.gz"
SOURCE_SHA256 = "5a238e756f04b7796f16938fff3f1154c2401c8051a9ff35556e2574ffe91c1f"
MAX_BYTES = 20_000_000
ALLOWED_PICKLE_GLOBALS = {
    ("builtins", "slice"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
}


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def pickle_globals(raw: bytes) -> set[tuple[str, str]]:
    """Resolve protocol-4 GLOBAL references without executing the pickle."""
    mark = object()
    opaque = object()
    stack: list = []
    memo: dict[int, object] = {}
    found = set()

    def marked():
        index = len(stack) - 1
        while index >= 0 and stack[index] is not mark:
            index -= 1
        if index < 0:
            raise ValueError("Malformed pickle stack")
        values = stack[index + 1:]
        del stack[index:]
        return values

    scalars = {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8",
               "STRING", "SHORT_BINSTRING", "BINSTRING", "SHORT_BINBYTES",
               "BINBYTES", "BINBYTES8", "BYTEARRAY8", "BININT", "BININT1",
               "BININT2", "INT", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT"}
    for opcode, argument, position in pickletools.genops(raw):
        name = opcode.name
        if name in {"PROTO", "FRAME"}:
            continue
        if name == "MARK":
            stack.append(mark)
        elif name == "NONE":
            stack.append(None)
        elif name in {"NEWTRUE", "NEWFALSE"}:
            stack.append(name == "NEWTRUE")
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name in scalars:
            stack.append(argument)
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[int(argument)] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET", "GET"}:
            stack.append(memo[int(argument)])
        elif name == "STACK_GLOBAL":
            global_name, module = stack.pop(), stack.pop()
            if not isinstance(module, str) or not isinstance(global_name, str):
                raise ValueError(f"Unresolved STACK_GLOBAL at {position}")
            found.add((module, global_name))
            stack.append(opaque)
        elif name == "GLOBAL":
            module, global_name = argument.split(" ", 1)
            found.add((module, global_name))
            stack.append(opaque)
        elif name == "TUPLE":
            stack.append(tuple(marked()))
        elif name == "TUPLE1":
            stack.append((stack.pop(),))
        elif name in {"TUPLE2", "TUPLE3"}:
            length = int(name[-1])
            values = stack[-length:]
            del stack[-length:]
            stack.append(tuple(values))
        elif name == "APPEND":
            value = stack.pop()
            if isinstance(stack[-1], list):
                stack[-1].append(value)
        elif name == "APPENDS":
            values = marked()
            if isinstance(stack[-1], list):
                stack[-1].extend(values)
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            if isinstance(stack[-1], dict):
                stack[-1][key] = value
        elif name == "SETITEMS":
            values = marked()
            if isinstance(stack[-1], dict):
                for index in range(0, len(values), 2):
                    stack[-1][values[index]] = values[index + 1]
        elif name in {"REDUCE", "NEWOBJ"}:
            stack.pop()
            stack.pop()
            stack.append(opaque)
        elif name == "BUILD":
            stack.pop()
        elif name != "STOP":
            raise ValueError(f"Unsupported pickle opcode {name} at {position}")
    return found


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"Forbidden global: {module}.{name}")
        return getattr(importlib.import_module(module), name)


def restricted_load(raw: bytes):
    globals_used = pickle_globals(raw)
    forbidden = globals_used - ALLOWED_PICKLE_GLOBALS
    if forbidden:
        raise pickle.UnpicklingError(f"Forbidden globals: {sorted(forbidden)}")
    return _RestrictedUnpickler(io.BytesIO(raw)).load(), globals_used


def inspect_tables(blob: bytes, expected_sha256: str = SOURCE_SHA256) -> dict:
    """Inspect pinned tables after static opcode and restricted-global checks."""
    import pandas as pd

    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")
    rows = []
    all_globals = set()
    dialogue_sizes = defaultdict(lambda: {"characters": 0, "maximum_turn_characters": 0,
                                          "cells": 0, "dataframes": 0})
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if not member.isfile() or len(path.parts) != 4 or path.parts[1] != "tables":
                continue
            raw = archive.extractfile(member).read()
            tables, globals_used = restricted_load(raw)
            all_globals.update(globals_used)
            if not isinstance(tables, dict) or not tables:
                raise ValueError(f"Expected a non-empty table dictionary: {member.name}")
            rendered = "\n\n".join(f"TABLE {key}\n{frame.to_csv()}"
                                     for key, frame in tables.items())
            size = dialogue_sizes[path.parts[2]]
            size["characters"] += len(rendered)
            size["maximum_turn_characters"] = max(size["maximum_turn_characters"],
                                                   len(rendered))
            for key, frame in tables.items():
                if not isinstance(key, str) or not isinstance(frame, pd.DataFrame):
                    raise ValueError(f"Unexpected table payload: {member.name}")
                cells = int(frame.shape[0] * frame.shape[1])
                size["cells"] += cells
                size["dataframes"] += 1
                rows.append({"source_path": member.name, "table_key": key,
                             "rows": int(frame.shape[0]), "columns": int(frame.shape[1]),
                             "cells": cells, "index_type": type(frame.index).__name__,
                             "column_type": type(frame.columns).__name__,
                             "dtypes": sorted({str(value) for value in frame.dtypes})})
    checks = {
        "expected_pickle_files": len({r["source_path"] for r in rows}) == 50,
        "only_allowed_globals": all_globals <= ALLOWED_PICKLE_GLOBALS,
        "nonempty_frames": bool(rows) and all(r["rows"] and r["columns"] for r in rows),
    }
    biggest = sorted(rows, key=lambda row: row["cells"], reverse=True)[:10]
    for size in dialogue_sizes.values():
        size["character_div_4_token_proxy"] = (size["characters"] + 3) // 4
        size["maximum_turn_token_proxy"] = (size["maximum_turn_characters"] + 3) // 4
    return {
        "protocol_id": "DBQR-RQ-v1", "stage": "practice_restricted_table_audit",
        "no_model_calls": True, "execution_authorized": False,
        "source_sha256": digest, "pickle_files": len({r["source_path"] for r in rows}),
        "dataframes": len(rows), "total_rows": sum(r["rows"] for r in rows),
        "total_cells": sum(r["cells"] for r in rows),
        "maximum_rows": max(r["rows"] for r in rows),
        "maximum_cells": max(r["cells"] for r in rows),
        "globals": [f"{module}.{name}" for module, name in sorted(all_globals)],
        "checks": checks, "restricted_table_pass": all(checks.values()),
        "raw_prompt_feasibility_pass": False,
        "raw_prompt_feasibility_reason": (
            "Some per-turn query results contain thousands of rows. Token budgets and a "
            "prospectively frozen, non-label-leaking representation require separate review."),
        "largest_tables": biggest,
        "dialogue_size_proxy": dict(sorted(dialogue_sizes.items())),
    }


def inspect_archive(blob: bytes, expected_sha256: str = SOURCE_SHA256) -> dict:
    """Inspect JSON and table-file presence; this does NOT validate table contents."""
    digest = hashlib.sha256(blob).hexdigest()
    if len(blob) > MAX_BYTES or digest != expected_sha256:
        raise ValueError("Source size/hash mismatch; do not silently accept a new version")
    rows = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or sum(m.size for m in members) > MAX_BYTES:
            raise ValueError("Duplicate archive entries or excessive expanded size")
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or not path.parts or path.parts[0] != "practice"
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unexpected archive member")
        files = {m.name: m for m in members if m.isfile()}
        for name, member in sorted(files.items()):
            path = PurePosixPath(name)
            if len(path.parts) != 4 or path.parts[1] != "questions" or path.suffix != ".json":
                continue
            row = json.load(archive.extractfile(member), parse_constant=_reject_constant)
            required = {"sectionID", "conversationID", "questionID", "question",
                        "answer", "program", "queries"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f"Missing question fields: {name}")
            if any(type(row[k]) is not int for k in ("sectionID", "conversationID", "questionID")):
                raise ValueError(f"Invalid identifiers: {name}")
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"Empty question: {name}")
            if (not isinstance(row["program"], list) or not row["program"]
                    or not all(isinstance(v, str) for v in row["program"])
                    or not isinstance(row["queries"], dict) or not _finite(row["answer"])
                    or row["answer"] is None or isinstance(row["answer"], bool)):
                raise ValueError(f"Invalid reference fields: {name}")
            table = str(PurePosixPath("practice", "tables", path.parts[2], path.stem + ".pkl"))
            rows.append({"source_path": name, "chat": path.parts[2],
                         "section_id": row["sectionID"],
                         "conversation_id": row["conversationID"], "turn": row["questionID"],
                         "question": row["question"],
                         "answer_type": type(row["answer"]).__name__,
                         "reference_answer": row["answer"],
                         "reference_program": row["program"],
                         "program_statements": len(row["program"]),
                         "table_present": table in files,
                         "table_bytes": files[table].size if table in files else None,
                         "semantic_review": "pending", "gold_verified": False})
    chats = defaultdict(list)
    for row in rows:
        chats[row["chat"]].append(row)
    checks = {
        "expected_questions": len(rows) == 50,
        "expected_chats": len(chats) == 5,
        "ten_unique_ordered_turns": bool(chats) and all(
            sorted(r["turn"] for r in group) == list(range(1, 11))
            for group in chats.values()),
        "consistent_chat_identity": bool(chats) and all(
            len({(r["section_id"], r["conversation_id"]) for r in group}) == 1
            for group in chats.values()),
        "unique_question_ids": len({(r["section_id"], r["conversation_id"], r["turn"])
                                    for r in rows}) == len(rows),
        "table_files_present": bool(rows) and all(r["table_present"] for r in rows),
    }
    return {
        "protocol_id": "DBQR-RQ-v1", "stage": "practice_structural_audit",
        "no_model_calls": True, "execution_authorized": False,
        "source_url": SOURCE_URL, "source_sha256": digest,
        "questions": len(rows), "dialogues": len(chats),
        "answer_types": dict(sorted(Counter(r["answer_type"] for r in rows).items())),
        "checks": checks, "structural_pass": all(checks.values()),
        "ready_for_model_run": False,
        "pending": ["independent_gold_recalculation", "semantic_dependency_review",
                    "raw_table_representation_decision", "template_overlap_and_cluster_split",
                    "frozen_paraphrases", "prospective_protocol_and_execution_approval"],
        "metric_eligibility_not_measurements": {
            "CQ": "Each turn has a label; verifier and semantics still pending",
            "CS": "Requires isolated same-context K samples",
            "RS": "Requires reviewed paraphrases; undefined outside correct-primary subset",
            "LS": "Requires explanation steps and real NLI, not reference programs alone",
            "ES": "Window 5: turns 1-4 warmup; turns 5-10 potentially eligible",
            "SS": "Requires same-context traces and real BERTScore",
            "RQ6_mean": "All six measured and finite only; no imputation",
        },
        "es_eligible_upper_bound": sum(max(0, len(g) - 4) for g in chats.values()),
        "rows": rows,
    }
