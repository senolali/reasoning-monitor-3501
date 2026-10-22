"""Audit the pinned DBQR practice archive without model calls or pickle loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.dbqr_audit import (  # noqa: E402
    MAX_BYTES,
    SOURCE_SHA256,
    SOURCE_URL,
    inspect_archive,
    inspect_tables,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download public practice data only")
    parser.add_argument("--write", action="store_true", help="Write reproducible audit report")
    parser.add_argument("--safe-tables", action="store_true",
                        help="Statically check and restricted-load supplied tables")
    args = parser.parse_args(argv)
    source = ROOT / "data/dbqr/practice.tar.gz"
    if not source.exists():
        if not args.fetch:
            parser.error("Practice archive missing; use --fetch once")
        print("Downloading pinned public practice archive (no model calls)...", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            blob = response.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES or hashlib.sha256(blob).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded source differs from pinned archive")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(blob)
    print("Inspecting JSON and table-file presence; pickle execution disabled...", flush=True)
    report = inspect_archive(source.read_bytes())
    table_report = inspect_tables(source.read_bytes()) if args.safe_tables else None
    if args.write:
        output = ROOT / "results/dbqr_v1_audit"
        output.mkdir(parents=True, exist_ok=True)
        (output / "assessment.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8")
        if table_report:
            (output / "table_assessment.json").write_text(
                json.dumps(table_report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    if table_report:
        print(json.dumps(table_report, indent=2), flush=True)
    return 0 if report["structural_pass"] and (
        table_report is None or table_report["restricted_table_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
