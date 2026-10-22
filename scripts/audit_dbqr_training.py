"""Download and audit pinned DBQR training data without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.dbqr_training_audit import (  # noqa: E402
    TRAINING_MAX_BYTES,
    TRAINING_SHA256,
    TRAINING_URL,
    inspect_training,
)


def _load_or_fetch(path: Path, fetch: bool) -> bytes:
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"{path} missing; rerun with --fetch")
        print("Downloading pinned public training archive (no model calls)...", flush=True)
        with urllib.request.urlopen(TRAINING_URL, timeout=90) as response:
            blob = response.read(TRAINING_MAX_BYTES + 1)
        if len(blob) > TRAINING_MAX_BYTES or hashlib.sha256(blob).hexdigest() != TRAINING_SHA256:
            raise ValueError("Downloaded training archive differs from pinned source")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return path.read_bytes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    training = _load_or_fetch(ROOT / "data/dbqr/training.tar.gz", args.fetch)
    practice_path = ROOT / "data/dbqr/practice.tar.gz"
    report = inspect_training(training, practice_path.read_bytes() if practice_path.exists() else None)
    if args.write:
        output = ROOT / "results/dbqr_v1_audit"
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_assessment.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "dialogues_detail"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
