"""Audit the pinned LIC math source without model calls. Default: no writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.lost_conversation import (  # noqa: E402
    load_math_tasks,
    prefix_metadata,
    preparation_report,
    validate_design,
)


def project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Preparation paths must remain inside this repository")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lost_conversation_v1_design.yaml")
    parser.add_argument("--write", action="store_true", help="Write call-free audit artifacts")
    args = parser.parse_args(argv)
    path = project_path(args.config)
    raw_config = path.read_bytes()
    config = yaml.safe_load(raw_config)
    validate_design(config)
    source = config["source"]
    tasks, counts = load_math_tasks(project_path(source["path"]), source["sha256"])
    report = preparation_report(tasks, counts, config)
    report.update(source=source, config_sha256=hashlib.sha256(raw_config).hexdigest())
    artifacts = {
        "assessment.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "math_inventory.json": json.dumps(
            {"protocol_id": config["protocol_id"], "pool_only": True,
             "source_sha256": source["sha256"],
             "tasks": [task.as_record() for task in tasks]}, indent=2) + "\n",
        "prefix_review.jsonl": "".join(
            json.dumps(prefix_metadata(task, turn)) + "\n"
            for task in tasks for turn in range(1, len(task.shards) + 1)),
    }
    if args.write:
        output = project_path(config["output_dir"])
        # Refuse to replace a changed protocol/report silently on a subsequent run.
        for name, content in artifacts.items():
            target = output / name
            if target.exists() and target.read_bytes() != content.encode("utf-8"):
                raise ValueError(f"Existing artifact differs; choose a new output directory: {target}")
        output.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            target = output / name
            if not target.exists():
                with target.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print("No model calls. Execution is neither implemented nor authorized by this tool.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
