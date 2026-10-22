"""Freeze LIC-RQ-v2 without making model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reasoning_monitor.lost_conversation_manifest import (  # noqa: E402
    build_manifest,
    write_artifacts,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lost_conversation_v2_manifest.yaml")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError("Config must remain inside the repository")
    artifacts = build_manifest(ROOT, config_path)
    if args.write:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output = (ROOT / config["output_dir"]).resolve()
        write_artifacts(output, artifacts)
    print(json.dumps(artifacts["assessment.json"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

