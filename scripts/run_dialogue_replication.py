"""Run the frozen CDR-v1.4 replication with live checkpoint progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_experiment import (  # noqa: E402
    load_dialogue_checkpoint,
    records_to_turn_frame,
)
from reasoning_monitor.dialogue_replication import run_replication_model  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlled_dialogue_v1_4_replication_asu.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    if _sha256(bank_path) != config["bank_file_sha256"]:
        raise ValueError("dialogue bank hash differs from replication config")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        str(spec["name"]): output / f"records_CDR14_{spec['name']}.jsonl"
        for spec in config["models"]
    }
    completed = sum(len(load_dialogue_checkpoint(path)) for path in paths.values())
    reporter = ProgressReporter(
        total=int(config["expected_total_calls"]),
        label="CDR-v1.4",
        completed=completed,
    )
    reporter.print(completed)
    manifests = []

    def on_checkpoint(_record: dict) -> None:
        nonlocal completed
        completed += 1
        reporter.print(completed)

    for model_spec in config["models"]:
        model_name = str(model_spec["name"])
        client = build_client(model_spec, seed=int(config["seed"]))
        record_path = paths[model_name]
        records = run_replication_model(
            bank=bank,
            client=client,
            model_name=model_name,
            selected_dialogue_ids=config["selected_dialogue_ids"],
            conditions=config["conditions"],
            system_prompt=str(config["system_prompt"]),
            checkpoint_path=record_path,
            seed=int(config["seed"]),
            checkpoint_callback=on_checkpoint,
        )
        frame = records_to_turn_frame(records)
        frame.to_csv(output / f"per_turn_CDR14_{model_name}.csv", index=False)
        manifests.append(
            {
                "model": model_name,
                "records": len(records),
                "primary_turns": len(frame),
                "record_path": str(record_path.relative_to(ROOT)),
                "record_sha256": _sha256(record_path),
            }
        )
    reporter.print(completed, final=True)
    manifest = {
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "bank_path": config["bank_path"],
        "bank_file_sha256": config["bank_file_sha256"],
        "expected_total_calls": int(config["expected_total_calls"]),
        "models": manifests,
    }
    manifest_path = output / "replication_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

