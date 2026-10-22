"""Run the 72-call CDR-v1.2 user-only history mechanism arm."""

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

from reasoning_monitor.dialogue_experiment import records_to_turn_frame  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.user_only_dialogue_experiment import (  # noqa: E402
    run_user_only_dialogue_model,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlled_dialogue_v1_2_user_only_probe_asu.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    if _sha256(bank_path) != config["bank_file_sha256"]:
        raise ValueError("dialogue bank hash differs from user-only config")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"] / "data"
    output.mkdir(parents=True, exist_ok=True)
    manifests = []

    for model_spec in config["models"]:
        model_name = str(model_spec["name"])
        client = build_client(model_spec, seed=int(config["seed"]))
        record_path = output / f"records_CDR12_user_only_{model_name}.jsonl"
        records = run_user_only_dialogue_model(
            bank=bank,
            client=client,
            model_name=model_name,
            selected_dialogue_ids=config["selected_dialogue_ids"],
            system_prompt=str(config["system_prompt"]),
            checkpoint_path=record_path,
            seed=int(config["seed"]),
        )
        frame = records_to_turn_frame(records)
        frame.to_csv(output / f"per_turn_CDR12_user_only_{model_name}.csv", index=False)
        manifests.append(
            {
                "model": model_name,
                "records": len(records),
                "record_path": str(record_path.relative_to(ROOT)),
                "record_sha256": _sha256(record_path),
            }
        )
    manifest = {
        "protocol_id": config["protocol_id"],
        "condition": config["condition"],
        "history_transport": config["history_transport"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "expected_total_calls": int(config["expected_total_calls"]),
        "models": manifests,
    }
    manifest_path = output / "user_only_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

