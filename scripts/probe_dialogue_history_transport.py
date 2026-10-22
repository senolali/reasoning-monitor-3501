"""Run the 12-call CDR-v1.1 packed-history transport probe."""

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
    strict_final_integer,
)
from reasoning_monitor.dialogue_transport import build_packed_turn_messages  # noqa: E402
from reasoning_monitor.metrics.dimensions import answers_equal  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlled_dialogue_v1_1_transport_probe_asu.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bank_path = ROOT / config["bank_path"]
    if _sha256(bank_path) != config["bank_file_sha256"]:
        raise ValueError("dialogue bank hash differs from transport probe config")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    dialogues = {dialogue["dialogue_id"]: dialogue for dialogue in bank["dialogues"]}
    output = ROOT / config["output_dir"] / "data"
    output.mkdir(parents=True, exist_ok=True)
    manifests = []

    for model_spec in config["models"]:
        model = str(model_spec["name"])
        source_spec = config["source_v1_records"][model]
        source_path = ROOT / source_spec["path"]
        if _sha256(source_path) != source_spec["sha256"]:
            raise ValueError(f"frozen source record hash differs for {model}")
        source = load_dialogue_checkpoint(source_path)
        source_by_key = {
            (record["dialogue_id"], record["condition"], int(record["turn"])): record
            for record in source
            if record.get("role") == "primary"
        }
        record_path = output / f"records_CDR_transport_{model}.jsonl"
        existing = load_dialogue_checkpoint(record_path)
        saved = {
            (record["dialogue_id"], record["condition"]): record
            for record in existing
        }
        client = build_client(model_spec, seed=int(config["seed"]))
        records = []
        with record_path.open("a", encoding="utf-8", newline="\n") as handle:
            for dialogue_offset, dialogue_id in enumerate(
                config["selected_dialogue_ids"]
            ):
                dialogue = dialogues[dialogue_id]
                first_turn = dialogue["turns"][0]
                probe_turn = dialogue["turns"][int(config["probe_turn"]) - 1]
                for condition_offset, condition in enumerate(config["conditions"]):
                    key = (dialogue_id, condition)
                    record = saved.get(key)
                    if record is None:
                        if condition == "self_history":
                            prior_response = source_by_key[
                                (dialogue_id, "self_history", 1)
                            ]["raw_text"]
                        elif condition == "oracle_history":
                            prior_response = first_turn["oracle_response"]
                        else:
                            raise ValueError(f"unsupported probe condition: {condition}")
                        history = [
                            {"role": "user", "content": first_turn["user_prompt"]},
                            {"role": "assistant", "content": prior_response},
                        ]
                        messages = build_packed_turn_messages(
                            current_user_prompt=probe_turn["user_prompt"],
                            history=history,
                            system_prompt=config["system_prompt"],
                        )
                        sample = client.generate_messages(
                            messages,
                            seed=(
                                int(config["seed"])
                                + 100 * dialogue_offset
                                + condition_offset
                            ),
                            gold=str(probe_turn["gold_answer"]),
                        )
                        marker = strict_final_integer(sample.raw_text)
                        record = {
                            "protocol_id": config["protocol_id"],
                            "model": model,
                            "dialogue_id": dialogue_id,
                            "template": dialogue["template"],
                            "turn": int(probe_turn["turn"]),
                            "condition": condition,
                            "history_transport": config["history_transport"],
                            "messages": messages,
                            "gold_answer": str(probe_turn["gold_answer"]),
                            "answer": sample.answer,
                            "raw_text": sample.raw_text,
                            "strict_final_marker": marker is not None,
                            "correct": answers_equal(
                                sample.answer,
                                str(probe_turn["gold_answer"]),
                                "dialogue_integer",
                            ),
                            "ok": bool(sample.ok),
                            "finish_reason": sample.finish_reason,
                            "n_tokens": int(sample.n_tokens),
                            "latency_s": float(sample.latency_s),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        saved[key] = record
                    records.append(record)
        manifests.append(
            {
                "model": model,
                "records": len(records),
                "record_path": str(record_path.relative_to(ROOT)),
                "record_sha256": _sha256(record_path),
            }
        )

    manifest = {
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "expected_total_calls": int(config["expected_total_calls"]),
        "models": manifests,
    }
    manifest_path = output / "transport_probe_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

