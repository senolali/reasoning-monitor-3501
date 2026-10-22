"""Retry the 10 capacity-censored CDR-v1.2 cells with exact messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_experiment import (  # noqa: E402
    load_dialogue_checkpoint,
    messages_sha256,
)
from reasoning_monitor.models.clients import build_client  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlled_dialogue_v1_2_capacity_retry_asu.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["output_dir"] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records_CDR12a_capacity_retry.jsonl"
    saved = {
        (str(record["model"]), str(record["qid"])): record
        for record in _load_jsonl(output_path)
    }
    produced: list[dict[str, Any]] = []

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for model_name, model_spec in config["models"].items():
            source_path = ROOT / model_spec["source_path"]
            if _sha256(source_path) != model_spec["source_sha256"]:
                raise ValueError(f"source record hash differs for {model_name}")
            source = {
                str(record["qid"]): record
                for record in load_dialogue_checkpoint(source_path)
            }
            client = build_client(model_spec["client"], seed=int(config["seed"]))
            for target in model_spec["targets"]:
                qid = str(target["qid"])
                original = source.get(qid)
                if original is None:
                    raise ValueError(f"missing frozen source record: {model_name}/{qid}")
                if (
                    int(original.get("n_tokens", 0))
                    != int(config["original_max_tokens"])
                    or str(original.get("gold_answer")) != str(target["gold_answer"])
                    or messages_sha256(original["messages"])
                    != target["messages_sha256"]
                ):
                    raise ValueError(f"frozen target differs: {model_name}/{qid}")
                key = (model_name, qid)
                record = saved.get(key)
                if record is None:
                    sample = client.generate_messages(
                        original["messages"],
                        seed=int(target["call_seed"]),
                        gold=str(target["gold_answer"]),
                    )
                    record = dict(original)
                    record.update(
                        {
                            "protocol_id": "CDR-v1.2a",
                            "role": "primary",
                            "retry_role": "capacity_retry",
                            "source_max_tokens": int(config["original_max_tokens"]),
                            "retry_max_tokens": int(config["retry_max_tokens"]),
                            "source_messages_sha256": target["messages_sha256"],
                            "messages_sha256": messages_sha256(original["messages"]),
                            "call_seed": int(target["call_seed"]),
                            "raw_text": sample.raw_text,
                            "answer": sample.answer,
                            "reasoning_steps": list(sample.reasoning_steps),
                            "n_tokens": int(sample.n_tokens),
                            "latency_s": float(sample.latency_s),
                            "ok": bool(sample.ok),
                            "finish_reason": sample.finish_reason,
                        }
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    saved[key] = record
                produced.append(record)

    manifest = {
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "expected_calls": int(config["expected_total_calls"]),
        "records": len(produced),
        "output_path": str(output_path.relative_to(ROOT)),
        "output_sha256": _sha256(output_path),
    }
    manifest_path = output_dir / "capacity_retry_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
