"""Retry the sole ceiling-hit cell from the CDR-v1.1 transport probe."""

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
from reasoning_monitor.metrics.dimensions import answers_equal  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/controlled_dialogue_v1_1_transport_retry_asu.yaml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = ROOT / config["source_probe_path"]
    if _sha256(source_path) != config["source_probe_sha256"]:
        raise ValueError("source transport-probe hash differs from retry config")
    target = config["target"]
    source = load_dialogue_checkpoint(source_path)
    matches = [
        record
        for record in source
        if record.get("model") == target["model"]
        and record.get("dialogue_id") == target["dialogue_id"]
        and record.get("condition") == target["condition"]
        and int(record.get("turn", 0)) == int(target["turn"])
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen retry source record")
    original = matches[0]
    if (
        bool(original.get("correct"))
        or bool(original.get("strict_final_marker"))
        or int(original.get("n_tokens", 0)) != int(target["original_max_tokens"])
    ):
        raise ValueError("frozen source no longer matches the ceiling-hit diagnosis")

    output_path = ROOT / config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        client = build_client(config["model"], seed=int(config["seed"]))
        sample = client.generate_messages(
            original["messages"],
            seed=int(config["seed"]),
            gold=str(target["gold_answer"]),
        )
        marker = strict_final_integer(sample.raw_text)
        result = {
            "protocol_id": config["protocol_id"],
            "model": target["model"],
            "dialogue_id": target["dialogue_id"],
            "condition": target["condition"],
            "turn": int(target["turn"]),
            "source_probe_sha256": config["source_probe_sha256"],
            "messages": original["messages"],
            "gold_answer": str(target["gold_answer"]),
            "answer": sample.answer,
            "raw_text": sample.raw_text,
            "strict_final_marker": marker is not None,
            "correct": answers_equal(
                sample.answer,
                str(target["gold_answer"]),
                "dialogue_integer",
            ),
            "ok": bool(sample.ok),
            "finish_reason": sample.finish_reason,
            "n_tokens": int(sample.n_tokens),
            "max_tokens": int(config["retry_max_tokens"]),
            "latency_s": float(sample.latency_s),
        }
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    manifest = {
        "protocol_id": config["protocol_id"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "expected_calls": int(config["expected_calls"]),
        "output_path": str(output_path.relative_to(ROOT)),
        "output_sha256": _sha256(output_path),
        "result": result,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

