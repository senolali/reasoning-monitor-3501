"""Audit the completed CDR-v1 pilot for native-history transport failure."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_experiment import load_dialogue_checkpoint  # noqa: E402
from reasoning_monitor.evaluation.dialogue_transport import (  # noqa: E402
    audit_native_history_transport,
)

SOURCES = {
    "gpt4o_mini": (
        "results/controlled_dialogue_v1_pilot/data/records_CDR_gpt4o_mini.jsonl",
        "23541a3ea121769a708837436e730ab2ab133fbde84b1342749c083eda5ec3d2",
    ),
    "llama4_scout": (
        "results/controlled_dialogue_v1_pilot/data/records_CDR_llama4_scout.jsonl",
        "779bca9c0097cafd1f749b30cf669f95d7add6db710893d37018b6525c509a6a",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    records_by_model = {}
    source_sha256 = {}
    for model, (relative, expected) in SOURCES.items():
        path = ROOT / relative
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"frozen CDR-v1 record hash differs for {model}")
        source_sha256[relative] = digest
        records_by_model[model] = load_dialogue_checkpoint(path)
    report = audit_native_history_transport(records_by_model)
    report["source_sha256"] = source_sha256
    output = ROOT / "results" / "controlled_dialogue_v1_pilot" / "data"
    report_path = output / "controlled_dialogue_v1_transport_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(not report["transport_failure_signature_detected"])


if __name__ == "__main__":
    raise SystemExit(main())

