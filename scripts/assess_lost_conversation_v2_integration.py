"""Call-free semantic-contract assessment of the completed LIC-RQ-v2 integration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_six_metrics import StrictScorers  # noqa: E402
from reasoning_monitor.lost_conversation import load_math_tasks  # noqa: E402
from reasoning_monitor.lost_conversation_runner import (  # noqa: E402
    expected_call_keys,
    score_records,
)
from reasoning_monitor.lost_conversation_semantic_rescore import (  # noqa: E402
    contract_summary,
    final_endpoint,
    metric_summary,
    plot_metrics,
    semantic_records,
    write_json,
)


def main() -> int:
    config_path = ROOT / "configs/lost_conversation_v2_integration_asu.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["manifest_path"]).read_text(encoding="utf-8"))
    design = yaml.safe_load((ROOT / "configs/lost_conversation_v1_design.yaml").read_text(
        encoding="utf-8"))
    tasks, _counts = load_math_tasks(ROOT / design["source"]["path"],
                                     design["source"]["sha256"])
    tasks_by_id = {task.task_id: task for task in tasks}
    task_ids = manifest["integration_task_ids"]
    checkpoint = ROOT / config["output_dir"] / "data" / "records.jsonl"
    raw = checkpoint.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected = len(expected_call_keys(config, task_ids, tasks_by_id))
    keys = {tuple(row[key] for key in (
        "model", "task_id", "condition", "turn", "role", "call_index")) for row in records}
    if len(records) != expected or len(keys) != expected:
        raise ValueError("Incomplete or duplicate integration records")
    rescored = semantic_records(records)
    print("Loading real local NLI and BERTScore for call-free rescore...", flush=True)
    frame = score_records(rescored, tasks_by_id, StrictScorers(with_ss=True),
                          k=config["k_repeats"], n_paraphrases=config["l_paraphrases"])
    output = ROOT / config["output_dir"] / "semantic_rescore"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics_semantic.csv", index=False)
    figures = plot_metrics(frame, output / "figures")
    report = {
        "analysis": "LIC-RQ-v2 call-free semantic-contract integration assessment",
        "no_model_calls": True,
        "exploratory_technical_integration": True,
        "records": len(records),
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "models": contract_summary(rescored),
        "final_endpoint": final_endpoint(frame),
        "metrics": metric_summary(frame),
        "figures": figures,
        "early_warning_established": False,
        "science_run_authorized": False,
        "interpretation": (
            "The relaxed parser accepts only explicit last-line markers and numeric-unit variants. "
            "This integration diagnoses observability and does not estimate a scientific effect."
        ),
    }
    write_json(output / "assessment.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
