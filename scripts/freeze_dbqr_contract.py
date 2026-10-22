"""Freeze DBQR call budgets, label-blind manifests and plot layout without calls."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dbqr_audit import MAX_BYTES, SOURCE_SHA256  # noqa: E402
from reasoning_monitor.dbqr_campaign_contract import (  # noqa: E402
    fixture_plot_frame,
    framed_prompts,
    load_budget_contract,
    plot_seven_metrics,
    write_budget_report,
)
from reasoning_monitor.dbqr_prompt_manifest import build_prompt_manifest  # noqa: E402
from reasoning_monitor.dbqr_training_audit import (  # noqa: E402
    TRAINING_MAX_BYTES,
    TRAINING_SHA256,
)


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    config_path = ROOT / "configs/dbqr_rq_v1_budget.yaml"
    config, budget = load_budget_contract(config_path)
    output = ROOT / "results/dbqr_v1_contract"
    output.mkdir(parents=True, exist_ok=True)

    fixture_path = ROOT / "results/dbqr_v1_fixture/data/records_DBQR_fixture.jsonl"
    records = _records(fixture_path)
    prompt_totals = defaultdict(int)
    for record in records:
        if record["condition"] == "self_history":
            prompt_totals[record["dialogue_id"]] += len(record["messages"][-1]["content"])
    ordered = sorted(prompt_totals, key=lambda key: (prompt_totals[key], key))
    structural_median = ordered[len(ordered) // 2]
    selected = config["stages"]["practice_transport"]["selected_dialogues"]

    example = "ORIGINAL QUESTION AND SCHEMA"
    variants = framed_prompts(example, config)
    practice_manifest = build_prompt_manifest(
        (ROOT / "data/dbqr/practice.tar.gz").read_bytes(),
        split="practice",
        expected_sha256=SOURCE_SHA256,
        maximum_bytes=MAX_BYTES,
    )
    training_manifest = build_prompt_manifest(
        (ROOT / "data/dbqr/training.tar.gz").read_bytes(),
        split="training",
        expected_sha256=TRAINING_SHA256,
        maximum_bytes=TRAINING_MAX_BYTES,
    )
    selection_dir = ROOT / "configs/selections"
    selection_dir.mkdir(parents=True, exist_ok=True)
    practice_path = selection_dir / "dbqr_practice_prompt_manifest.json"
    training_path = selection_dir / "dbqr_training_prompt_manifest.json"
    practice_path.write_text(json.dumps(practice_manifest, indent=2), encoding="utf-8")
    training_path.write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")

    answers = {}
    for record in records:
        answers.setdefault(record["qid"], record["gold_answer"])
    vault_body = {
        "protocol_id": "DBQR-RQ-v1",
        "split": "practice",
        "source_sha256": practice_manifest["source_sha256"],
        "answers": answers,
    }
    vault_body["vault_sha256"] = hashlib.sha256(
        json.dumps(vault_body, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")).encode("ascii")
    ).hexdigest()
    vault_path = output / "practice_oracle_vault.json"
    vault_path.write_text(json.dumps(vault_body, indent=2), encoding="utf-8")

    figures = plot_seven_metrics(
        fixture_plot_frame(records),
        output / "figures",
        title="CALL-FREE FIXTURE LAYOUT CHECK - NOT MODEL EVIDENCE",
    )
    checks = {
        "practice_selection_is_structural_median": selected == [structural_median],
        "two_distinct_framing_variants_preserve_original": (
            len(variants) == 2 and len(set(variants)) == 2
            and all(example in variant for variant in variants)),
        "practice_manifest_is_label_blind_and_complete": (
            practice_manifest["label_blind"] and len(practice_manifest["cases"]) == 50),
        "training_manifest_is_label_blind_and_complete": (
            training_manifest["label_blind"] and len(training_manifest["cases"]) == 200),
        "seven_panel_plot_contract_rendered": len(figures) == 2
                                             and all(path.exists() for path in figures),
        "no_paid_execution_authorized_after_failed_review": (
            config["stages"]["practice_transport"]["stage_review_authorized"] is False
            and config["stages"]["practice_transport"]["full_execution_authorized"] is False
            and config["stages"]["practice_transport"]["execution_authorized"] is False
            and config["stages"]["training_calibration"]["execution_authorized"] is False
            and config["stages"]["test_confirmation"]["execution_authorized"] is False
        ),
    }
    report = {
        **budget,
        "stage": "call_free_budget_manifest_and_plot_contract",
        "structural_prompt_character_totals": dict(sorted(prompt_totals.items())),
        "structural_median_dialogue": structural_median,
        "practice_prompt_manifest_sha256": practice_manifest["manifest_sha256"],
        "training_prompt_manifest_sha256": training_manifest["manifest_sha256"],
        "practice_oracle_vault_sha256": vault_body["vault_sha256"],
        "plot_files": [str(path.relative_to(ROOT)) for path in figures],
        "checks": checks,
        "pass": all(checks.values()),
    }
    write_budget_report(report, output / "assessment.json")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
