"""Plan and run the frozen LIC-RQ-v2 technical integration; default is call-free."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_six_metrics import StrictScorers  # noqa: E402
from reasoning_monitor.lost_conversation import load_math_tasks  # noqa: E402
from reasoning_monitor.lost_conversation_runner import (  # noqa: E402
    METRICS,
    acquire,
    expected_call_keys,
    operational_report,
    score_records,
)
from reasoning_monitor.models.clients import Sample, build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter, count_jsonl_records  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "lost_conversation_v2_integration_asu.yaml"
DESIGN_PATH = ROOT / "configs" / "lost_conversation_v1_design.yaml"
RUNNER_PATH = ROOT / "src" / "reasoning_monitor" / "lost_conversation_runner.py"
PLOT_METRICS = (*METRICS, "RQ6_mean")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_inputs() -> tuple[dict, dict, dict, list[str]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    design = yaml.safe_load(DESIGN_PATH.read_text(encoding="utf-8"))
    tasks, _counts = load_math_tasks(ROOT / design["source"]["path"],
                                     design["source"]["sha256"])
    tasks_by_id = {task.task_id: task for task in tasks}
    task_ids = manifest["integration_task_ids"]
    if config["protocol_id"] != manifest["protocol_id"] or config["stage"] != "technical_integration":
        raise ValueError("Protocol or integration stage changed")
    if config["conditions"] != manifest["conditions"]:
        raise ValueError("Condition contract changed")
    if [(row["name"], row["provider_model"]) for row in config["models"]] != [
            (row["name"], row["provider_model"]) for row in manifest["models"]]:
        raise ValueError("Model contract changed")
    if (config["k_repeats"], config["l_paraphrases"]) != (5, 2):
        raise ValueError("K/L contract changed")
    expected = len(expected_call_keys(config, task_ids, tasks_by_id))
    if expected != manifest["integration_budget"]["logical_model_calls"] or expected != 1092:
        raise ValueError("Integration budget changed")
    validate_probes(config)
    return config, manifest, tasks_by_id, task_ids


def validate_probes(config: dict) -> None:
    for model in config["models"]:
        probe = json.loads((ROOT / model["probe"]).read_text(encoding="utf-8"))
        if (probe.get("model") != model["provider_model"]
                or probe.get("base_url") != config["base_url"]
                or probe.get("api_key_env") != config["api_key_env"]
                or not probe.get("executed")
                or not probe.get("results")
                or not all(row.get("accepted") for row in probe["results"])):
            raise ValueError(f"Invalid production probe: {model['name']}")


def config_hash(config: dict) -> str:
    paths = [CONFIG_PATH, ROOT / config["manifest_path"], DESIGN_PATH, RUNNER_PATH, Path(__file__)]
    return hashlib.sha256("".join(digest(path) for path in paths).encode("ascii")).hexdigest()


def build_clients(config: dict) -> dict:
    return {
        model["name"]: build_client({
            "backend": "openai_compatible", "name": model["name"],
            "model": model["provider_model"], "base_url": config["base_url"],
            "api_key_env": config["api_key_env"], "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens"],
            "request_options": config["request_options"],
            "max_retries": config["max_retries"], "backoff_base": 1.0,
        })
        for model in config["models"]
    }


def plot_metrics(frame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    written = []
    panels = [(model, data) for model, data in frame.groupby("model")]
    panels += [(f"{model}_{task_id.replace('/', '-')}", data)
               for (model, task_id), data in frame.groupby(["model", "task_id"])]
    colors = {"dialogue_history": "#0072B2", "fresh_prefix": "#D55E00"}
    labels = {"dialogue_history": "Dialogue history", "fresh_prefix": "Fresh prefix"}
    for name, data in panels:
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True,
                                 constrained_layout=True)
        for metric, axis in zip(PLOT_METRICS, axes.flat, strict=False):
            axis.set_title(metric + (" (complete cases)" if metric == "RQ6_mean" else ""))
            support = []
            for condition in ("dialogue_history", "fresh_prefix"):
                subset = data[data.condition == condition]
                grouped = subset.groupby("information_fraction")[metric]
                mean, count = grouped.mean(), grouped.count()
                if mean.notna().any():
                    axis.plot(mean.index, mean, "o-", color=colors[condition],
                              label=labels[condition])
                support.append(f"{labels[condition]} n={int(count.sum())}")
            if not data[metric].notna().any():
                axis.text(0.5, 0.5, "NOT OBSERVABLE\nunder metric contract",
                          transform=axis.transAxes, ha="center", va="center", color="#666666")
            axis.text(0.02, 0.02, " | ".join(support), transform=axis.transAxes, fontsize=7)
            axis.set_ylim(-0.03, 1.08)
            axis.set_xlim(0.0, 1.03)
            axis.grid(alpha=0.2)
            axis.set_xlabel("Revealed information fraction")
            axis.set_ylabel("Score")
        axes.flat[-1].axis("off")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="upper center", ncol=2)
        fig.suptitle(f"LIC-RQ-v2 | {name} | metric trajectories and explicit missingness")
        for extension in ("png", "pdf"):
            path = output / f"seven_metrics_{name}.{extension}"
            fig.savefig(path, dpi=150)
            written.append(path.name)
        plt.close(fig)
    return written


class FixtureClient:
    def generate_messages(self, messages, seed=0, gold=""):
        text = ("1. I preserve the facts revealed so far.\n"
                "2. The remaining facts are not yet available.\n\n"
                "STATUS: NEED_MORE_INFORMATION")
        return Sample(answer="", reasoning_steps=("One", "Two"), latency_s=0.001,
                      n_tokens=24, raw_text=text)


class FixtureScorers:
    def coherence(self, steps):
        return 1.0 if len(steps) >= 2 else np.nan

    def similarity(self, left, right):
        return [1.0] * len(left)


def fixture_preflight(config: dict) -> dict:
    from reasoning_monitor.lost_conversation import MathTask

    fixture = MathTask("fixture/math", "fixture", "5", ("Fact one.", "Fact two."))
    fixture_config = dict(config)
    fixture_config["models"] = [{"name": "fixture", "provider_model": "fixture/model"}]
    with tempfile.TemporaryDirectory(prefix="lic-rq-v2-") as directory:
        root = Path(directory)
        checkpoint = root / "records.jsonl"
        records = acquire(config=fixture_config, tasks={fixture.task_id: fixture},
                          task_ids=[fixture.task_id], clients={"fixture": FixtureClient()},
                          checkpoint=checkpoint, config_hash="fixture-v1", callback=lambda _n: None)
        resumed = acquire(config=fixture_config, tasks={fixture.task_id: fixture},
                          task_ids=[fixture.task_id], clients={"fixture": FixtureClient()},
                          checkpoint=checkpoint, config_hash="fixture-v1", callback=lambda _n: None)
        frame = score_records(records, {fixture.task_id: fixture}, FixtureScorers())
        plots = plot_metrics(frame, root / "plots")
        checks = {
            "expected_records": len(records) == 28,
            "resume_reuses_all_records": len(resumed) == 28,
            "no_duplicate_checkpoint_lines": count_jsonl_records(checkpoint) == 28,
            "explicit_prefix_missingness": set(frame.CQ_status) == {
                "prefix_gold_ineligible", "invalid_final_answer"},
            "ls_and_ss_measured": bool(frame.LS.notna().all() and frame.SS.notna().all()),
            "rq6_not_imputed": bool(frame.RQ6_mean.isna().all()),
            "png_and_pdf_plots": len(plots) == 4
                                  and {Path(path).suffix for path in plots} == {".png", ".pdf"},
        }
    return {"no_model_calls": True, "checks": checks, "pass": all(checks.values())}


def score_and_write(records, tasks_by_id, config, output: Path, *, no_model_calls: bool) -> dict:
    print("Scoring with real local NLI and BERTScore...", flush=True)
    scorers = StrictScorers(with_ss=True)
    frame = score_records(records, tasks_by_id, scorers, k=config["k_repeats"],
                          n_paraphrases=config["l_paraphrases"])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)
    figures = plot_metrics(frame, output / "figures")
    status = {metric: frame[f"{metric}_status"].value_counts().to_dict()
              for metric in PLOT_METRICS}
    report = {
        "protocol_id": config["protocol_id"], "no_model_calls": no_model_calls,
        "metric_status": status, "figures": figures,
        "warning": "Missing metric values are not zeros; RQ6_mean is complete-case only.",
    }
    (output / "scoring_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture-preflight", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute-integration", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    config, manifest, tasks_by_id, task_ids = load_frozen_inputs()
    fingerprint = config_hash(config)
    expected = manifest["integration_budget"]["logical_model_calls"]
    output = ROOT / config["output_dir"]
    checkpoint = output / "data" / "records.jsonl"
    summary = {
        "protocol_id": config["protocol_id"], "stage": config["stage"],
        "execution_authorized": config["execution_authorized"],
        "tasks": len(task_ids), "calls": expected,
        "maximum_output_tokens": expected * config["max_output_tokens"],
        "checkpoint": str(checkpoint.relative_to(ROOT)), "config_hash": fingerprint,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not any(vars(args).values()):
        return 0
    if args.fixture_preflight:
        report = fixture_preflight(config)
        output.mkdir(parents=True, exist_ok=True)
        report.update({"protocol_id": config["protocol_id"], "config_hash": fingerprint})
        (output / "fixture_preflight_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["pass"] else 1
    if args.preflight:
        print("Loading real local NLI and BERTScore scorers...", flush=True)
        scorers = StrictScorers(with_ss=True)
        report = {
            "protocol_id": config["protocol_id"], "config_hash": fingerprint,
            "no_model_calls": True, "real_nli_loaded": True,
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "nli_revision": getattr(scorers.nli.model.config, "_commit_hash", None),
            "real_bertscore_loaded": scorers.bert is not None,
            "bertscore_model": "distilbert-base-uncased",
            "pass": scorers.bert is not None,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "real_scorer_preflight_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print("Preflight passed. No ASU model call was made.", flush=True)
        return 0 if report["pass"] else 1
    if args.execute_integration and not config["execution_authorized"]:
        raise ValueError("Integration execution is locked pending explicit approval")
    if not checkpoint.exists() and args.score_only:
        raise ValueError("No integration checkpoint to score")
    if args.score_only:
        records = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        if len(records) != expected:
            raise ValueError("Checkpoint is incomplete; resume acquisition before scoring")
        report = score_and_write(records, tasks_by_id, config, output, no_model_calls=True)
        print(json.dumps(report, indent=2), flush=True)
        return 0
    if not os.environ.get(config["api_key_env"]):
        raise ValueError(f"Missing {config['api_key_env']}")
    print("Loading real local NLI and BERTScore scorers before paid calls...", flush=True)
    StrictScorers(with_ss=True)
    completed = count_jsonl_records(checkpoint)
    reporter = ProgressReporter(total=expected, label="LIC-RQ-v2 integration", completed=completed)
    reporter.print(completed)
    records = acquire(config=config, tasks=tasks_by_id, task_ids=task_ids,
                      clients=build_clients(config), checkpoint=checkpoint,
                      config_hash=fingerprint, callback=reporter.print)
    reporter.print(len(records), final=True)
    operation = operational_report(records, expected)
    output.mkdir(parents=True, exist_ok=True)
    (output / "operational_report.json").write_text(json.dumps(operation, indent=2),
                                                     encoding="utf-8")
    score_and_write(records, tasks_by_id, config, output, no_model_calls=False)
    print(json.dumps(operation, indent=2), flush=True)
    return 0 if operation["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
