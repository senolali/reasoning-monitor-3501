"""Run a frozen CDR-Stream stream set (pilot or main).

Default mode prints the plan and makes no call.

  --preflight   call-free: hashes, regenerated instances, probes, budget
  --execute     paid calls (locked until approval is recorded in the config);
                resumable, progress with count, %, calls/min, ETA
  --report      call-free operational report from the checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor import cdr_stream_runner as runner  # noqa: E402
from reasoning_monitor.data import cdr_stream as cs  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_frozen(config: dict) -> tuple[dict, list[dict], list[dict]]:
    for key in ("bank", "mappings", "manifest"):
        path = ROOT / config[f"{key}_path"]
        if sha256_file(path) != config[f"{key}_sha256"]:
            raise SystemExit(f"Frozen {key} hash mismatch: {path}")
    bank = json.loads((ROOT / config["bank_path"]).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / config["manifest_path"]).read_text(encoding="utf-8"))
    units = manifest["units"]
    skeletons = {d["dialogue_id"]: d for d in bank["dialogues"]}
    instances = []
    for unit in units:
        instance = cs.instantiate(skeletons[unit["dialogue_id"]], unit["mapping"],
                                  config["ladder_version"])
        if runner.instance_sha256(instance) != unit["instance_sha256"]:
            raise SystemExit(f"Regenerated instance differs from manifest: {unit['dialogue_id']}")
        instances.append(instance)
    return bank, units, instances


def fingerprint(config: dict) -> str:
    frozen = {key: config[key] for key in (
        "protocol_id", "bank_sha256", "mappings_sha256", "manifest_sha256", "ladder_version",
        "temperature", "max_output_tokens", "request_options", "k_repeats", "l_paraphrases",
        "control_calls")}
    frozen["models"] = [(m["name"], m["provider_model"], bool(m.get("send_seed", True)))
                        for m in config["models"]]
    frozen["prompt_sha256"] = runner.prompt_sha256()
    if config.get("truncation_policy"):  # absent in older configs: fingerprints unchanged
        frozen["truncation_policy"] = config["truncation_policy"]
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def build_clients(config: dict) -> dict:
    return {m["name"]: build_client({
        "backend": "openai_compatible", "name": m["name"], "model": m["provider_model"],
        "base_url": config["base_url"], "api_key_env": config["api_key_env"],
        "temperature": config["temperature"], "max_tokens": config["max_output_tokens"],
        "request_options": config["request_options"], "max_retries": config["max_retries"],
        "backoff_base": 1.0,
        # Opt-in minimum wait after HTTP 429; absent in older configs (= 0).
        "rate_limit_backoff": float(config.get("rate_limit_backoff_s", 0.0)),
        "send_seed": bool(m.get("send_seed", True)), "timeout": 300})
        for m in config["models"]}


def operational_report(rows: list[dict], units: list[dict], k: int) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    report = {}
    for model, items in sorted(by_model.items()):
        dialogue = [r for r in items if r["arm"] == "dialogue"]
        groups: dict[tuple, list[str]] = defaultdict(list)
        for r in dialogue:
            if r["role"] in {"primary", "repeat"}:
                groups[(r["dialogue_id"], r["turn"])].append(r["raw_text"])
        full = [texts for texts in groups.values() if len(texts) == k]
        report[model] = {
            "calls": len(items),
            "contract_validity_dialogue": sum(r["contract_valid"] for r in dialogue)
            / max(1, len(dialogue)),
            "contract_validity_control": sum(r["contract_valid"] for r in items
                                             if r["arm"] == "control")
            / max(1, sum(r["arm"] == "control" for r in items)),
            "truncated": sum(r["truncated"] for r in items),
            "empty_primary": sum(r["role"] == "primary" and r["arm"] == "dialogue"
                                 and not r["raw_text"].strip() for r in items),
            "k_repeat_text_diversity": sum(len(set(t)) > 1 for t in full) / max(1, len(full)),
            "max_prompt_tokens_char4": max((r["prompt_chars"] for r in items), default=0) // 4,
            "median_output_tokens": statistics.median([r["n_tokens"] for r in items])
            if items else None,
            "max_output_tokens_seen": max((r["n_tokens"] for r in items), default=0),
            "median_latency_s": statistics.median([r["latency_s"] for r in items])
            if items else None,
        }
    report["_units"] = len(units)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--arms", nargs="+", choices=("dialogue", "control"), default=None,
                        help="opt-in: submit only these arms (fingerprint unchanged)")
    mode = parser.add_mutually_exclusive_group()
    for flag in ("--preflight", "--execute", "--report"):
        mode.add_argument(flag, action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.workers is not None:
        config["workers"] = args.workers
    _, units, instances = load_frozen(config)
    fp = fingerprint(config)
    k, l_par, control = config["k_repeats"], config["l_paraphrases"], config["control_calls"]
    output = ROOT / config["output_dir"]
    checkpoint = output / "data/records.jsonl"
    planned = sum(len(i["turns"]) for i in instances) * (k + l_par + control)
    budget = [runner.estimate_unit_budget(i, k, l_par, control) for i in instances]
    summary = {"protocol_id": config["protocol_id"], "fingerprint": fp,
               "units": len(units), "planned_calls": planned,
               "estimated_input_tokens_char4": int(sum(b["input_tokens"] for b in budget)),
               "max_context_tokens_char4": int(max(b["max_context_tokens"] for b in budget)),
               "workers": config["workers"], "tokens_per_minute": config["tokens_per_minute"],
               "execution_authorized": config["execution_authorized"],
               "checkpoint": str(checkpoint.relative_to(ROOT))}
    print(json.dumps(summary, indent=2), flush=True)
    if not (args.preflight or args.execute or args.report):
        print("Plan only: no ASU model call was made.", flush=True)
        return 0

    if args.preflight:
        probes = {}
        for model in config["models"]:
            path = ROOT / model["probe"]
            probe = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            disable = [r for r in probe.get("results", []) if r.get("name") == "disable_all"]
            probes[model["name"]] = {
                "exists": path.exists(),
                "model_matches": probe.get("model") == model["provider_model"],
                "accepted": bool(disable) and all(r.get("accepted") for r in disable)}
        ok = all(p["exists"] and p["model_matches"] and p["accepted"] for p in probes.values())
        report = {**summary, "no_model_calls": True, "probes": probes,
                  "instances_regenerated_and_verified": len(instances), "pass": ok}
        write_json(output / "preflight.json", report)
        print(json.dumps({"pass": ok, "probes": probes}, indent=2))
        return 0 if ok else 1

    if args.report:
        saved, failures = runner.load_checkpoint(checkpoint, fp)
        report = {"completed": len(saved), "planned": planned,
                  "failed_attempts_recorded": failures,
                  "by_model": operational_report(list(saved.values()), units, k),
                  "records_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
                  "note": "technical health only; accuracy deliberately not reported"}
        write_json(output / "operational_report.json", report)
        print(json.dumps(report, indent=2))
        return 0

    if not config["execution_authorized"]:
        raise SystemExit("Execution is locked pending explicit investigator approval.")
    if not os.environ.get(config["api_key_env"]):
        raise SystemExit(f"Missing {config['api_key_env']}")
    saved_now = runner.load_checkpoint(checkpoint, fp)[0]
    if args.arms:
        arms = set(args.arms)
        planned = sum(len(i["turns"]) * ((k + l_par) * ("dialogue" in arms)
                                         + control * ("control" in arms)) for i in instances)
        completed = sum(key[3] in arms for key in saved_now)
        print(f"Arm filter {sorted(arms)}: {completed}/{planned} already saved.", flush=True)
    else:
        completed = len(saved_now)
    reporter = ProgressReporter(total=planned, label=config["protocol_id"], completed=completed)
    reporter.print(completed)
    filtered_count = [completed]

    def filtered_progress(_total_saved: int) -> None:
        # The runner reports all saved calls; with a filter, count new calls only.
        filtered_count[0] += 1
        reporter.print(filtered_count[0])
    started = time.monotonic()
    try:
        result = runner.acquire_streams(
            units, instances, build_clients(config), {m["name"]: m for m in config["models"]},
            checkpoint, fp, config["max_output_tokens"], k=k, l_paraphrases=l_par,
            control=control, workers=int(config["workers"]),
            tokens_per_minute=int(config["tokens_per_minute"]),
            callback=filtered_progress if args.arms else reporter.print,
            arms=tuple(args.arms) if args.arms else None,
            mask_truncated_primaries=config.get("truncation_policy")
            == "truncated_primary_placeholder_v1")
    except runner.CircuitBreakerOpen as exc:
        print(f"\nStopped: {exc}", flush=True)
        return 2
    reporter.print(result["completed"], final=True)
    result["elapsed_minutes"] = round((time.monotonic() - started) / 60, 1)
    print(json.dumps(result, indent=2))
    if result["missing"]:
        print("Some calls are missing (failed after retries or blocked behind a failed "
              "primary); rerun the same command to resume.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
