"""CDR-Stream G1 single-turn difficulty calibration.

Default mode prints the plan and makes no call. Paid modes are locked until the
investigator records an explicit approval in the config.

  --freeze-items   generate + validate the G1 item bank and record its hash
  --preflight      call-free readiness check (items, prompt, probes, budget)
  --execute        run the 2,880 primary G1 calls (resumable, progress + ETA)
  --execute-g1b    run contingency levels 9-10 for the cells named by the rule
  --assess         apply the frozen rule to saved records (call-free) + figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor import cdr_stream_g1 as g1  # noqa: E402
from reasoning_monitor.data import cdr_stream as cs  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/cdr_stream_v1_g1_asu.yaml"
CONFIG_PATH = DEFAULT_CONFIG  # replaced by --config


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_items(config: dict) -> list[dict]:
    path = ROOT / config["items_path"]
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    digest = g1.items_sha256(items)
    if config.get("items_sha256") != digest:
        raise ValueError(f"G1 items hash mismatch: config={config.get('items_sha256')} "
                         f"file={digest}")
    return items


def spec_for(config: dict) -> g1.G1Spec:
    return g1.SPECS[config.get("g1_version", "v1")]


def fingerprint(config: dict) -> str:
    frozen = {k: config[k] for k in ("protocol_id", "items_sha256", "temperature",
                                      "max_output_tokens", "request_options", "order_seed")}
    frozen["models"] = [(m["name"], m["provider_model"]) for m in config["models"]]
    # Extensions only enter the fingerprint when used, so v1/v2 stay unchanged.
    extras = {m["name"]: {k: m[k] for k in ("templates", "send_seed") if k in m}
              for m in config["models"]}
    if any(extras.values()):
        frozen["model_options"] = extras
    frozen["prompt_sha256"] = g1.prompt_sha256()
    if config.get("g1_version", "v1") != "v1":  # v1 fingerprint stays unchanged
        frozen["g1_version"] = config["g1_version"]
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def build_clients(config: dict) -> dict:
    return {m["name"]: build_client({
        "backend": "openai_compatible", "name": m["name"], "model": m["provider_model"],
        "base_url": config["base_url"], "api_key_env": config["api_key_env"],
        "temperature": config["temperature"], "max_tokens": config["max_output_tokens"],
        "request_options": config["request_options"], "max_retries": config["max_retries"],
        "backoff_base": 1.0, "send_seed": bool(m.get("send_seed", True))})
        for m in config["models"]}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def freeze_items(config: dict) -> int:
    spec = spec_for(config)
    items = g1.generate_items(spec)
    shared = config.get("shared_items")
    if shared:
        # Items of templates already frozen elsewhere must be bit-identical.
        frozen_items = json.loads((ROOT / shared["path"]).read_text(encoding="utf-8"))
        if frozen_items["items_sha256"] != shared["sha256"]:
            print("Shared item file hash mismatch")
            return 1
        reference = {i["item_id"]: i for i in frozen_items["items"]
                     if i["template"] in shared["templates"]}
        regenerated = {i["item_id"]: i for i in items if i["template"] in shared["templates"]}
        if reference != regenerated:
            print("Regenerated shared-template items differ from the frozen file")
            return 1
        print(f"Shared-template items identical to the frozen file: {len(reference)}")
    problems = g1.validate_items(items)
    if problems:
        print(json.dumps(problems[:20], indent=2))
        return 1
    digest = g1.items_sha256(items)
    if digest != g1.items_sha256(g1.generate_items(spec)):
        print("G1 item generation is not deterministic")
        return 1
    path = ROOT / config["items_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"g1_id": spec.g1_id, "seed": spec.seed,
                                "version": spec.version,
                                "items_sha256": digest, "items": items}, indent=1)
                    + "\n", encoding="utf-8")
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = re.sub(r"^items_sha256: .*$", f"items_sha256: {digest}", text, count=1,
                  flags=re.MULTILINE)
    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(json.dumps({"items": len(items), "items_sha256": digest,
                      "primary": sum(not i["contingency"] for i in items),
                      "contingency": sum(i["contingency"] for i in items)}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    for flag in ("--freeze-items", "--preflight", "--smoke", "--execute", "--execute-g1b",
                 "--combine",
                 "--assess"):
        mode.add_argument(flag, action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the config's parallel API calls (not in the fingerprint)")
    args = parser.parse_args(argv)
    global CONFIG_PATH
    CONFIG_PATH = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config()
    spec = spec_for(config)
    if args.workers is not None:
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        config["workers"] = args.workers
    if args.freeze_items:
        return freeze_items(config)

    items = load_items(config)
    output = ROOT / config["output_dir"]
    checkpoint = output / "data/records.jsonl"
    fp = fingerprint(config)
    g1b = list(config.get("g1b_cells") or [])
    plan = g1.planned_calls(items, config["models"], g1b if args.execute_g1b else (),
                            config["order_seed"])
    chars = [len(json.dumps(g1.control_messages(i["state_summary"], i["turn_text"])))
             for _, i in plan]
    summary = {"protocol_id": config["protocol_id"], "fingerprint": fp,
               "items_sha256": config["items_sha256"], "prompt_sha256": g1.prompt_sha256(),
               "planned_calls": len(plan),
               "estimated_input_tokens_char4": int(sum(chars) / cs.CHARS_PER_TOKEN),
               "max_output_tokens_ceiling": len(plan) * config["max_output_tokens"],
               "execution_authorized": config["execution_authorized"],
               "g1b_execution_authorized": config["g1b_execution_authorized"],
               "g1b_cells": g1b, "workers": int(config.get("workers", 1)),
               "checkpoint": str(checkpoint.relative_to(ROOT))}
    print(json.dumps(summary, indent=2), flush=True)
    if not any(v for k, v in vars(args).items() if k not in {"config", "workers"}):
        print("Plan only: no ASU model call was made.", flush=True)
        return 0

    if args.preflight:
        problems = g1.validate_items(items)
        probes = {}
        for model in config["models"]:
            probe_path = ROOT / model["probe"]
            if not probe_path.exists():
                probes[model["name"]] = {"path": model["probe"], "exists": False,
                                         "model_matches": False, "executed": False,
                                         "accepted": False}
                continue
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            disable = [r for r in probe.get("results", []) if r.get("name") == "disable_all"]
            probes[model["name"]] = {
                "path": model["probe"], "exists": True, "sha256": sha256_file(probe_path),
                "model_matches": probe.get("model") == model["provider_model"],
                "executed": bool(probe.get("executed")),
                "accepted": bool(disable) and all(r.get("accepted") for r in disable)}
        report = {**summary, "no_model_calls": True, "item_problems": problems,
                  "probes": probes,
                  "prompt_chars_median": statistics.median(chars),
                  "prompt_chars_max": max(chars),
                  "pass": not problems and all(p["model_matches"] and p["executed"]
                                               and p["accepted"]
                                               for p in probes.values())}
        write_json(output / "preflight.json", report)
        print(json.dumps({"pass": report["pass"], "probes": probes}, indent=2))
        return 0 if report["pass"] else 1

    if args.combine:
        sources = [json.loads((ROOT / path).read_text(encoding="utf-8"))
                   for path in config["combine_with"]]
        sources.append(json.loads((output / "assessment.json").read_text(encoding="utf-8")))
        # Investigator-recorded post-hoc overrides of the operational rule
        # (DEVIATIONS_FROM_PROPOSAL). Applied in memory only; the saved
        # assessment keeps its original operational_pass value.
        overrides = dict(config.get("operational_overrides", {}))
        for model in overrides:
            report = sources[-1]["models"].get(model)
            if report is None:
                raise SystemExit(f"Override names a model not in this assessment: {model}")
            report["operational_pass"] = True
        declared = {m["name"]: m["declared_exclusions"] for m in config["models"]
                    if m.get("declared_exclusions")}
        for path in config.get("declared_exclusions_from", []):
            other = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
            declared.update({m["name"]: m["declared_exclusions"] for m in other["models"]
                             if m.get("declared_exclusions")})
        combined = g1.combine_mappings(sources, declared_exclusions=declared)
        combined["declared_exclusions"] = declared
        # Models replaced after a stream-stage gate failure leave the new mappings
        # file; their calibration stays in the cited source assessments.
        dropped = dict(config.get("drop_models", {}))
        for model in dropped:
            combined["models"].pop(model, None)
        if dropped:
            combined["dropped_models"] = dropped
        if overrides:
            combined["operational_overrides"] = overrides
        combined["sources"] = list(config["combine_with"]) + [
            str((output / "assessment.json").relative_to(ROOT))]
        write_json(ROOT / config["mappings_output"], combined)
        print(json.dumps(combined["models"], indent=2))
        return 0

    if args.assess:
        saved, failures = g1.load_checkpoint(checkpoint, fp)
        primary = {(m["name"], i["item_id"]) for m, i in
                   g1.planned_calls(items, config["models"], (), config["order_seed"])}
        missing = primary - set(saved)
        if missing:
            raise SystemExit(f"{len(missing)} primary G1 calls are not completed yet; "
                             "rerun --execute to resume before assessing.")
        records = list(saved.values())
        models = [m["name"] for m in config["models"]]
        assessment = g1.assess_g1(records, models, spec)
        assessment["records_used"] = len(records)
        assessment["failed_attempts_in_checkpoint"] = failures
        assessment["records_sha256"] = sha256_file(checkpoint)
        assessment["figures"] = g1.plot_g1(assessment, output)
        write_json(output / "assessment.json", assessment)
        print(json.dumps({"models": assessment["models"],
                          "needs_G1b": assessment["needs_G1b"]}, indent=2))
        return 0

    authorized = (config["g1b_execution_authorized"] if args.execute_g1b
                  else config["execution_authorized"])
    if not authorized:
        raise SystemExit("Execution is locked pending explicit investigator approval.")
    if args.execute_g1b and not g1b:
        raise SystemExit("No G1b cells were named by the frozen rule.")
    if not os.environ.get(config["api_key_env"]):
        raise SystemExit(f"Missing {config['api_key_env']}")
    if args.smoke:
        # Technical smoke: one primary item per (model, template, level), same keys
        # and seeds as the full plan, so --execute later reuses these records.
        plan = g1.smoke_subset(plan, per_cell=1)
    completed = len(g1.load_checkpoint(checkpoint, fp)[0]
                    .keys() & {(m["name"], i["item_id"]) for m, i in plan})
    reporter = ProgressReporter(total=len(plan), label=spec.g1_id, completed=completed)
    reporter.print(completed)
    try:
        records = g1.acquire(plan, build_clients(config), checkpoint, fp,
                             config["max_output_tokens"], callback=reporter.print,
                             workers=int(config.get("workers", 1)), g1_id=spec.g1_id)
    except g1.CircuitBreakerOpen as exc:
        print(f"\nStopped: {exc}", flush=True)
        return 2
    reporter.print(len(records), final=True)
    if args.smoke:
        report = {"smoke_calls": len(plan), "completed": len(records),
                  "note": "technical health only; accuracy deliberately not reported",
                  "by_model": g1.operational_summary(records)}
        write_json(output / "smoke_report.json", report)
        print(json.dumps(report, indent=2))
        return 0 if len(records) == len(plan) else 2
    _, failures = g1.load_checkpoint(checkpoint, fp)
    operation = {"planned": len(plan), "completed": len(records),
                 "still_missing": len(plan) - len(records),
                 "failed_attempts_recorded": failures,
                 "truncated": sum(r["truncated"] for r in records),
                 "contract_valid": (sum(r["contract_valid"] for r in records)
                                    / max(1, len(records)))}
    write_json(output / "operational_report.json", operation)
    print(json.dumps(operation, indent=2))
    if operation["still_missing"]:
        print("Some calls failed after retries; rerun the same command to retry only "
              "those calls.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
