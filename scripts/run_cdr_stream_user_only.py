"""Acquire the CDR-Stream user-only history arm (CDR_STREAM_USER_ONLY_PROTOCOL.md).

Default mode prints the plan and makes no call.

  --preflight   call-free: frozen hashes, regenerated instances, probes, budget
  --execute     paid calls (locked until approval and model option are recorded);
                resumable, progress with count, %, calls/min, ETA
  --report      call-free operational report (no accuracy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cdr_stream import build_clients, load_frozen, sha256_file, write_json  # noqa: E402

from reasoning_monitor import cdr_stream_runner as runner  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/cdr_stream_v1_user_only_asu.yaml"


def selected_models(config: dict) -> list[dict]:
    option = config.get("model_option")
    allowed = {"A": {"A"}, "B": {"A", "B"}}.get(option, {"A", "B"})
    return [m for m in config["models"] if m["option"] in allowed]


def fingerprint(config: dict) -> str:
    frozen = {key: config[key] for key in (
        "protocol_id", "bank_sha256", "mappings_sha256", "manifest_sha256", "ladder_version",
        "temperature", "max_output_tokens", "request_options", "protocol_sha256")}
    frozen["models"] = [(m["name"], m["provider_model"], bool(m.get("send_seed", True)))
                        for m in selected_models(config)]
    frozen["prompt_sha256"] = runner.prompt_sha256()
    frozen["arm"] = runner.USER_ONLY_ARM
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    for flag in ("--preflight", "--execute", "--report"):
        mode.add_argument(flag, action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = selected_models(config)
    names = {m["name"] for m in models}
    _, units, instances = load_frozen(config)
    keep = [i for i, u in enumerate(units) if u["model"] in names]
    units, instances = [units[i] for i in keep], [instances[i] for i in keep]
    fp = fingerprint(config)
    output = ROOT / config["output_dir"]
    checkpoint = output / "data/records.jsonl"
    planned = sum(len(i["turns"]) for i in instances)
    chars = sum(sum(len(m["content"]) for m in runner.user_only_messages(inst, t))
                for inst in instances for t in range(1, len(inst["turns"]) + 1))
    summary = {"protocol_id": config["protocol_id"], "fingerprint": fp,
               "model_option": config.get("model_option"), "models": sorted(names),
               "units": len(units), "planned_calls": planned,
               "estimated_input_tokens_char4": chars // 4,
               "execution_authorized": config["execution_authorized"],
               "checkpoint": str(checkpoint.relative_to(ROOT))}
    print(json.dumps(summary, indent=2), flush=True)
    if not (args.preflight or args.execute or args.report):
        print("Plan only: no ASU model call was made.", flush=True)
        return 0
    if args.preflight:
        probes = {}
        for model in models:
            path = ROOT / model["probe"]
            probe = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            disable = [r for r in probe.get("results", []) if r.get("name") == "disable_all"]
            probes[model["name"]] = bool(disable) and all(r.get("accepted") for r in disable) \
                and probe.get("model") == model["provider_model"]
        protocol_ok = (config.get("protocol_sha256") is None
                       or sha256_file(ROOT / config["protocol_path"]) == config["protocol_sha256"])
        report = {**summary, "no_model_calls": True, "probes": probes,
                  "protocol_hash_ok": protocol_ok,
                  "pass": all(probes.values()) and protocol_ok}
        write_json(output / "preflight.json", report)
        print(json.dumps({"pass": report["pass"], "probes": probes,
                          "protocol_hash_ok": protocol_ok}, indent=2))
        return 0 if report["pass"] else 1
    if args.report:
        saved, failures = runner.load_checkpoint(checkpoint, fp)
        rows = list(saved.values())
        by_model = {}
        for name in sorted(names):
            r = [x for x in rows if x["model"] == name]
            by_model[name] = {"calls": len(r),
                              "contract_validity": sum(x["contract_valid"] for x in r)
                              / max(1, len(r)),
                              "truncated": sum(x["truncated"] for x in r)}
        report = {"completed": len(rows), "planned": planned, "failed_attempts": failures,
                  "by_model": by_model,
                  "records_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
                  "note": "technical health only; accuracy deliberately not reported"}
        write_json(output / "operational_report.json", report)
        print(json.dumps(report, indent=2))
        return 0
    if not config["execution_authorized"] or config.get("model_option") not in ("A", "B") \
            or not config.get("protocol_sha256"):
        raise SystemExit("Locked: needs approval, a recorded model option and a frozen "
                         "protocol hash.")
    if sha256_file(ROOT / config["protocol_path"]) != config["protocol_sha256"]:
        raise SystemExit("Protocol file differs from its frozen hash.")
    if not os.environ.get(config["api_key_env"]):
        raise SystemExit(f"Missing {config['api_key_env']}")
    completed = len(runner.load_checkpoint(checkpoint, fp)[0])
    reporter = ProgressReporter(total=planned, label=config["protocol_id"], completed=completed)
    reporter.print(completed)
    started = time.monotonic()
    try:
        result = runner.acquire_user_only(
            units, instances, build_clients({**config, "models": models}),
            {m["name"]: m for m in models}, checkpoint, fp, config["max_output_tokens"],
            workers=int(config["workers"]), tokens_per_minute=int(config["tokens_per_minute"]),
            callback=reporter.print)
    except runner.CircuitBreakerOpen as exc:
        print(f"\nStopped: {exc}", flush=True)
        return 2
    reporter.print(result["completed"], final=True)
    result["elapsed_minutes"] = round((time.monotonic() - started) / 60, 1)
    print(json.dumps(result, indent=2))
    return 2 if result["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
