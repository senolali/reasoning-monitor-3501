"""LHE-RQ-v1 runner: plan (default, call-free), preflight, execute, report.

  python scripts\\run_lhe_early_warning.py --config configs\\lhe_rq_v1_pilot_asu.yaml
  python scripts\\run_lhe_early_warning.py --config ... --preflight [--scorers]
  python scripts\\run_lhe_early_warning.py --config ... --execute     (approval required)
  python scripts\\run_lhe_early_warning.py --config ... --report      (call-free)

The default prints the frozen plan and a char/4 budget per model; no ASU call
is made. --execute refuses to run unless the config says
``execution_authorized: true``. Progress shows completed/total, percentage,
calls per minute and ETA. Rerunning --execute resumes from the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reasoning_monitor import lhe_acquisition as acq  # noqa: E402
from reasoning_monitor import lhe_early_warning as lhe  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402
from reasoning_monitor.progress import ProgressReporter  # noqa: E402

UNFINGERPRINTED = {"workers", "tokens_per_minute", "rate_limit_backoff_s",
                   "execution_authorized", "assumed_reply_chars", "prices_per_million"}


def fingerprint(config: dict, dictionary: dict) -> str:
    frozen = {k: v for k, v in config.items() if k not in UNFINGERPRINTED}
    frozen["data_hashes"] = lhe.DATA_HASHES
    frozen["prompt_sha256"] = {m["name"]: lhe.prompt_sha256(dictionary, m["k"],
                                                              m.get("mode", "cot"))
                               for m in config["models"]}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def budget(config: dict, dictionary: dict, units: list[dict]) -> dict:
    calls_per_turn = config["k_samples"] + config["l_paraphrases"]
    by_model: dict[str, dict] = {}
    for unit in units:
        chars = config["assumed_reply_chars"]
        if isinstance(chars.get("cot"), dict):         # pilot-2: per mode, then per k
            reply = chars[unit.get("mode", "cot")][unit["k"]]
        else:                                          # pilot-1: per k
            reply = chars[unit["k"]]
        est = lhe.estimate_conversation_tokens(dictionary, unit["conversation"],
                                               calls_per_turn, reply,
                                               mode=unit.get("mode", "cot"))
        entry = by_model.setdefault(unit["model"], {"provider_model": unit["provider_model"],
                                                    "calls": 0, "input_tokens": 0.0,
                                                    "output_tokens": 0.0})
        for key in ("calls", "input_tokens", "output_tokens"):
            entry[key] += est[key]
    for entry in by_model.values():
        price = config["prices_per_million"].get(entry["provider_model"])
        entry["input_tokens"] = int(entry["input_tokens"])
        entry["output_tokens"] = int(entry["output_tokens"])
        entry["usd_estimate"] = (None if price is None else round(
            entry["input_tokens"] / 1e6 * price[0] + entry["output_tokens"] / 1e6 * price[1], 2))
    return by_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    for flag in ("--preflight", "--execute", "--report"):
        mode.add_argument(flag, action="store_true")
    parser.add_argument("--scorers", action="store_true",
                        help="preflight: also load the real NLI/BERTScore scorers")
    parser.add_argument("--api-key-env", default=None,
                        help="execute: read the API key from this environment variable "
                             "instead of the config's (fingerprint unchanged)")
    args = parser.parse_args(argv)

    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    dictionary, samples = lhe.load_dataset(ROOT / config["data_dir"])
    units = acq.build_units(config["models"], samples, config["conversation_ids"],
                            config["horizon"])
    fp = fingerprint(config, dictionary)
    output = ROOT / config["output_dir"]
    checkpoint = output / "data/records.jsonl"
    k_s, l_p = config["k_samples"], config["l_paraphrases"]
    planned = len(units) * config["horizon"] * (k_s + l_p)
    plan = {"protocol_id": config["protocol_id"], "fingerprint": fp, "units": len(units),
            "planned_calls": planned, "budget_char4": budget(config, dictionary, units),
            "execution_authorized": config["execution_authorized"],
            "checkpoint": str(checkpoint.relative_to(ROOT))}
    print(json.dumps(plan, indent=2), flush=True)
    if not (args.preflight or args.execute or args.report):
        print("Plan only: no ASU model call was made.", flush=True)
        return 0

    if args.preflight:
        probes = {}
        for model in config["models"]:
            probe_path = ROOT / model["probe"]
            probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
            disable = [r for r in probe.get("results", []) if r.get("name") == "disable_all"]
            probes[model["name"]] = {"exists": probe_path.exists(),
                                     "model_matches": probe.get("model") == model["provider_model"],
                                     "accepted": bool(disable) and all(r.get("accepted")
                                                                       for r in disable)}
        scorers = None
        if args.scorers:
            from reasoning_monitor.dialogue_six_metrics import StrictScorers
            StrictScorers(with_ss=True)
            scorers = "real NLI and BERTScore loaded"
        ok = all(all(p.values()) for p in probes.values())
        report = {**plan, "no_model_calls": True, "data_hashes_verified": True,
                  "probes": probes, "scorers": scorers, "pass": ok}
        output.mkdir(parents=True, exist_ok=True)
        (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n",
                                               encoding="utf-8")
        print(json.dumps({"pass": ok, "probes": probes, "scorers": scorers}, indent=2))
        return 0 if ok else 1

    if args.report:
        saved, failures = acq.load_checkpoint(checkpoint, fp)
        rows = list(saved.values())
        by_model = {}
        for name in sorted({r["model"] for r in rows}):
            part = [r for r in rows if r["model"] == name]
            by_model[name] = {
                "calls": len(part), "truncated": sum(bool(r["truncated"]) for r in part),
                "parse_rate": round(sum(r["answer"] is not None for r in part) / len(part), 4),
                "strict_rate": round(sum(bool(r["strict"]) for r in part) / len(part), 4),
                "median_output_tokens": statistics.median(r["n_tokens"] for r in part)}
        report = {"completed": len(rows), "planned": planned, "failed_attempts": failures,
                  "by_model": by_model,
                  "note": "technical health only; accuracy deliberately not reported"}
        print(json.dumps(report, indent=2))
        return 0

    if not config["execution_authorized"]:
        raise SystemExit("Execution is locked pending explicit investigator approval.")
    key_env = args.api_key_env or config["api_key_env"]
    if not os.environ.get(key_env):
        raise SystemExit(f"Missing {key_env}")
    print(f"API key read from ${key_env}", flush=True)
    clients = {m["name"]: build_client({
        "backend": "openai_compatible", "name": m["name"], "model": m["provider_model"],
        "base_url": config["base_url"], "api_key_env": key_env,
        "temperature": config["temperature"], "max_tokens": config["max_output_tokens"],
        "request_options": config["request_options"], "max_retries": config["max_retries"],
        "backoff_base": 1.0, "rate_limit_backoff": float(config["rate_limit_backoff_s"]),
        "send_seed": bool(m.get("send_seed", True)), "timeout": 300})
        for m in config["models"]}
    completed = len(acq.load_checkpoint(checkpoint, fp)[0])
    reporter = ProgressReporter(total=planned, label=config["protocol_id"], completed=completed)
    reporter.print(completed)
    started = time.monotonic()
    try:
        result = acq.acquire(units, dictionary, clients, checkpoint, fp,
                             config["max_output_tokens"], k_samples=k_s, l_paraphrases=l_p,
                             workers=int(config["workers"]),
                             tokens_per_minute=int(config["tokens_per_minute"]),
                             callback=reporter.print)
    except acq.CircuitBreakerOpen as exc:
        print(f"\nStopped: {exc}", flush=True)
        return 2
    reporter.print(result["completed"], final=True)
    result["elapsed_minutes"] = round((time.monotonic() - started) / 60, 1)
    print(json.dumps(result, indent=2))
    return 0 if not result["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
