"""Context-capacity probe for Llama-3 candidates on the ASU gateway.

The Llama-3 models are assumed to have an 8k context window, but this was never
measured on the ASU gateway. This probe sends the hash-verified pilot prompts of
the Llama dialogue at increasing turns (growing packed history) and records, for
each call, whether it succeeded, the gateway error text, and the prompt tokens
the gateway reports. A "needle" call at each length replaces the new message
with a recall question about the oldest user message, which detects silent
history truncation. Technical only; accuracy is not reported.

  default     plan only
  --execute   paid calls (locked until approval is recorded in the config)
  --report    call-free report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_cdr_stream_k_diversity import load_pilot  # noqa: E402

from reasoning_monitor import cdr_stream_g1 as g1  # noqa: E402
from reasoning_monitor import cdr_stream_runner as runner  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/cdr_stream_v1_context_probe_llama3_asu.yaml"
NEEDLE_QUESTION = ("Ignore the usual answer format for this message. Quote, word for word, "
                   "the oldest user message in the conversation so far. Output only that quote.")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_cases(config: dict) -> list[dict]:
    index, state, saved = load_pilot(config)
    first_user = state.instance["turns"][0]["text"]
    cases = []
    for turn in config["turns"]:
        spec = runner.CallSpec(index, turn, "dialogue", "primary", 0)
        messages = runner.build_messages(state, spec)
        recorded = saved[runner.call_key(state.unit, spec)]
        if g1.messages_sha256(messages) != recorded["messages_sha256"]:
            raise SystemExit(f"Rebuilt pilot prompt differs at turn {turn}")
        needle = runner.dialogue_messages(state.history_before(turn), NEEDLE_QUESTION)
        for kind, msgs in (("real", messages), ("needle", needle)):
            cases.append({"turn": turn, "kind": kind, "messages": msgs,
                          "prompt_chars": sum(len(m["content"]) for m in msgs)})
    return cases, first_user


def call(config: dict, model: str, messages: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "stream": False,
            "temperature": config["temperature"], "max_tokens": config["max_output_tokens"],
            **config["request_options"]}
    request = urllib.request.Request(
        f"{config['base_url'].rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ[config['api_key_env']]}",
                 "Content-Type": "application/json"})
    for attempt in range(int(config["max_retries"]) + 1):
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            return {"ok": True, "http_status": 200, "error": None,
                    "raw_text": choice["message"]["content"] or "",
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "latency_s": round(time.time() - started, 3)}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1500]
            if exc.code == 429 and attempt < int(config["max_retries"]):
                time.sleep(float(config["rate_limit_backoff_s"]))
                continue
            return {"ok": False, "http_status": exc.code, "error": detail, "raw_text": "",
                    "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                    "latency_s": round(time.time() - started, 3)}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < int(config["max_retries"]):
                time.sleep(5)
                continue
            return {"ok": False, "http_status": None, "error": f"{type(exc).__name__}: {exc}",
                    "raw_text": "", "finish_reason": None, "prompt_tokens": None,
                    "completion_tokens": None, "latency_s": round(time.time() - started, 3)}
    raise AssertionError("unreachable")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cases, first_user = build_cases(config)
    fp = hashlib.sha256(json.dumps({k: config[k] for k in (
        "protocol_id", "pilot_records_sha256", "turns", "models", "temperature",
        "max_output_tokens", "request_options")}, sort_keys=True).encode()).hexdigest()
    output = ROOT / config["output_dir"]
    checkpoint = output / "records.jsonl"
    planned = len(cases) * len(config["models"])
    print(json.dumps({"protocol_id": config["protocol_id"], "fingerprint": fp,
                      "models": config["models"], "turns": config["turns"],
                      "prompt_tokens_char4": {f"{c['turn']}/{c['kind']}": c["prompt_chars"] // 4
                                              for c in cases},
                      "planned_calls": planned,
                      "execution_authorized": config["execution_authorized"]}, indent=2))

    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["fingerprint"] != fp:
                    raise SystemExit("Checkpoint belongs to a different probe setup")
                saved[(row["model"], row["turn"], row["kind"])] = row

    if args.execute:
        if not config["execution_authorized"]:
            raise SystemExit("Execution is locked pending explicit investigator approval.")
        if not os.environ.get(config["api_key_env"]):
            raise SystemExit(f"Missing {config['api_key_env']}")
        output.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as stream:
            for model in config["models"]:
                for case in cases:
                    key = (model, case["turn"], case["kind"])
                    if key in saved:
                        continue
                    result = call(config, model, case["messages"])
                    row = {"fingerprint": fp, "model": model, "turn": case["turn"],
                           "kind": case["kind"], "prompt_chars": case["prompt_chars"],
                           "messages_sha256": g1.messages_sha256(case["messages"]), **result}
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                    stream.flush()
                    saved[key] = row
                    print(f"{len(saved)}/{planned} {model} turn {case['turn']} {case['kind']}: "
                          f"{'ok' if row['ok'] else 'HTTP ' + str(row['http_status'])} "
                          f"prompt_tokens={row['prompt_tokens']}", flush=True)

    if args.execute or args.report:
        target = normalise(first_user)[:80]
        report = {"completed": len(saved), "planned": planned, "by_model": {},
                  "note": "technical only; accuracy deliberately not reported"}
        for model in config["models"]:
            rows = []
            for case in cases:
                row = saved.get((model, case["turn"], case["kind"]))
                if row is None:
                    continue
                entry = {"turn": row["turn"], "kind": row["kind"],
                         "prompt_tokens_char4": row["prompt_chars"] // 4, "ok": row["ok"],
                         "http_status": row["http_status"],
                         "gateway_prompt_tokens": row["prompt_tokens"],
                         "finish_reason": row["finish_reason"],
                         "error": (row["error"] or "")[:300] or None}
                if row["kind"] == "needle" and row["ok"]:
                    entry["needle_recalled"] = target in normalise(row["raw_text"])
                rows.append(entry)
            real_ok = [r["turn"] for r in rows if r["kind"] == "real" and r["ok"]]
            needle_ok = [r["turn"] for r in rows if r.get("needle_recalled")]
            report["by_model"][model] = {
                "calls": rows,
                "max_turn_real_ok": max(real_ok, default=None),
                "max_turn_needle_recalled": max(needle_ok, default=None),
                "full_100_turns_feasible": (config["turns"][-1] in real_ok
                                            and config["turns"][-1] in needle_ok)}
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
        print(json.dumps({m: {k: v for k, v in r.items() if k != "calls"}
                          for m, r in report["by_model"].items()}, indent=2))
        return 0
    print(f"Plan only: no ASU model call was made. Verified {len(cases)} prompts per model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
