"""K-repeat diversity probe for an alternative Llama endpoint or model.

The technical pilot failed only on Llama-4 Scout (aws) K-repeat raw-text
diversity (0.38 < 0.80). This probe replays the *identical* pilot prompts of the
Llama dialogue at fixed turns against a candidate endpoint, with K=3 and
distinct seeds, and compares raw-text diversity with the aws repeats already
recorded in the pilot. Histories are rebuilt from the recorded pilot primaries
and checked against the recorded request hashes. Accuracy is not reported.

  default     plan only
  --execute   paid calls (locked until approval is recorded in the config)
  --report    call-free report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_cdr_stream import fingerprint as pilot_fingerprint  # noqa: E402
from run_cdr_stream import load_frozen  # noqa: E402

from reasoning_monitor import cdr_stream_g1 as g1  # noqa: E402
from reasoning_monitor import cdr_stream_runner as runner  # noqa: E402
from reasoning_monitor.data import cdr_stream as cs  # noqa: E402
from reasoning_monitor.models.clients import build_client  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/cdr_stream_v1_kdiv_probe_asu.yaml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pilot(config: dict):
    pilot_config = yaml.safe_load((ROOT / config["pilot_config"]).read_text(encoding="utf-8"))
    records_path = ROOT / pilot_config["output_dir"] / "data/records.jsonl"
    if sha256_file(records_path) != config["pilot_records_sha256"]:
        raise SystemExit("Pilot records hash mismatch")
    _, units, instances = load_frozen(pilot_config)
    fp = pilot_fingerprint(pilot_config)
    saved, _ = runner.load_checkpoint(records_path, fp)
    index = next(i for i, u in enumerate(units) if u["model"] == config["pilot_model"])
    state = runner.UnitState(units[index], instances[index])
    for row in saved.values():
        if (row["model"] == config["pilot_model"] and row["arm"] == "dialogue"
                and row["role"] == "primary"):
            state.primaries[row["turn"]] = row["raw_text"]
    return index, state, saved


def probe_prompts(config: dict):
    index, state, saved = load_pilot(config)
    prompts = []
    for turn in config["turns"]:
        spec = runner.CallSpec(index, turn, "dialogue", "primary", 0)
        messages = runner.build_messages(state, spec)
        recorded = saved[runner.call_key(state.unit, spec)]
        if g1.messages_sha256(messages) != recorded["messages_sha256"]:
            raise SystemExit(f"Rebuilt pilot prompt differs at turn {turn}")
        aws_texts = [saved[runner.call_key(state.unit, runner.CallSpec(
            index, turn, "dialogue", role, i))]["raw_text"]
            for role, i in (("primary", 0), ("repeat", 1), ("repeat", 2))]
        prompts.append({"turn": turn, "messages": messages,
                        "messages_sha256": recorded["messages_sha256"],
                        "gold": state.instance["turns"][turn - 1]["gold"],
                        "aws_texts": aws_texts})
    return prompts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prompts = probe_prompts(config)
    candidate = config["candidate"]
    fp = hashlib.sha256(json.dumps({
        "protocol_id": config["protocol_id"], "pilot_records": config["pilot_records_sha256"],
        "turns": config["turns"], "k": config["k"], "candidate": candidate,
        "temperature": config["temperature"], "max_output_tokens": config["max_output_tokens"],
        "request_options": config["request_options"]}, sort_keys=True).encode()).hexdigest()
    output = ROOT / config["output_dir"]
    checkpoint = output / "records.jsonl"
    planned = len(prompts) * config["k"]
    print(json.dumps({"protocol_id": config["protocol_id"], "fingerprint": fp,
                      "candidate": candidate["provider_model"], "turns": config["turns"],
                      "planned_calls": planned,
                      "execution_authorized": config["execution_authorized"]}, indent=2))

    saved = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["fingerprint"] != fp:
                    raise SystemExit("Checkpoint belongs to a different probe setup")
                if row["ok"]:
                    saved[(row["turn"], row["k"])] = row

    if args.execute:
        if not config["execution_authorized"]:
            raise SystemExit("Execution is locked pending explicit investigator approval.")
        if not os.environ.get(config["api_key_env"]):
            raise SystemExit(f"Missing {config['api_key_env']}")
        client = build_client({
            "backend": "openai_compatible", "name": candidate["name"],
            "model": candidate["provider_model"], "base_url": config["base_url"],
            "api_key_env": config["api_key_env"], "temperature": config["temperature"],
            "max_tokens": config["max_output_tokens"],
            "request_options": config["request_options"],
            "max_retries": int(config.get("max_retries", 4)), "backoff_base": 1.0,
            "rate_limit_backoff": float(config.get("rate_limit_backoff_s", 0.0)),
            "send_seed": bool(candidate.get("send_seed", True)),
            "timeout": 300})
        todo = [(p, k) for p in prompts for k in range(config["k"])
                if (p["turn"], k) not in saved]
        output.mkdir(parents=True, exist_ok=True)
        with (checkpoint.open("a", encoding="utf-8") as stream,
              ThreadPoolExecutor(max_workers=int(config["workers"])) as pool):
            futures = {}
            for prompt, k in todo:
                seed = cs._derive_seed(fp, prompt["turn"], k) % (2 ** 31)
                futures[pool.submit(client.generate_messages, prompt["messages"],
                                    seed=seed, gold="")] = (prompt, k, seed)
            for future in as_completed(futures):
                prompt, k, seed = futures[future]
                sample = future.result()
                row = {"fingerprint": fp, "model": candidate["name"],
                       "provider_model": candidate["provider_model"], "turn": prompt["turn"],
                       "k": k, "call_seed": seed, "messages_sha256": prompt["messages_sha256"],
                       "raw_text": sample.raw_text,
                       **g1.score_response(sample.raw_text, prompt["gold"]),
                       "ok": bool(sample.ok), "n_tokens": int(sample.n_tokens),
                       "finish_reason": getattr(sample, "finish_reason", None),
                       "latency_s": sample.latency_s}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                if row["ok"]:
                    saved[(prompt["turn"], k)] = row
                print(f"\r{len(saved)}/{planned} calls", end="", flush=True)
        print()

    if args.execute or args.report:
        by_turn = defaultdict(list)
        for (turn, _k), row in saved.items():
            by_turn[turn].append(row["raw_text"])
        complete = [turn for turn in config["turns"] if len(by_turn[turn]) == config["k"]]
        candidate_div = sum(len(set(by_turn[t])) > 1 for t in complete) / max(1, len(complete))
        aws_div = sum(len(set(p["aws_texts"])) > 1 for p in prompts) / len(prompts)
        rows = list(saved.values())
        report = {
            "completed": len(saved), "planned": planned, "turns_complete": len(complete),
            "candidate": candidate["provider_model"],
            "candidate_k_diversity": candidate_div,
            "aws_k_diversity_same_turns": aws_div,
            "candidate_contract_validity": sum(r["contract_valid"] for r in rows)
            / max(1, len(rows)),
            "candidate_truncated": sum(r["n_tokens"] >= config["max_output_tokens"]
                                       or r["finish_reason"] == "length" for r in rows),
            "gate": {"k_diversity_min": config["gate_k_diversity"],
                     "contract_validity_min": config["gate_contract_validity"]},
            "note": "technical only; accuracy deliberately not reported",
        }
        report["pass"] = (len(complete) == len(config["turns"])
                          and report["candidate_k_diversity"] >= config["gate_k_diversity"]
                          and report["candidate_contract_validity"]
                          >= config["gate_contract_validity"]
                          and report["candidate_truncated"] == 0)
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["pass"] else 1
    print("Plan only: no ASU model call was made. Rebuilt pilot prompts verified: "
          f"{len(prompts)} turns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
