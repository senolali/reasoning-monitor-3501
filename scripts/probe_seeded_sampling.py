"""Probe whether ASU Llama-4 accepts per-call seeds and yields raw diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.models.clients import OpenAICompatibleClient  # noqa: E402

SOURCE_RECORDS = (
    ROOT
    / "results"
    / "proposal_s1_metric_health_gpt_llama4"
    / "data"
    / "records_S1_llama4_scout_seed42_W5.jsonl"
)
DEFAULT_QID = "gsm8k-00131"
SYSTEM_PROMPT = (
    "Use no more than six numbered reasoning lines, one short sentence per line. "
    "After line 6 or sooner, immediately give the best answer even if uncertain. "
    "Put exactly one plain-text FINAL_ANSWER or FINAL_CHOICE line last, outside "
    "LaTeX and code blocks, and write nothing after it."
)
REQUEST_OPTIONS = {
    "enable_history": False,
    "enable_search": False,
    "enhance_prompt": {"time": False, "date": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="aws/llama4_scout-17b")
    parser.add_argument("--base-url", default="https://api-main.aiml.asu.edu/v1")
    parser.add_argument("--api-key-env", default="ASU_API_KEY")
    parser.add_argument("--qid", default=DEFAULT_QID)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _source_prompt(qid: str) -> str:
    for line in SOURCE_RECORDS.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if (
            record.get("qid") == qid
            and record.get("role") == "run"
            and int(record.get("call_index", -1)) == 0
        ):
            return str(record["prompt"])
    raise ValueError(f"primary prompt for {qid!r} not found in Section 64 records")


def main() -> int:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    prompt = _source_prompt(args.qid)
    report = {
        "analysis": "section66_llama4_seed_forwarding_probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "qid": args.qid,
        "repeats": args.repeats,
        "seeds": list(range(args.repeats)),
        "send_seed": True,
        "temperature": 0.7,
        "max_tokens": 2048,
        "request_options": REQUEST_OPTIONS,
        "source_records_sha256": hashlib.sha256(SOURCE_RECORDS.read_bytes()).hexdigest(),
        "executed": args.execute,
    }
    if not args.execute:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("DRY RUN: no network request was sent; add --execute to run the probe.")
        return 0
    if not os.environ.get(args.api_key_env):
        raise SystemExit(
            f"{args.api_key_env} is not set; refusing to send gateway requests"
        )

    client = OpenAICompatibleClient(
        name="llama4_scout_seed_probe",
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=0.7,
        max_tokens=2048,
        request_options=REQUEST_OPTIONS,
        send_seed=True,
        max_retries=0,
        system_prompt=SYSTEM_PROMPT,
    )
    results = []
    for seed in range(args.repeats):
        sample = client.generate(prompt, seed=seed)
        results.append(
            {
                "seed": seed,
                "accepted": sample.ok,
                "latency_s": sample.latency_s,
                "finish_reason": sample.finish_reason,
                "answer": sample.answer,
                "response": sample.raw_text,
            }
        )
    successful = [row for row in results if row["accepted"]]
    report["results"] = results
    report["accepted_calls"] = len(successful)
    report["unique_raw_responses"] = len(
        {str(row["response"]).strip() for row in successful}
    )
    report["unique_answers"] = len({str(row["answer"]) for row in successful})
    report["capability_pass"] = (
        len(successful) == args.repeats and report["unique_raw_responses"] >= 2
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(len(successful) != args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
