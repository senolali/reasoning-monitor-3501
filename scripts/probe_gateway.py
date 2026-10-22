"""Safely probe optional request controls on an OpenAI-compatible gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reasoning_monitor.models.clients import (  # noqa: E402
    OpenAICompatibleClient,
    gateway_probe_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://api-main.aiml.asu.edu/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="ASU_API_KEY")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case",
        choices=[name for name, _options in gateway_probe_cases()],
        help="Run only one probe case, for example baseline.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send the five minimal requests. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = gateway_probe_cases()
    if args.case:
        cases = [(name, options) for name, options in cases if name == args.case]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "executed": args.execute,
        "cases": [{"name": name, "request_options": options} for name, options in cases],
    }

    if not args.execute:
        print(json.dumps(report, indent=2))
        print("DRY RUN: no network request was sent; add --execute to run the probe.")
        return 0
    if not os.environ.get(args.api_key_env):
        raise SystemExit(
            f"{args.api_key_env} is not set; refusing to send gateway requests"
        )

    results = []
    for name, options in cases:
        client = OpenAICompatibleClient(
            name=f"gateway_probe_{name}",
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            request_options=options,
            max_retries=0,
        )
        sample = client.generate("Return only the token OK.")
        results.append({
            "name": name,
            "request_options": options,
            "accepted": sample.ok,
            "latency_s": sample.latency_s,
            "response": sample.raw_text,
        })

    report["results"] = results
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(item["accepted"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
