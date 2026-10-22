"""List the models the ASU CreateAI gateway offers to this project (no model call).

GET {base_url}/models is a catalogue request: it generates no tokens and costs
nothing. The raw catalogue is saved for provenance, and model IDs are printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://api-main.aiml.asu.edu/v1")
    parser.add_argument("--api-key-env", default="ASU_API_KEY")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "asu_model_catalogue.json")
    args = parser.parse_args(argv)

    token = os.environ.get(args.api_key_env, "").strip()
    if not token:
        print(f"Set {args.api_key_env} first.", file=sys.stderr)
        return 1
    request = urllib.request.Request(f"{args.base_url.rstrip('/')}/models",
                                     headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:500]}",
              file=sys.stderr)
        return 1

    models = sorted(item.get("id", "") for item in payload.get("data", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url, "model_count": len(models), "raw": payload,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(models)} models (saved to {args.output}):")
    for model in models:
        print(f"  {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
