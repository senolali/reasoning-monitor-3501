"""Watch one or more JSONL checkpoints and show live aggregate progress."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.progress import (  # noqa: E402
    ProgressReporter,
    ProgressTarget,
    count_jsonl_records,
)


def _target(value: str) -> ProgressTarget:
    try:
        label, remainder = value.split("=", 1)
        path_text, expected_text = remainder.rsplit(",", 1)
        expected = int(expected_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "target must be LABEL=PATH,EXPECTED"
        ) from error
    if not label.strip() or not path_text.strip() or expected <= 0:
        raise argparse.ArgumentTypeError("target values must be non-empty and positive")
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return ProgressTarget(label=label.strip(), path=path, expected=expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        type=_target,
        help="Repeat as LABEL=PATH,EXPECTED",
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    targets: list[ProgressTarget] = args.target
    total = sum(target.expected for target in targets)
    initial = sum(count_jsonl_records(target.path) for target in targets)
    reporter = ProgressReporter(total=total, label="overall", completed=initial)
    previous = ""
    try:
        while True:
            counts = [count_jsonl_records(target.path) for target in targets]
            completed = sum(counts)
            detail = " | ".join(
                f"{target.label} {count}/{target.expected}"
                for target, count in zip(targets, counts, strict=True)
            )
            line = f"{reporter.line(completed)} | {detail}"
            print(line.ljust(max(120, len(previous))), end="\r", flush=True)
            previous = line
            if args.once or all(
                count >= target.expected
                for target, count in zip(targets, counts, strict=True)
            ):
                print()
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nProgress watcher stopped; the experiment is unaffected.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

