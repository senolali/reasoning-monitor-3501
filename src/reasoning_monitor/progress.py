"""Small progress utilities for checkpointed long-running experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


def count_jsonl_records(path: str | Path) -> int:
    """Count non-empty checkpoint records without parsing the whole file."""
    source = Path(path)
    if not source.exists():
        return 0
    with source.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class ProgressTarget:
    label: str
    path: Path
    expected: int


class ProgressReporter:
    """Print rate and ETA as checkpointed calls complete."""

    def __init__(self, *, total: int, label: str, completed: int = 0):
        if total <= 0:
            raise ValueError("total must be positive")
        self.total = int(total)
        self.label = str(label)
        self.initial_completed = max(0, int(completed))
        self.started_at = time.monotonic()

    def line(self, completed: int) -> str:
        completed = max(0, int(completed))
        elapsed = max(0.0, time.monotonic() - self.started_at)
        newly_completed = max(0, completed - self.initial_completed)
        rate_per_minute = newly_completed / elapsed * 60 if elapsed > 0 else 0.0
        remaining = max(0, self.total - completed)
        eta = remaining / (rate_per_minute / 60) if rate_per_minute > 0 else None
        percent = min(100.0, completed / self.total * 100)
        return (
            f"{self.label}: {completed}/{self.total} ({percent:5.1f}%) | "
            f"{rate_per_minute:5.1f} calls/min | ETA {format_duration(eta)}"
        )

    def print(self, completed: int, *, final: bool = False) -> None:
        end = "\n" if final else "\r"
        print(self.line(completed).ljust(120), end=end, flush=True)

