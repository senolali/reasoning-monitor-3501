from __future__ import annotations

from reasoning_monitor.progress import (
    ProgressReporter,
    count_jsonl_records,
    format_duration,
)


def test_count_jsonl_records_ignores_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")

    assert count_jsonl_records(path) == 2
    assert count_jsonl_records(tmp_path / "missing.jsonl") == 0


def test_progress_format_is_stable():
    reporter = ProgressReporter(total=10, label="model", completed=2)
    line = reporter.line(2)

    assert "model: 2/10" in line
    assert "20.0%" in line
    assert "ETA --:--:--" in line
    assert format_duration(3661) == "01:01:01"
