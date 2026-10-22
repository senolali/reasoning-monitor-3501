"""reasoning-monitor: multidimensional reasoning-quality monitoring for LLM
query streams under concept drift."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Single source of truth is pyproject.toml; this mirrors the installed
# distribution so `reasoning_monitor.__version__` and the run manifest can never
# disagree (they did: 0.1.0 here, 0.2.0 in pyproject, "0.1.3" in experiment.py).
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("reasoning-monitor")
except Exception:                       # not installed (e.g. plain PYTHONPATH)
    __version__ = "0.2.1"

if TYPE_CHECKING:
    from reasoning_monitor.experiment import ExperimentConfig, run_experiment
    from reasoning_monitor.monitor import MonitorConfig, ReasoningMonitor


def __getattr__(name: str) -> Any:
    """Load the experiment stack only when a public object is requested.

    Lightweight utilities such as the gateway probe can import model clients
    without requiring pandas and the rest of the analysis dependencies.
    """
    if name in {"ExperimentConfig", "run_experiment"}:
        from reasoning_monitor import experiment

        return getattr(experiment, name)
    if name in {"MonitorConfig", "ReasoningMonitor"}:
        from reasoning_monitor import monitor

        return getattr(monitor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ReasoningMonitor",
    "MonitorConfig",
    "ExperimentConfig",
    "run_experiment",
    "__version__",
]
