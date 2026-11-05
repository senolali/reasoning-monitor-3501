"""Build lightweight Overleaf and call-free reproducibility archives.

Raw provider responses are intentionally excluded. The reproducibility archive
contains frozen protocols, code, derived data and a SHA-256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIXED_TIME = (2026, 11, 5, 0, 0, 0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_under(path: Path):
    if path.is_file():
        yield path
        return
    if path.exists():
        for item in sorted(path.rglob("*")):
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                yield item


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name.replace("\\", "/"), FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def build_overleaf() -> Path:
    output = PAPER / "merged_overleaf.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for path in files_under(PAPER / "merged"):
            if path.suffix in {".aux", ".bbl", ".blg", ".log", ".out"}:
                continue
            add_bytes(archive, path.relative_to(PAPER / "merged").as_posix(), path.read_bytes())
    return output


def build_reproducibility() -> Path:
    roots = [
        ROOT / "paper/REPRODUCIBILITY_PACKAGE_README.md",
        ROOT / "MERGED_PAPER_CLAIMS.md",
        ROOT / "LIC_RQ_V2_PROTOCOL.md",
        ROOT / "LIC_RQ_V2_STREAM_REPLAY_PROTOCOL.md",
        ROOT / "CDR_STREAM_ANALYSIS_PROTOCOL.md",
        ROOT / "CDR_STREAM_CONTROL_COMPLETION_PROTOCOL.md",
        ROOT / "CDR_STREAM_CONTROL_MONITORING_PROTOCOL.md",
        ROOT / "CDR_STREAM_CONTROL_EARLY_SIGNAL_ADDENDUM.md",
        ROOT / "ADWIN_DEFECT_AUDIT.md",
        ROOT / "paper/merged",
        ROOT / "src/reasoning_monitor",
        ROOT / "configs/lost_conversation_v2_science_asu.yaml",
        ROOT / "configs/cdr_stream_v1_main_asu.yaml",
        ROOT / "configs/cdr_stream_v1_analysis.yaml",
        ROOT / "scripts/paper_lic_rq_v2_assets.py",
        ROOT / "scripts/paper_merged_cdr_assets.py",
        ROOT / "scripts/paper_merged_supplement_assets.py",
        ROOT / "scripts/analyze_cdr_stream_control_completion.py",
        ROOT / "scripts/analyze_cdr_stream_control.py",
        ROOT / "scripts/control_early_signal.py",
        ROOT / "scripts/control_aggregated_monitoring.py",
        ROOT / "results/lost_conversation_v2_science/confirmatory/assessment.json",
        ROOT / "results/lost_conversation_v2_science/crl_incremental_posthoc/assessment.json",
        ROOT / "results/lost_conversation_v2_stream_replay/summary.json",
        ROOT / "results/cdr_stream_v1_mappings_llama3.json",
        ROOT / "results/cdr_stream_v1_main/analysis_stage_a",
        ROOT / "results/cdr_stream_v1_control_completion/assessment.json",
        ROOT / "results/cdr_stream_v1_control_completion/paired_cq.csv",
        ROOT / "results/cdr_stream_v1_control_monitoring/gpt_screen",
        ROOT / "results/cdr_stream_v1_control_monitoring/validation",
        ROOT / "results/cdr_stream_v1_power/power_grid.csv",
        ROOT / "results/cdr_stream_v1_power/adwin_v/power_grid.csv",
    ]
    selected: dict[str, Path] = {}
    for root in roots:
        for path in files_under(root):
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix in {".aux", ".bbl", ".blg", ".log", ".out"}:
                continue
            selected[rel] = path
    manifest = {
        "created": "2026-11-05",
        "raw_model_responses_included": False,
        "files": {rel: {"sha256": sha(path), "bytes": path.stat().st_size}
                  for rel, path in sorted(selected.items())},
    }
    output = PAPER / "merged_reproducibility.zip"
    with zipfile.ZipFile(output, "w") as archive:
        add_bytes(archive, "MANIFEST.json", (json.dumps(manifest, indent=2) + "\n").encode())
        for rel, path in sorted(selected.items()):
            add_bytes(archive, rel, path.read_bytes())
    return output


if __name__ == "__main__":
    overleaf = build_overleaf()
    reproducibility = build_reproducibility()
    print(json.dumps({
        "overleaf": str(overleaf.relative_to(ROOT)),
        "overleaf_sha256": sha(overleaf),
        "reproducibility": str(reproducibility.relative_to(ROOT)),
        "reproducibility_sha256": sha(reproducibility),
    }, indent=2))
