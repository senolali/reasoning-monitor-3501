"""Call-free preflight for the frozen LIC-RQ-v2 48-task science campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.dialogue_six_metrics import StrictScorers  # noqa: E402
from reasoning_monitor.lost_conversation import sha256_file  # noqa: E402
from reasoning_monitor.lost_conversation_science import load_science_plan  # noqa: E402

CONFIG = ROOT / "configs/lost_conversation_v2_science_asu.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-scorers", action="store_true",
                        help="load cached real NLI and BERTScore models; no ASU calls")
    args = parser.parse_args(argv)
    config, _manifest, _tasks, task_ids, report = load_science_plan(ROOT, CONFIG)
    report["config_sha256"] = sha256_file(CONFIG)
    report["science_task_ids"] = task_ids
    if args.real_scorers:
        print("Loading real local NLI and BERTScore scorers; no ASU calls...", flush=True)
        scorers = StrictScorers(with_ss=True)
        report["real_scorers"] = {
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "nli_revision": getattr(scorers.nli.model.config, "_commit_hash", None),
            "bertscore_model": "distilbert-base-uncased",
            "nli_loaded": scorers.nli is not None,
            "bertscore_loaded": scorers.bert is not None,
        }
        report["checks"]["real_scorers"] = bool(scorers.nli is not None
                                                   and scorers.bert is not None)
        report["execution_ready"] = all(report["checks"].values())
    output = ROOT / config["preflight_output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "assessment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    authorization = ("authorized" if report["execution_authorized"] else "locked")
    print(f"No ASU model call was made. Science execution is {authorization}.", flush=True)
    return 0 if report["execution_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
