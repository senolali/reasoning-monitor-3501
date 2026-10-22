"""Prepare the CDR-Stream study call-free: v2 bank, unit manifests, G0, budget.

Writes the v2 skeleton bank (final templates, G1 v2 ladder), selects pilot and
main units from the frozen G1 mappings, instantiates every unit, runs the G0
checks on each instance, and stores per-unit instance hashes (not the texts,
which are regenerated deterministically by the runner) plus a real-length
budget estimate. No model call is made.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor import cdr_stream_runner as runner  # noqa: E402
from reasoning_monitor.data import cdr_stream as cs  # noqa: E402

BANK_PATH = ROOT / "configs/selections/cdr_stream_v1_bank_v2.json"
MAPPINGS_PATH = ROOT / "results/cdr_stream_v1_mappings.json"
MAPPINGS_SHA256 = "9a63512bde255b96394d3dc31ca365daf57644ee5e38e0f80b95b9a867db6571"
OUTPUT = ROOT / "results/cdr_stream_v1_prep"
MODELS = ("gpt4o_mini", "llama4_scout", "gemma3_12b_it")
CORE_SCENARIOS = tuple(s for s, spec in cs.SCENARIOS.items() if spec["core"])
# Pilot: one calibration-split S1 staircase dialogue per model, templates spread
# so that every final template except lab is exercised once.
PILOT_TEMPLATES = {"gpt4o_mini": "milestone_chains", "llama4_scout": "inventory_logistics",
                   "gemma3_12b_it": "budget_ledger"}
MAIN_PER_MODEL = 30
K, L, CONTROL = 3, 2, 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    if sha256_file(MAPPINGS_PATH) != MAPPINGS_SHA256:
        print("Frozen mappings hash mismatch")
        return 1
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))

    bank = cs.generate_bank(version="v2")
    bank_hash = cs.bank_sha256(bank)
    if bank_hash != cs.bank_sha256(cs.generate_bank(version="v2")):
        print("Bank generation is not deterministic")
        return 1
    cs.write_bank(BANK_PATH, bank)
    problems = [p for d in bank["dialogues"] for p in cs.validate_skeleton(d)]

    sets = {
        "pilot": runner.select_units(bank, mappings, MODELS, split="calibration",
                                     scenarios=("S1",), per_model=1, indices=[0],
                                     templates=PILOT_TEMPLATES),
        "main": runner.select_units(bank, mappings, MODELS, split="holdout",
                                    scenarios=CORE_SCENARIOS, per_model=MAIN_PER_MODEL),
    }
    skeletons = {d["dialogue_id"]: d for d in bank["dialogues"]}
    report = {"bank_path": str(BANK_PATH.relative_to(ROOT)), "bank_sha256": bank_hash,
              "mappings_sha256": MAPPINGS_SHA256, "prompt_sha256": runner.prompt_sha256(),
              "k": K, "l": L, "control": CONTROL, "sets": {}}
    for name, units in sets.items():
        manifest = []
        budget = defaultdict(lambda: {"calls": 0, "input_tokens": 0.0,
                                      "max_context_tokens": 0.0, "dialogues": 0})
        for unit in units:
            instance = cs.instantiate(skeletons[unit["dialogue_id"]], unit["mapping"], "v2")
            found, _ = cs.validate_instance(instance)
            problems += [f"[{unit['model']}] {p}" for p in found]
            manifest.append({**unit, "instance_sha256": runner.instance_sha256(instance)})
            est = runner.estimate_unit_budget(instance, K, L, CONTROL)
            model_budget = budget[unit["model"]]
            model_budget["calls"] += est["calls"]
            model_budget["input_tokens"] += est["input_tokens"]
            model_budget["max_context_tokens"] = max(model_budget["max_context_tokens"],
                                                     est["max_context_tokens"])
            model_budget["dialogues"] += 1
        manifest_hash = write_json(OUTPUT / f"manifest_{name}.json",
                                   {"set": name, "bank_sha256": bank_hash,
                                    "mappings_sha256": MAPPINGS_SHA256, "units": manifest})
        totals = {"calls": sum(b["calls"] for b in budget.values()),
                  "input_tokens": int(sum(b["input_tokens"] for b in budget.values()))}
        report["sets"][name] = {
            "manifest_path": str((OUTPUT / f"manifest_{name}.json").relative_to(ROOT)),
            "manifest_sha256": manifest_hash, "units": len(manifest),
            "units_by_model_template": Counter(f"{u['model']}|{u['template']}"
                                               for u in manifest),
            "units_by_scenario": Counter(u["scenario"] for u in manifest),
            "budget_by_model": {m: {**b, "input_tokens": int(b["input_tokens"]),
                                    "max_context_tokens": int(b["max_context_tokens"])}
                                for m, b in budget.items()},
            "budget_total": totals}
    report["g0_pass"] = not problems
    report["problem_count"] = len(problems)
    report["problems"] = problems[:100]
    write_json(OUTPUT / "prep_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, indent=2))
    for name, data in report["sets"].items():
        print(f"{name}: {data['units']} units, manifest {data['manifest_sha256'][:12]}, "
              f"calls {data['budget_total']['calls']}, "
              f"input tokens {data['budget_total']['input_tokens']:,}")
        for model, b in data["budget_by_model"].items():
            print(f"   {model}: {b['dialogues']} dialogues, {b['calls']} calls, "
                  f"{b['input_tokens']:,} input tokens, max context {b['max_context_tokens']:,}")
    return 0 if report["g0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
