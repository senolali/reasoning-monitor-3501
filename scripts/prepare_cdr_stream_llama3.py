"""Prepare CDR-Stream manifests bound to the Llama-3 70B mappings (call-free).

After the investigator's deviation (DEVIATIONS_FROM_PROPOSAL, 2026-10-03),
Llama-3 70B replaces Llama-4 Scout. This script leaves every original
preparation artifact untouched and writes new manifests under
``results/cdr_stream_v1_prep_llama3/``:

- ``manifest_pilot.json``: one Llama-3 70B technical-pilot unit, the same
  calibration-split S1 dialogue skeleton that Scout's pilot used
  (inventory_logistics, index 0), instantiated with the 70B mapping.
- ``manifest_main.json``: the main holdout sample for all three core models.
  GPT-4o-mini and Gemma 3 units must be byte-identical to the original main
  manifest; the script fails otherwise.

The frozen v2 bank file is read, not regenerated. No model call is made.
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
BANK_FILE_SHA256 = "bf3799d4d6fda5dbc354b54a9627bd4745160c2e097f160ece60bd9154195e53"
MAPPINGS_PATH = ROOT / "results/cdr_stream_v1_mappings_llama3.json"
MAPPINGS_SHA256 = "4ab2af3d7af6deff18af72baf3cd349bd30bee133ee4185ca33c8e6efea8a68b"
ORIGINAL_MAIN = ROOT / "results/cdr_stream_v1_prep/manifest_main.json"
OUTPUT = ROOT / "results/cdr_stream_v1_prep_llama3"
MODELS = ("gpt4o_mini", "llama3_70b", "gemma3_12b_it")
CARRIED_OVER = ("gpt4o_mini", "gemma3_12b_it")
CORE_SCENARIOS = tuple(s for s, spec in cs.SCENARIOS.items() if spec["core"])
PILOT_TEMPLATES = {"llama3_70b": "inventory_logistics"}
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
    for path, expected in ((BANK_PATH, BANK_FILE_SHA256), (MAPPINGS_PATH, MAPPINGS_SHA256)):
        if sha256_file(path) != expected:
            print(f"Frozen input hash mismatch: {path}")
            return 1
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    mappings = json.loads(MAPPINGS_PATH.read_text(encoding="utf-8"))
    skeletons = {d["dialogue_id"]: d for d in bank["dialogues"]}

    sets = {
        "pilot": runner.select_units(bank, mappings, ["llama3_70b"], split="calibration",
                                     scenarios=("S1",), per_model=1, indices=[0],
                                     templates=PILOT_TEMPLATES),
        "main": runner.select_units(bank, mappings, MODELS, split="holdout",
                                    scenarios=CORE_SCENARIOS, per_model=MAIN_PER_MODEL),
    }
    problems: list[str] = []
    report = {"bank_path": str(BANK_PATH.relative_to(ROOT)), "bank_file_sha256": BANK_FILE_SHA256,
              "mappings_path": str(MAPPINGS_PATH.relative_to(ROOT)),
              "mappings_sha256": MAPPINGS_SHA256, "prompt_sha256": runner.prompt_sha256(),
              "k": K, "l": L, "control": CONTROL, "sets": {}}
    manifests = {}
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
        manifests[name] = manifest
        manifest_hash = write_json(OUTPUT / f"manifest_{name}.json",
                                   {"set": name, "bank_sha256": cs.bank_sha256(bank),
                                    "mappings_sha256": MAPPINGS_SHA256, "units": manifest})
        report["sets"][name] = {
            "manifest_path": str((OUTPUT / f"manifest_{name}.json").relative_to(ROOT)),
            "manifest_sha256": manifest_hash, "units": len(manifest),
            "units_by_model_template": Counter(f"{u['model']}|{u['template']}"
                                               for u in manifest),
            "units_by_scenario": Counter(u["scenario"] for u in manifest),
            "budget_by_model": {m: {**b, "input_tokens": int(b["input_tokens"]),
                                    "max_context_tokens": int(b["max_context_tokens"])}
                                for m, b in budget.items()},
            "budget_total": {"calls": sum(b["calls"] for b in budget.values()),
                             "input_tokens": int(sum(b["input_tokens"]
                                                     for b in budget.values()))}}

    original = json.loads(ORIGINAL_MAIN.read_text(encoding="utf-8"))["units"]
    carried = {m: [u for u in manifests["main"] if u["model"] == m] for m in CARRIED_OVER}
    report["carried_over_identical"] = {
        m: carried[m] == [u for u in original if u["model"] == m] for m in CARRIED_OVER}
    if not all(report["carried_over_identical"].values()):
        problems.append("GPT/Gemma main units differ from the original main manifest")
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
