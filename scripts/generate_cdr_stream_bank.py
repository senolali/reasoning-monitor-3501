"""Generate the draft CDR-Stream bank, run the call-free G0 checks, and write a
human-review sample plus a real-length budget estimate.

The bank is a DRAFT: it is not frozen and authorizes no model call. Concrete
text uses the review-only reference mapping, not a G1 calibration result.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning_monitor.data import cdr_stream as cs  # noqa: E402

BANK_PATH = ROOT / "configs/selections/cdr_stream_v1_draft_bank.json"
OUTPUT = ROOT / "results/cdr_stream_v1_draft"
REVIEW_FRACTION = 0.05
REVIEW_SEED = 20261004
G1_TURNS_PER_CELL = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    print("Generating bank twice for determinism...", flush=True)
    bank = cs.generate_bank()
    digest = cs.bank_sha256(bank)
    deterministic = digest == cs.bank_sha256(cs.generate_bank())
    cs.write_bank(BANK_PATH, bank)

    problems: list[str] = []
    for skeleton in bank["dialogues"]:
        problems += cs.validate_skeleton(skeleton)

    print("Instantiating all dialogues with the reference mapping...", flush=True)
    instances = {}
    stats_rows = []
    for index, skeleton in enumerate(bank["dialogues"], start=1):
        instance = cs.instantiate(skeleton, cs.REFERENCE_MAPPING)
        found, stats = cs.validate_instance(instance)
        problems += found
        instances[skeleton["dialogue_id"]] = instance
        stats_rows.append({"dialogue_id": skeleton["dialogue_id"], **stats})
        if index % 20 == 0 or index == len(bank["dialogues"]):
            print(f"  instantiated {index}/{len(bank['dialogues'])}", flush=True)

    print("Instantiating one dialogue per template under all 56 mappings...", flush=True)
    mapping_checks = 0
    for template in cs.TEMPLATES:
        skeleton = next(d for d in bank["dialogues"]
                        if d["template"] == template and d["scenario"] == "S1")
        for mapping in cs.all_mappings():
            found, _ = cs.validate_instance(cs.instantiate(skeleton, mapping))
            problems += [f"[mapping {mapping}] {p}" for p in found]
            mapping_checks += 1

    # Descriptive statistics under the reference mapping.
    level_by_regime: dict[str, Counter] = defaultdict(Counter)
    family_counts = Counter()
    dependency = []
    gold_by_template: dict[str, list[int]] = defaultdict(list)
    turns_total = 0
    for instance in instances.values():
        for turn in instance["turns"]:
            level_by_regime[turn["regime"]][turn["level"]] += 1
            family_counts[turn["family"]] += 1
            dependency.append(turn["dependency_distance"])
            gold_by_template[instance["template"]].append(turn["gold"])
            turns_total += 1

    def quantiles(values):
        ordered = sorted(values)
        return {q: ordered[int(q * (len(ordered) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)}

    # Budget: package A uses the core holdout, 2 dialogues per cell.
    package = cs.package_a_dialogues(bank, per_cell=2)
    budget = {}
    for k in (3, 5):
        totals = Counter()
        final_context = []
        for skeleton in package:
            est = cs.estimate_dialogue_tokens(instances[skeleton["dialogue_id"]], k, 2, 1)
            for key in ("calls", "input_tokens", "output_tokens_assumed"):
                totals[key] += est[key]
            final_context.append(est["final_turn_context_tokens"])
        budget[f"K{k}_L2_control1"] = {
            "dialogues_per_model": len(package),
            "calls_per_model": int(totals["calls"]),
            "calls_three_models": int(3 * totals["calls"]),
            "input_tokens_per_model": int(totals["input_tokens"]),
            "input_tokens_three_models": int(3 * totals["input_tokens"]),
            "output_tokens_assumed_three_models": int(3 * totals["output_tokens_assumed"]),
            "final_turn_context_tokens_max": int(max(final_context)),
            "final_turn_context_tokens_median": int(statistics.median(final_context)),
        }
    g1_calls = len(cs.LEVELS) * len(cs.TEMPLATES) * G1_TURNS_PER_CELL
    single_turn_chars = [len(t["state_summary_before"]) + len(t["text"])
                         + cs.SYSTEM_CHARS + cs.CONTRACT_CHARS
                         for inst in instances.values() for t in inst["turns"][1:]]
    budget["G1_single_turn"] = {
        "calls_per_model": g1_calls, "calls_three_models": 3 * g1_calls,
        "input_tokens_three_models": int(3 * g1_calls * statistics.mean(single_turn_chars)
                                         / cs.CHARS_PER_TOKEN)}

    # Human-review sample (D6): a random 5% of all turns.
    rng = random.Random(REVIEW_SEED)
    all_turns = [(inst, turn) for inst in instances.values() for turn in inst["turns"]]
    sample = rng.sample(all_turns, k=round(REVIEW_FRACTION * len(all_turns)))
    sample.sort(key=lambda item: (item[0]["dialogue_id"], item[1]["t"]))
    with (output / "review_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dialogue_id", "template", "scenario", "t", "regime", "level",
                         "family", "verbosity", "noise", "state_before", "turn_text",
                         "paraphrase_1", "paraphrase_2", "gold",
                         "REVIEW_gold_correct(Y/N)", "REVIEW_paraphrases_equivalent(Y/N)",
                         "REVIEW_natural_wording(1-5)", "REVIEW_comment"])
        for inst, turn in sample:
            writer.writerow([inst["dialogue_id"], inst["template"], inst["scenario"],
                             turn["t"], turn["regime"], turn["level"], turn["family"],
                             turn["verbosity"], turn["noise"], turn["state_summary_before"],
                             turn["text"], turn["paraphrases"][0], turn["paraphrases"][1],
                             turn["gold"], "", "", "", ""])

    # A readable excerpt of one S1 dialogue around its change points.
    example = instances[next(d["dialogue_id"] for d in bank["dialogues"]
                             if d["scenario"] == "S1" and d["template"] == "budget_ledger")]
    lines = [f"# Example CDR-Stream dialogue: {example['dialogue_id']}", "",
             "Review-only reference mapping " + json.dumps(cs.REFERENCE_MAPPING)
             + ". Not a G1 result.", ""]
    for turn in example["turns"]:
        if turn["t"] in (1, 2, 3, 32, 33, 34, 35, 66, 67, 68, 69, 99, 100):
            lines += [f"## Turn {turn['t']} (regime {turn['regime']}, level {turn['level']}, "
                      f"{turn['family']}, dependency distance {turn['dependency_distance']})",
                      "", turn["text"], "", f"Gold: **{turn['gold']}**", "",
                      f"Paraphrase 1: {turn['paraphrases'][0]}", ""]
    (output / "example_dialogue.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "protocol_id": cs.PROTOCOL_ID, "status": cs.BANK_STATUS,
        "bank_path": str(BANK_PATH.relative_to(ROOT)), "bank_sha256": digest,
        "deterministic": deterministic, "dialogues": len(bank["dialogues"]),
        "turns": turns_total,
        "dialogues_by_scenario": Counter(d["scenario"] for d in bank["dialogues"]),
        "dialogues_by_split": Counter(f"{'core' if d['core'] else 'secondary'}:{d['split']}"
                                      for d in bank["dialogues"]),
        "reference_mapping": cs.REFERENCE_MAPPING,
        "mapping_instantiations_checked": mapping_checks,
        "g0_pass": deterministic and not problems,
        "problems": problems[:200], "problem_count": len(problems),
        "level_share_by_regime": {r: {str(k): v / sum(c.values()) for k, v in sorted(c.items())}
                                  for r, c in level_by_regime.items()},
        "family_share": {k: v / turns_total for k, v in family_counts.items()},
        "dependency_distance_quantiles": quantiles(dependency),
        "gold_quantiles_by_template": {k: quantiles(v) for k, v in gold_by_template.items()},
        "guessability": {
            "gold_in_current_text_rate_mean": statistics.mean(
                r["gold_in_current_text_rate"] for r in stats_rows),
            "max_single_gold_share_max": max(r["max_single_gold_share"] for r in stats_rows),
        },
        "paraphrase_similarity_mean": statistics.mean(
            r["paraphrase_similarity_mean"] for r in stats_rows),
        "budget_estimate_char4_proxy": budget,
        "budget_assumptions": {"system_chars": cs.SYSTEM_CHARS,
                               "contract_chars": cs.CONTRACT_CHARS,
                               "assumed_reply_chars": cs.ASSUMED_REPLY_CHARS,
                               "chars_per_token": cs.CHARS_PER_TOKEN},
        "review_sample_rows": len(sample),
    }
    (output / "g0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("bank_sha256", "deterministic", "dialogues",
                                             "turns", "g0_pass", "problem_count")}, indent=2))
    print(json.dumps(budget, indent=2))
    return 0 if report["g0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
