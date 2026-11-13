"""Call-free tests for LHE-RQ-v1 (no network, no GPU, no model call)."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from reasoning_monitor import lhe_acquisition as acq
from reasoning_monitor import lhe_early_warning as lhe

DATA = Path(__file__).resolve().parents[1] / "data/external/long_horizon_execution"
WORDS = ["apple", "brave", "cloud", "dream", "eagle", "flame", "grape", "house",
         "ivory", "jolly", "knife", "lemon"]


def toy_dataset(n: int = 6, length: int = 200, seed: int = 0):
    rng = random.Random(seed)
    dictionary = {w: rng.randint(-99, 99) for w in WORDS}
    samples = []
    for i in range(n):
        keys = [rng.choice(WORDS) for _ in range(length)]
        values = [dictionary[k] for k in keys]
        running, outputs = 0, []
        for v in values:
            running += v
            outputs.append(running)
        samples.append({"id": i, "input": keys, "values": values, "output": outputs})
    return dictionary, samples


@pytest.mark.skipif(not (DATA / "test.jsonl").exists(), reason="dataset not downloaded")
def test_real_dataset_hash_and_running_sums():
    dictionary, samples = lhe.load_dataset(DATA)
    assert len(samples) == 100 and len(dictionary) == 100
    for k in (1, 2, 4):
        conv = lhe.build_conversation(samples[0], k, 60)
        assert conv["turns"][-1]["true_sum"] == samples[0]["output"][60 * k - 1]


def test_build_conversation_detects_corrupted_gold():
    _, samples = toy_dataset(1)
    bad = {**samples[0], "output": list(samples[0]["output"])}
    bad["output"][3] += 1
    with pytest.raises(ValueError, match="mismatch"):
        lhe.build_conversation(bad, 1, 10)


@pytest.mark.parametrize("text,value,strict", [
    ("so 5 + 7 = 12. <answer>12</answer>", 12, True),
    ("<answer>-1,234</answer>", -1234, False),
    ("first <answer>3</answer> then <answer>4</answer>", 4, False),
    ("<answer>f10</answer>", 10, False),
    ("The total is 12.", None, False),
    ("<answer>unknown</answer>", None, False),
])
def test_parse_answer(text, value, strict):
    assert lhe.parse_answer(text) == (value, strict)


def test_demonstrations_are_arithmetically_correct():
    for k in (1, 2):
        prompt = lhe.system_prompt({"x": 1}, k)
        assert "f10" not in prompt
        answers = [lhe.parse_answer(f"<answer>{a}</answer>")[0]
                   for a in __import__("re").findall(r"<answer>(-?\d+)</answer>", prompt)]
        expected = [5, 5, 12, 8] if k == 1 else [5, 8, 9]
        assert answers == expected


def test_paraphrases_keep_keys_and_differ():
    texts = [lhe.turn_text(["apple", "grape"], v) for v in range(3)]
    assert len(set(texts)) == 3 and all("apple, grape" in t for t in texts)


def test_onset_and_healthy_start():
    step = [1.0] * 12 + [0, 1, 0, 0, 1] + [0.0] * 10
    assert lhe.healthy_start(step)
    assert lhe.decline_onset(step) == 13
    assert lhe.decline_onset([1.0] * 40) is None
    assert not lhe.healthy_start([1, 0, 0, 1, 1] + [1.0] * 10)


def test_cusum_detects_planted_shift_and_is_quiet_on_noise():
    rng = random.Random(1)
    flat = [0.9 + rng.gauss(0, 0.02) for _ in range(60)]
    shifted = flat[:25] + [v - 0.3 for v in flat[25:]]
    rule = lhe.MonitorRule("CS", -1, 0.05, 4.0)
    assert lhe.cusum_alarm(flat, rule) is None
    alarm = lhe.cusum_alarm(shifted, rule)
    assert alarm is not None and 26 <= alarm <= 28
    assert lhe.cusum_alarm([math.nan] * 60, rule) is None


def test_classify():
    assert lhe.classify(None, None) == "quiet"
    assert lhe.classify(5, None) == "false_alarm"
    assert lhe.classify(10, 25) == "false_alarm"
    assert lhe.classify(20, 25) == "detection"
    assert lhe.classify(40, 25) == "late"
    assert lhe.classify(None, 25) == "miss"


def run_simulation(tmp_path, n=8, horizon=40, onset=18):
    dictionary, samples = toy_dataset(n)
    models = [{"name": "sim", "provider_model": "sim/model", "k": 1}]
    units = acq.build_units(models, samples, list(range(n)), horizon)
    client = lhe.SimulatedLHEClient(dictionary, onset_turn=onset, error_rate_after=0.7)
    ckpt = tmp_path / "records.jsonl"
    result = acq.acquire(units, dictionary, {"sim": client}, ckpt, "fp", 1024,
                         k_samples=3, l_paraphrases=2, workers=4,
                        tokens_per_minute=10**9)
    return dictionary, units, ckpt, result, client


def test_simulated_acquisition_resumes_and_scores(tmp_path):
    dictionary, units, ckpt, result, client = run_simulation(tmp_path)
    assert result == {"expected": 8 * 40 * 5, "completed": 1600, "missing": 0}
    again = acq.acquire(units, dictionary, {"sim": client}, ckpt, "fp", 1024,
                        k_samples=3, l_paraphrases=2, workers=4,
                        tokens_per_minute=10**9)
    assert again["completed"] == 1600 and ckpt.read_text().count("\n") == 1600
    saved, _ = acq.load_checkpoint(ckpt, "fp")
    rows = [r for r in saved.values() if r["dialogue_id"] == units[0]["dialogue_id"]]
    scored = lhe.score_conversation(rows, units[0]["conversation"], 3, 2)
    early = [r for r in scored if r["turn"] < 18]
    assert all(r["step_correct"] == 1.0 and r["CS"] == 1.0 for r in early)
    assert any(r["step_correct"] == 0.0 for r in scored if r["turn"] >= 18)


def test_pipeline_detects_planted_consistency_signal(tmp_path):
    _, units, ckpt, _, _ = run_simulation(tmp_path, n=8, horizon=40, onset=18)
    saved, _ = acq.load_checkpoint(ckpt, "fp")
    convs = []
    for unit in units:
        rows = [r for r in saved.values() if r["dialogue_id"] == unit["dialogue_id"]]
        scored = lhe.score_conversation(rows, unit["conversation"], 3, 2)
        step = [r["step_correct"] for r in scored]
        convs.append({"tau": lhe.decline_onset(step),
                      "signals": {"CS": [r["CS"] for r in scored],
                                  "STEP": [r["step_correct"] for r in scored]}})
    assert all(c["tau"] is not None and c["tau"] >= 18 for c in convs)
    rule = lhe.MonitorRule("CS", -1, 0.1, 2.0)
    result = lhe.evaluate(rule, convs)
    assert result["detection_rate"] >= 0.75 and result["false_alarm_rate"] == 0.0


def test_budget_estimate_grows_with_history():
    dictionary, samples = toy_dataset(1)
    conv = lhe.build_conversation(samples[0], 1, 20)
    small = lhe.estimate_conversation_tokens(dictionary, conv, 7, 150)
    conv40 = lhe.build_conversation(samples[0], 1, 40)
    large = lhe.estimate_conversation_tokens(dictionary, conv40, 7, 150)
    assert small["calls"] == 140 and large["input_tokens"] > 2 * small["input_tokens"]


def test_direct_mode_prompt_and_unchanged_cot_hash():
    d = {"apple": 5}
    direct = lhe.pack_messages(lhe.system_prompt(d, 4, "direct"), [], "apple", "direct")
    assert "Do not output any other text" in direct[1]["content"]
    assert "think step by step" not in direct[0]["content"].lower()
    assert lhe.prompt_sha256(d, 4, "cot") == lhe.prompt_sha256(d, 4)
    assert lhe.prompt_sha256(d, 4, "direct") != lhe.prompt_sha256(d, 4)
    with pytest.raises(ValueError):
        lhe.contract("other")
